from __future__ import annotations

import base64
import json
import re
import urllib.request
import urllib.error
from datetime import datetime

import numpy as np


class VLMEvidenceLayer:
    """Vision-Language Model for generating legal violation narratives.

    Uses Qwen2.5-VL via Ollama for local inference on Mac (Metal GPU).
    Falls back to HuggingFace Transformers on remote servers.
    If both fail, uses a structured template for evidence text.

    Reference: SafePLUG (CVPR 2025), arXiv:2508.06763
    """

    DEFAULT_MODEL = "qwen2.5vl:3b"
    OLLAMA_BASE = "http://localhost:11434"

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._available = False
        self._load_error = None
        self._backend = None
        self._try_load()

    def _try_load(self):
        if self._try_load_ollama():
            return
        if "Ollama running" in (self._load_error or ""):
            return
        self._try_load_transformers()

    def _try_load_ollama(self):
        try:
            req = urllib.request.Request(
                f"{self.OLLAMA_BASE}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                if any(self._model_name in m for m in models):
                    self._available = True
                    self._backend = "ollama"
                    return True
                if models:
                    self._model_name = models[0]
                    self._available = True
                    self._backend = "ollama"
                    return True
            self._load_error = f"Ollama running but model '{self._model_name}' not found. Run: ollama pull {self._model_name}"
            return False
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            self._load_error = "Ollama not running. Start with: ollama serve"
            return False

    def _try_load_transformers(self):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            model_id = "Qwen/Qwen2-VL-2B-Instruct"
            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True,
            )
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            self._available = True
            self._backend = "transformers"
            return True
        except Exception as e:
            if not self._load_error:
                self._load_error = str(e)
            return False

    @property
    def status(self) -> dict:
        return {
            "available": self._available,
            "backend": self._backend,
            "model_name": self._model_name,
            "error": self._load_error,
        }

    def narrate(
        self,
        violation,
        clip_frames: list[np.ndarray] | None = None,
        current_frame: np.ndarray | None = None,
        detection_context: dict | None = None,
    ) -> str | None:
        if not self._available:
            return self._template_narrate(violation)

        prompt = self._build_prompt(violation)

        if self._backend == "ollama" and current_frame is not None:
            result = self._narrate_ollama(prompt, current_frame)
            if result:
                return self._validate_narration(result, violation)

        if self._backend == "transformers" and current_frame is not None:
            result = self._narrate_transformers(prompt, current_frame)
            if result:
                return self._validate_narration(result, violation)

        return self._template_narrate(violation)

    def _narrate_ollama(self, prompt: str, frame: np.ndarray) -> str | None:
        try:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            payload = json.dumps({
                "model": self._model_name,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 200,
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.OLLAMA_BASE}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "").strip()
        except Exception:
            return None

    def _narrate_transformers(self, prompt: str, frame: np.ndarray) -> str | None:
        try:
            import torch
            import cv2
            from PIL import Image

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text],
                images=[pil_img],
                return_tensors="pt",
                padding=True,
            )

            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.7,
                )

            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            return self._processor.decode(generated, skip_special_tokens=True).strip()
        except Exception:
            return None

    def _build_prompt(self, violation) -> str:
        safe_plate = "<PLATE>"
        if violation.plate_number:
            cleaned = re.sub(r'[^A-Z0-9]', '', violation.plate_number.upper())
            if self._is_safe_plate_text(cleaned):
                safe_plate = violation.plate_number

        parts = [
            "You are a traffic enforcement AI generating a legal violation description for an e-challan.",
            "Only state facts directly observable. Do not fabricate details.",
            f"Violation: {violation.type.value.replace('_', ' ')}.",
            f"Confidence: {violation.confidence:.0%}.",
            f"Plate: {safe_plate}.",
            f"Time: {datetime.now().strftime('%H:%M:%S IST')}.",
            "Generate a concise, formal violation description suitable for legal enforcement.",
        ]
        return " ".join(parts)

    def _validate_narration(self, narration: str, violation) -> str:
        if violation.plate_number:
            clean_plate = re.sub(r'[\s\-]', '', violation.plate_number.upper())
            narr_match = re.search(
                r'[A-Z]{2}[\s\-]?\d{1,2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{1,4}',
                narration.upper(),
            )
            if narr_match:
                narr_plate = re.sub(r'[\s\-]', '', narr_match.group())
                if narr_plate != clean_plate:
                    narration = narration.replace(narr_match.group(), violation.plate_number)

        vtype_words = violation.type.value.replace("_", " ").lower()
        if vtype_words.split()[0] not in narration.lower():
            narration = f"{violation.type.value.replace('_', ' ').title()}: {narration}"

        return narration

    def _is_safe_plate_text(self, text: str) -> bool:
        dangerous = ["instruct", "ignore", "system", "prompt", "override", "admin", "jailbreak"]
        lower = text.lower()
        return not any(d in lower for d in dangerous)

    def _template_narrate(self, violation) -> str:
        vtype = violation.type.value.replace("_", " ").title()
        ts = datetime.now().strftime("%H:%M:%S IST")
        parts = [f"Vehicle observed committing {vtype.lower()} at {ts}."]
        if violation.plate_number:
            parts.append(f"Plate number: {violation.plate_number}.")
        parts.append(f"Confidence: {violation.confidence:.0%}.")
        if violation.requires_human_review:
            parts.append("FLAGGED for human review.")
        if violation.vehicle_speed_kmh is not None:
            parts.append(f"Estimated speed: {violation.vehicle_speed_kmh:.0f} km/h.")
        parts.append("Recorded via automated CCTV surveillance.")
        return " ".join(parts)
