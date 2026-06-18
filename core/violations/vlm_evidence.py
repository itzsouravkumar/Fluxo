from __future__ import annotations

from datetime import datetime

import numpy as np


class VLMEvidenceLayer:
    """Lightweight VLM for generating legal violation narratives.

    Uses a small vision-language model (MobileVLM v2 or Qwen2-VL-2B)
    to generate human-readable violation descriptions from detection
    context and violation clip frames.

    Triggered only post-confirmation (not per-frame) to avoid latency
    in the real-time pipeline. Runs asynchronously on the violation clip.

    Reference: SafePLUG (CVPR 2025), arXiv:2508.06763
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self._model = None
        self._processor = None
        self._model_name = model_name
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self._available = True
        except Exception:
            self._available = False

    def narrate(
        self,
        violation,
        clip_frames: list[np.ndarray] | None = None,
        current_frame: np.ndarray | None = None,
    ) -> str | None:
        if not self._available or self._model is None:
            return self._template_narrate(violation)

        try:
            prompt = self._build_prompt(violation)
            if current_frame is not None and self._processor is not None:
                from PIL import Image
                import cv2
                rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                inputs = self._processor(
                    text=prompt,
                    images=pil_img,
                    return_tensors="pt",
                ).to(self._model.device)

                output = self._model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                )
                response = self._processor.decode(output[0], skip_special_tokens=True)
                return response.split("Violation:")[-1].strip() if "Violation:" in response else response.strip()
        except Exception:
            pass

        return self._template_narrate(violation)

    def _build_prompt(self, violation) -> str:
        parts = [
            f"Traffic violation detected at {datetime.now().strftime('%H:%M:%S IST')}.",
            f"Type: {violation.type.value.replace('_', ' ').title()}.",
            f"Confidence: {violation.confidence:.0%}.",
            f"Track ID: {violation.track_id}.",
        ]
        if violation.plate_number:
            parts.append(f"Plate: {violation.plate_number}.")
        if violation.bbox != (0, 0, 0, 0):
            parts.append(f"Location in frame: bbox {violation.bbox}.")
        parts.append("Generate a formal violation description for an e-challan.")
        return " ".join(parts)

    def _template_narrate(self, violation) -> str:
        vtype = violation.type.value.replace("_", " ").title()
        ts = datetime.now().strftime("%H:%M:%S IST")
        parts = [f"Vehicle observed committing {vtype.lower()} at {ts}."]
        if violation.plate_number:
            parts.append(f"Plate number: {violation.plate_number}.")
        parts.append(f"Violation confidence: {violation.confidence:.0%}.")
        parts.append("Recorded via automated CCTV surveillance system.")
        return " ".join(parts)
