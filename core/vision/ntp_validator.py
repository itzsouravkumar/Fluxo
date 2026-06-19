from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import ntplib


@dataclass
class NTPSyncStatus:
    synced: bool
    offset_sec: float
    last_sync_time: float
    drift_sec: float
    server: str


class NTPValidator:
    """Validates NTP timestamp synchronization for evidence chain integrity.

    Ensures edge camera nodes are synced to BTP's internal NTP server.
    If NTP sync age > 60 seconds, violation timestamps are flagged as
    'approximate' and routed to human review.

    Reference: Vol 9, EC-L1 — timestamp manipulation / NTP drift.
    """

    def __init__(
        self,
        ntp_servers: list[str] | None = None,
        max_sync_age_sec: float = 60.0,
        drift_threshold_sec: float = 0.5,
    ):
        self.ntp_servers = ntp_servers or [
            "pool.ntp.org",
            "time.google.com",
            "time.cloudflare.com",
        ]
        self.max_sync_age_sec = max_sync_age_sec
        self.drift_threshold_sec = drift_threshold_sec
        self._last_sync: NTPSyncStatus | None = None
        self._local_offset: float = 0.0

    def sync(self) -> NTPSyncStatus:
        client = ntplib.NTPClient()
        for server in self.ntp_servers:
            try:
                response = client.request(server, version=3)
                self._local_offset = response.offset
                self._last_sync = NTPSyncStatus(
                    synced=True,
                    offset_sec=response.offset,
                    last_sync_time=time.time(),
                    drift_sec=abs(response.offset),
                    server=server,
                )
                return self._last_sync
            except Exception:
                continue

        self._last_sync = NTPSyncStatus(
            synced=False,
            offset_sec=0.0,
            last_sync_time=time.time(),
            drift_sec=0.0,
            server="none",
        )
        return self._last_sync

    def get_accurate_timestamp(self) -> str:
        now = time.time() + self._local_offset
        return datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def is_timestamp_reliable(self) -> bool:
        if self._last_sync is None:
            return False
        if not self._last_sync.synced:
            return False
        age = time.time() - self._last_sync.last_sync_time
        return age < self.max_sync_age_sec

    def validate_violation_timestamp(self, violation_timestamp: str) -> dict:
        reliable = self.is_timestamp_reliable()
        needs_review = False
        flag_reason = None

        if self._last_sync is None:
            needs_review = True
            flag_reason = "NTP not synced"
        elif not self._last_sync.synced:
            needs_review = True
            flag_reason = "NTP sync failed"
        elif time.time() - self._last_sync.last_sync_time > self.max_sync_age_sec:
            needs_review = True
            flag_reason = f"NTP sync age {time.time() - self._last_sync.last_sync_time:.0f}s > {self.max_sync_age_sec}s"
        elif self._last_sync.drift_sec > self.drift_threshold_sec:
            needs_review = True
            flag_reason = f"NTP drift {self._last_sync.drift_sec:.3f}s > {self.drift_threshold_sec}s"

        return {
            "timestamp": violation_timestamp,
            "reliable": reliable,
            "needs_human_review": needs_review,
            "flag_reason": flag_reason,
            "ntp_server": self._last_sync.server if self._last_sync else "none",
            "drift_sec": self._last_sync.drift_sec if self._last_sync else 0.0,
        }
