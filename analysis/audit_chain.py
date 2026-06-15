"""
analysis/audit_chain.py — Blockchain-style Audit Chain for Defect Detection
=============================================================================

Adapted from organoid-fl's blockchain audit and PCB-Defect-FL's existing
audit.rs for Python-side integration.

Provides tamper-proof audit trail for:
1. Detection results — who detected what, when, with what confidence
2. Model updates — when was the model updated, by which factory
3. Compliance — OPC-UA + audit chain = industrial compliance

Each audit entry is SHA-256 chained: entry_n.hash = SHA-256(entry_n.data + entry_{n-1}.hash)

Pure Python. No external dependencies. Streamlit Cloud compatible.
"""

import hashlib
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class AuditEventType(Enum):
    DETECTION = "detection"
    MODEL_UPDATE = "model_update"
    DATA_UPLOAD = "data_upload"
    THRESHOLD_CHANGE = "threshold_change"
    CLIENT_JOIN = "client_join"
    CLIENT_LEAVE = "client_leave"
    AGGREGATION = "aggregation"
    ALERT = "alert"
    MANUAL_REVIEW = "manual_review"


@dataclass
class AuditEntry:
    """Single entry in the audit chain."""
    index: int
    timestamp: str
    event_type: str
    actor: str  # who performed the action (factory_id, user_id, system)
    data: Dict[str, Any]
    previous_hash: str
    hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this entry."""
        content = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "actor": self.actor,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def is_valid(self) -> bool:
        """Check if this entry's hash is correct."""
        return self.hash == self.compute_hash()


@dataclass
class AuditChainVerification:
    """Result of verifying the entire audit chain."""
    is_valid: bool
    total_entries: int
    tampered_entries: List[int]
    missing_links: List[int]
    timestamp_gaps: List[Dict[str, Any]]
    summary: str


class DefectAuditChain:
    """Blockchain-style audit chain for defect detection operations.

    Every detection, model update, and configuration change is recorded
    in an immutable chain. Any tampering is detectable.

    Usage:
        chain = DefectAuditChain(factory_id="factory_A")
        chain.record_detection(defect_type="short_circuit", confidence=0.95, image_hash="abc123")
        chain.record_model_update(model_version="v2.1", accuracy=0.96)
        verification = chain.verify()
    """

    GENESIS_HASH = "0" * 64  # Genesis block has no predecessor

    def __init__(self, factory_id: str = "default"):
        self.factory_id = factory_id
        self.chain: List[AuditEntry] = []
        self._create_genesis()

    def _create_genesis(self):
        """Create the genesis (first) entry."""
        genesis = AuditEntry(
            index=0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type=AuditEventType.CLIENT_JOIN.value,
            actor=self.factory_id,
            data={"action": "chain_initialized", "factory_id": self.factory_id},
            previous_hash=self.GENESIS_HASH,
        )
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)

    @property
    def last_hash(self) -> str:
        return self.chain[-1].hash if self.chain else self.GENESIS_HASH

    @property
    def length(self) -> int:
        return len(self.chain)

    def _add_entry(self, event_type: AuditEventType, actor: str, data: Dict[str, Any]) -> AuditEntry:
        """Add a new entry to the chain."""
        entry = AuditEntry(
            index=len(self.chain),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type=event_type.value,
            actor=actor,
            data=data,
            previous_hash=self.last_hash,
        )
        entry.hash = entry.compute_hash()
        self.chain.append(entry)
        return entry

    def record_detection(
        self,
        defect_type: str,
        confidence: float,
        image_hash: str = "",
        bbox: Optional[List[float]] = None,
        model_version: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record a defect detection event."""
        data = {
            "defect_type": defect_type,
            "confidence": confidence,
            "image_hash": image_hash,
            "model_version": model_version,
            "factory_id": self.factory_id,
        }
        if bbox:
            data["bbox"] = bbox
        if metadata:
            data.update(metadata)
        return self._add_entry(AuditEventType.DETECTION, self.factory_id, data)

    def record_model_update(
        self,
        model_version: str,
        accuracy: float,
        training_samples: int = 0,
        fl_round: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record a model update event."""
        data = {
            "model_version": model_version,
            "accuracy": accuracy,
            "training_samples": training_samples,
            "fl_round": fl_round,
            "factory_id": self.factory_id,
        }
        if metadata:
            data.update(metadata)
        return self._add_entry(AuditEventType.MODEL_UPDATE, self.factory_id, data)

    def record_data_upload(
        self,
        num_images: int,
        defect_types: List[str],
        uploader: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record a data upload event."""
        data = {
            "num_images": num_images,
            "defect_types": defect_types,
            "uploader": uploader or self.factory_id,
        }
        if metadata:
            data.update(metadata)
        return self._add_entry(AuditEventType.DATA_UPLOAD, self.factory_id, data)

    def record_aggregation(
        self,
        fl_round: int,
        num_clients: int,
        strategy: str = "fedavg",
        conformity_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record a federated aggregation event."""
        data = {
            "fl_round": fl_round,
            "num_clients": num_clients,
            "strategy": strategy,
            "conformity_score": conformity_score,
        }
        if metadata:
            data.update(metadata)
        return self._add_entry(AuditEventType.AGGREGATION, "fl_server", data)

    def record_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record an alert event."""
        data = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
        }
        if metadata:
            data.update(metadata)
        return self._add_entry(AuditEventType.ALERT, "system", data)

    def record_manual_review(
        self,
        reviewer: str,
        detection_index: int,
        verdict: str,  # "confirmed" | "false_positive" | "needs_recheck"
        notes: str = "",
    ) -> AuditEntry:
        """Record a manual review of a detection result."""
        data = {
            "reviewer": reviewer,
            "detection_index": detection_index,
            "verdict": verdict,
            "notes": notes,
        }
        return self._add_entry(AuditEventType.MANUAL_REVIEW, reviewer, data)

    def verify(self) -> AuditChainVerification:
        """Verify the integrity of the entire audit chain.

        Checks:
        1. Each entry's hash matches its computed hash
        2. Each entry's previous_hash matches the preceding entry's hash
        3. Timestamps are monotonically increasing
        """
        tampered = []
        missing_links = []
        timestamp_gaps = []

        for i, entry in enumerate(self.chain):
            # Check hash integrity
            if not entry.is_valid():
                tampered.append(i)

            # Check chain linkage
            if i > 0:
                if entry.previous_hash != self.chain[i - 1].hash:
                    missing_links.append(i)

            # Check timestamp ordering
            if i > 0:
                prev_time = self.chain[i - 1].timestamp
                curr_time = entry.timestamp
                if curr_time < prev_time:
                    timestamp_gaps.append({
                        "index": i,
                        "prev_timestamp": prev_time,
                        "curr_timestamp": curr_time,
                    })

        is_valid = len(tampered) == 0 and len(missing_links) == 0

        if is_valid:
            summary = f"Chain is valid. {len(self.chain)} entries, no tampering detected."
        else:
            parts = []
            if tampered:
                parts.append(f"{len(tampered)} tampered entries (indices: {tampered[:5]})")
            if missing_links:
                parts.append(f"{len(missing_links)} broken links (indices: {missing_links[:5]})")
            summary = f"Chain integrity compromised: {'; '.join(parts)}"

        return AuditChainVerification(
            is_valid=is_valid,
            total_entries=len(self.chain),
            tampered_entries=tampered,
            missing_links=missing_links,
            timestamp_gaps=timestamp_gaps,
            summary=summary,
        )

    def get_detections(self, defect_type: Optional[str] = None) -> List[AuditEntry]:
        """Get all detection entries, optionally filtered by defect type."""
        results = [e for e in self.chain if e.event_type == AuditEventType.DETECTION.value]
        if defect_type:
            results = [e for e in results if e.data.get("defect_type") == defect_type]
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics from the audit chain."""
        detections = self.get_detections()
        model_updates = [e for e in self.chain if e.event_type == AuditEventType.MODEL_UPDATE.value]
        alerts = [e for e in self.chain if e.event_type == AuditEventType.ALERT.value]

        defect_counts: Dict[str, int] = {}
        confidence_by_type: Dict[str, List[float]] = {}

        for d in detections:
            dt = d.data.get("defect_type", "unknown")
            defect_counts[dt] = defect_counts.get(dt, 0) + 1
            conf = d.data.get("confidence", 0.0)
            confidence_by_type.setdefault(dt, []).append(conf)

        avg_confidence = {
            dt: round(float(np.mean(confs)), 3) if confs else 0.0
            for dt, confs in confidence_by_type.items()
        }

        return {
            "total_entries": len(self.chain),
            "total_detections": len(detections),
            "total_model_updates": len(model_updates),
            "total_alerts": len(alerts),
            "defect_counts": defect_counts,
            "avg_confidence_by_type": avg_confidence,
            "chain_valid": self.verify().is_valid,
            "first_entry": self.chain[0].timestamp if self.chain else None,
            "last_entry": self.chain[-1].timestamp if self.chain else None,
        }

    def export_chain(self) -> List[Dict[str, Any]]:
        """Export the entire chain as a JSON-serializable list."""
        return [asdict(e) for e in self.chain]

    def to_json(self) -> str:
        """Export chain as JSON string."""
        return json.dumps(self.export_chain(), ensure_ascii=False, indent=2)


# Need numpy for statistics
try:
    import numpy as np
except ImportError:
    # Fallback for mean
    class np:
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0.0
