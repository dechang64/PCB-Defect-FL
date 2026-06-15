"""
analysis/conformity.py — EWA Conformity Detector for Defect Detection
=======================================================================

Adapted from ewa-fed (twc_core.ewa.conformity) for industrial defect detection.

Monitors whether minority expert clients' domain knowledge is being
suppressed by majority clients in federated defect detection.

Key use cases:
1. Transfer learning feasibility: predict if source→target transfer will fail
2. FL round monitoring: detect conformity issues during federated training
3. Cross-domain defect matching: measure semantic distance between defect types

Pure NumPy. No PyTorch. Streamlit Cloud compatible.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class RoundSnapshot:
    """Conformity metrics for one FL round."""
    round_id: int
    avg_conformity: float
    high_conformity_ratio: float
    minority_suppressed: int
    avg_entropy: float
    num_clients: int
    num_primitives: int
    num_classes: int
    strategy: str
    class_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ConformityAlert:
    """Alert when conformity exceeds threshold."""
    round_id: int
    severity: str  # "warning" | "critical"
    class_label: str
    message: str
    details: Dict[str, Any]


@dataclass
class TransferFeasibility:
    """Result of transfer learning feasibility check."""
    source_domain: str
    target_domain: str
    semantic_distance: float  # 0.0 (identical) to 1.0 (completely different)
    conformity_risk: float    # 0.0 (safe) to 1.0 (will fail)
    recommendation: str       # "proceed" | "caution" | "not_recommended" | "do_not_transfer"
    reasoning: str
    alternative_approaches: List[str] = field(default_factory=list)


class DefectConformityDetector:
    """Conformity detector for industrial defect detection.

    Tracks conformity trends across FL rounds and predicts
    transfer learning feasibility across defect domains.

    Conformity score per class: 0.0 (healthy) to 1.0 (total suppression).
    """

    def __init__(
        self,
        warning_threshold: float = 0.5,
        critical_threshold: float = 0.8,
        window_size: int = 5,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.window_size = window_size
        self.history: List[RoundSnapshot] = []
        self.alerts: List[ConformityAlert] = []

    def update(self, round_id: int, class_conformity: Dict[str, float],
               entropy_stats: Dict[str, float], num_clients: int = 3) -> RoundSnapshot:
        """Record conformity metrics from one aggregation round.

        Args:
            round_id: FL round number.
            class_conformity: {class_label: conformity_score} per class.
            entropy_stats: {"mean": float, "std": float, ...}.
            num_clients: Number of FL clients.

        Returns:
            RoundSnapshot for this round.
        """
        # Check per-class conformity and generate alerts
        class_details = {}
        for class_label, score in class_conformity.items():
            if score >= self.critical_threshold:
                status = "suppressed"
                self.alerts.append(ConformityAlert(
                    round_id=round_id,
                    severity="critical",
                    class_label=class_label,
                    message=f"Critical: {class_label} — expert knowledge likely suppressed "
                            f"(conformity={score:.2f})",
                    details={"conformity_score": score, "status": status},
                ))
            elif score >= self.warning_threshold:
                status = "partial"
                self.alerts.append(ConformityAlert(
                    round_id=round_id,
                    severity="warning",
                    class_label=class_label,
                    message=f"Warning: {class_label} — partial conformity detected "
                            f"(conformity={score:.2f})",
                    details={"conformity_score": score, "status": status},
                ))
            else:
                status = "healthy"

            class_details[class_label] = {
                "conformity_score": score,
                "status": status,
            }

        avg_conformity = float(np.mean(list(class_conformity.values()))) if class_conformity else 0.0
        high_ratio = sum(1 for s in class_conformity.values() if s > 0.5) / max(len(class_conformity), 1)
        minority_suppressed = sum(1 for s in class_conformity.values() if s >= self.critical_threshold)

        snapshot = RoundSnapshot(
            round_id=round_id,
            avg_conformity=avg_conformity,
            high_conformity_ratio=high_ratio,
            minority_suppressed=minority_suppressed,
            avg_entropy=entropy_stats.get("mean", 0.0),
            num_clients=num_clients,
            num_primitives=sum(class_conformity.keys().__len__() for _ in [0]),
            num_classes=len(class_conformity),
            strategy="entropy_weighted",
            class_details=class_details,
        )
        self.history.append(snapshot)
        return snapshot

    def check_transfer_feasibility(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_labels: List[str],
        target_labels: List[str],
        source_domain: str = "source",
        target_domain: str = "target",
    ) -> TransferFeasibility:
        """Predict if transfer learning from source to target will succeed.

        Uses feature distribution distance as a proxy for domain gap.

        Args:
            source_features: (N_s, D) feature matrix from source domain.
            target_features: (N_t, D) feature matrix from target domain.
            source_labels: Class labels for source samples.
            target_labels: Class labels for target samples.
            source_domain: Name of source domain.
            target_domain: Name of target domain.

        Returns:
            TransferFeasibility with recommendation.
        """
        if source_features.size == 0 or target_features.size == 0:
            return TransferFeasibility(
                source_domain=source_domain,
                target_domain=target_domain,
                semantic_distance=1.0,
                conformity_risk=1.0,
                recommendation="do_not_transfer",
                reasoning="Empty feature matrices — insufficient data for feasibility check.",
                alternative_approaches=["Collect more data from target domain"],
            )

        # Compute domain-level distance (MMD-like)
        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)
        source_std = np.std(source_features, axis=0) + 1e-8
        target_std = np.std(target_features, axis=0) + 1e-8

        # Normalized mean difference
        mean_diff = np.abs(source_mean - target_mean) / (0.5 * (source_std + target_std))
        semantic_distance = float(np.clip(np.mean(mean_diff), 0.0, 1.0))

        # Distribution overlap (simplified)
        source_var = np.var(source_features, axis=0)
        target_var = np.var(target_features, axis=0)
        var_ratio = np.minimum(source_var, target_var) / (np.maximum(source_var, target_var) + 1e-8)
        overlap = float(np.mean(var_ratio))

        # Label overlap
        source_set = set(source_labels)
        target_set = set(target_labels)
        label_overlap = len(source_set & target_set) / max(len(source_set | target_set), 1)

        # Conformity risk: high semantic distance + low overlap = high risk
        conformity_risk = float(np.clip(
            0.5 * semantic_distance + 0.3 * (1.0 - overlap) + 0.2 * (1.0 - overlap), 0.0, 1.0
        ))

        # Recommendation
        if conformity_risk < 0.3:
            recommendation = "proceed"
            reasoning = (
                f"Low domain gap (distance={semantic_distance:.3f}), "
                f"good label overlap ({label_overlap:.1%}). "
                f"Transfer learning likely to succeed."
            )
            alternatives = []
        elif conformity_risk < 0.5:
            recommendation = "caution"
            reasoning = (
                f"Moderate domain gap (distance={semantic_distance:.3f}). "
                f"Transfer may work with careful fine-tuning and domain adaptation."
            )
            alternatives = [
                "Use domain adaptation (DANN, MMD-based)",
                "Fine-tune only the last 2 layers",
                "Use lower learning rate for transferred weights",
            ]
        elif conformity_risk < 0.7:
            recommendation = "not_recommended"
            reasoning = (
                f"High domain gap (distance={semantic_distance:.3f}), "
                f"low label overlap ({label_overlap:.1%}). "
                f"Transfer learning likely to fail or produce negative results."
            )
            alternatives = [
                "Few-shot learning (ProtoNet, MAML)",
                "Domain adaptation with adversarial training",
                "Train from scratch on target domain if data permits",
                "Use synthetic data augmentation for target domain",
            ]
        else:
            recommendation = "do_not_transfer"
            reasoning = (
                f"Severe domain gap (distance={semantic_distance:.3f}). "
                f"Source and target domains are fundamentally different. "
                f"Transfer learning will almost certainly fail."
            )
            alternatives = [
                "Few-shot learning with metric-based approach",
                "Collect more target domain data",
                "Use unsupervised anomaly detection instead",
                "Consider zero-shot approaches with CLIP/DINOv2",
            ]

        return TransferFeasibility(
            source_domain=source_domain,
            target_domain=target_domain,
            semantic_distance=semantic_distance,
            conformity_risk=conformity_risk,
            recommendation=recommendation,
            reasoning=reasoning,
            alternative_approaches=alternatives,
        )

    def get_trend(self) -> Dict[str, Any]:
        """Analyze conformity trend over recent rounds."""
        if len(self.history) < 2:
            return {"status": "insufficient_data", "rounds": len(self.history)}

        recent = self.history[-self.window_size:]
        conformities = [s.avg_conformity for s in recent]

        x = np.arange(len(conformities))
        if len(conformities) > 1:
            slope = float(np.polyfit(x, conformities, 1)[0])
        else:
            slope = 0.0

        if slope > 0.02:
            direction = "rising"
        elif slope < -0.02:
            direction = "falling"
        else:
            direction = "stable"

        return {
            "status": direction,
            "current": round(conformities[-1], 4),
            "slope": round(slope, 4),
            "window_avg": round(float(np.mean(conformities)), 4),
            "window_max": round(float(np.max(conformities)), 4),
            "window_min": round(float(np.min(conformities)), 4),
            "total_rounds": len(self.history),
            "total_alerts": len(self.alerts),
        }

    def get_report(self) -> Dict[str, Any]:
        """Generate full conformity report."""
        trend = self.get_trend()

        class_analysis = {}
        if self.history:
            latest = self.history[-1]
            class_analysis = latest.class_details

        recommendations = []
        if trend["status"] == "rising":
            recommendations.append(
                "Conformity is increasing. Consider: "
                "(1) Adding more diverse clients, "
                "(2) Using entropy-weighted aggregation, "
                "(3) Reviewing data distribution for severe Non-IID."
            )
        critical_count = sum(1 for a in self.alerts if a.severity == "critical")
        if critical_count > 0:
            recommendations.append(
                f"{critical_count} critical alerts. "
                "Minority expertise may be lost. "
                "Review client data quality and class balance."
            )
        if not recommendations:
            recommendations.append("Conformity levels are healthy. No action needed.")

        return {
            "trend": trend,
            "class_analysis": class_analysis,
            "alerts": [
                {"round": a.round_id, "severity": a.severity,
                 "class": a.class_label, "message": a.message,
                 "details": a.details}
                for a in self.alerts[-20:]
            ],
            "recommendations": recommendations,
            "total_rounds_tracked": len(self.history),
        }

    def reset(self):
        """Clear all history and alerts."""
        self.history = []
        self.alerts = []
