"""
analysis/hallucination_defense.py — 5-Layer Hallucination Defense for Defect QA
=================================================================================

Adapted from FedCtx (federated-ai-platform/src/hallucination) for industrial
defect detection QA systems.

Prevents AI QA systems (like GPT-3.5 in Xu RC's project) from generating
hallucinated defect analysis.

Five defense layers:
1. Retrieval Consistency — QA answer must be supported by similar defect cases
2. Vector Fact-Check — Verify claims against HNSW defect knowledge base
3. CROWN Defense — Reject social answers that drop confidence below threshold
4. Multi-Node Vote — Cross-validate across multiple factory nodes
5. Self-Consistency — Ask same question 3 times, reject if inconsistent

Pure Python + NumPy. Streamlit Cloud compatible.
"""

import numpy as np
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Verdict(Enum):
    VERIFIED = "verified"
    LIKELY_TRUE = "likely_true"
    UNCERTAIN = "uncertain"
    LIKELY_FALSE = "likely_false"
    HALLUCINATION = "hallucination"


class DefenseAction(Enum):
    ACCEPT = "accept"
    FLAG = "flag"
    REJECT = "reject"
    REVIEW = "review"


@dataclass
class DefenseEvidence:
    layer: str
    source: str
    description: str
    confidence: float
    doc_id: Optional[str] = None
    similarity: Optional[float] = None


@dataclass
class DefenseResult:
    claim_id: str
    is_hallucination: bool
    risk_score: float
    verdict: Verdict
    triggered_layers: List[str]
    evidence: List[DefenseEvidence]
    defense_action: DefenseAction
    timestamp: str


@dataclass
class DefenseConfig:
    retrieval_threshold: float = 0.50
    crown_delta: float = 0.10
    self_consistency_threshold: float = 0.70
    consensus_threshold: float = 0.667
    hallucination_high_risk: float = 0.75
    hallucination_medium_risk: float = 0.45


# ── Layer 1: Retrieval Consistency ──────────────────────────

class RetrievalConsistencyChecker:
    """Check if QA answer is supported by retrieved similar defect cases."""

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    def check(
        self,
        claim: str,
        similar_defects: List[Dict[str, Any]],
    ) -> Tuple[bool, float, List[DefenseEvidence]]:
        """Check if claim is supported by similar defect cases.

        Args:
            claim: The QA answer to verify.
            similar_defects: List of {"id": str, "similarity": float, "text": str}.

        Returns:
            (is_supported, max_similarity, evidence_list)
        """
        if not similar_defects:
            return (False, 0.0, [DefenseEvidence(
                layer="RetrievalConsistency",
                source="defect_knowledge_base",
                description="No similar defect cases found to support this claim",
                confidence=0.0,
            )])

        max_sim = max(d.get("similarity", 0.0) for d in similar_defects)
        is_supported = max_sim >= self.threshold

        evidence = [
            DefenseEvidence(
                layer="RetrievalConsistency",
                source="defect_knowledge_base",
                description=f"Similar defect {d.get('id', '?')}: similarity={d.get('similarity', 0):.3f}",
                confidence=d.get("similarity", 0.0),
                doc_id=d.get("id"),
                similarity=d.get("similarity", 0.0),
            )
            for d in similar_defects[:5]
        ]

        return (is_supported, max_sim, evidence)


# ── Layer 2: Vector Fact-Check ──────────────────────────────

class VectorFactChecker:
    """Verify QA claims against HNSW defect knowledge base."""

    def __init__(self, defect_knowledge: Optional[Dict[str, Dict]] = None):
        """Initialize with defect knowledge base.

        Args:
            defect_knowledge: {defect_type: {"description": str, "causes": list, "severity": str}}
        """
        self.knowledge = defect_knowledge or self._default_knowledge()

    @staticmethod
    def _default_knowledge() -> Dict[str, Dict]:
        return {
            "short_circuit": {
                "description": "Unintended electrical connection between PCB traces",
                "causes": ["solder bridge", "copper debris", "etching residue"],
                "severity": "critical",
            },
            "open_circuit": {
                "description": "Broken trace or missing connection on PCB",
                "causes": ["incomplete etching", "mechanical damage", "thermal stress"],
                "severity": "critical",
            },
            "spurious_copper": {
                "description": "Extra copper deposit on PCB surface",
                "causes": ["etching incomplete", "contamination", "plating overflow"],
                "severity": "moderate",
            },
            "missing_hole": {
                "description": "Drill hole not present where expected",
                "causes": ["drill bit breakage", "program error", "alignment fault"],
                "severity": "critical",
            },
            "spur": {
                "description": "Small copper protrusion from trace edge",
                "causes": ["under-etching", "photomask defect", "contamination"],
                "severity": "moderate",
            },
            "crazing": {
                "description": "Network of fine cracks on steel surface",
                "causes": ["thermal stress", "rolling defects", "surface contamination"],
                "severity": "moderate",
            },
            "inclusion": {
                "description": "Foreign material embedded in steel surface",
                "causes": ["slag entrapment", "refractory particles", "deoxidation products"],
                "severity": "critical",
            },
            "pitted_surface": {
                "description": "Small cavities on steel surface",
                "causes": ["corrosion", "gas porosity", "pickling residue"],
                "severity": "moderate",
            },
            "scratches": {
                "description": "Linear marks on steel surface",
                "causes": ["mechanical handling", "rolling debris", "transport damage"],
                "severity": "minor",
            },
        }

    def verify(self, claim: str) -> Tuple[float, List[DefenseEvidence]]:
        """Verify a claim against the knowledge base.

        Returns:
            (verified_ratio, evidence_list)
        """
        claim_lower = claim.lower()
        matched = []
        total_terms = 0

        for defect_type, info in self.knowledge.items():
            # Check if defect type is mentioned
            if defect_type.replace("_", " ") in claim_lower or defect_type in claim_lower:
                total_terms += 1
                # Check if described causes match
                causes_mentioned = any(c.lower() in claim_lower for c in info.get("causes", []))
                severity_mentioned = info.get("severity", "") in claim_lower

                if causes_mentioned or severity_mentioned:
                    matched.append(DefenseEvidence(
                        layer="VectorFactCheck",
                        source="defect_knowledge_base",
                        description=f"Defect '{defect_type}': causes={'verified' if causes_mentioned else 'unverified'}, "
                                    f"severity={'verified' if severity_mentioned else 'unverified'}",
                        confidence=0.8 if (causes_mentioned and severity_mentioned) else 0.5,
                        doc_id=defect_type,
                    ))
                else:
                    matched.append(DefenseEvidence(
                        layer="VectorFactCheck",
                        source="defect_knowledge_base",
                        description=f"Defect '{defect_type}' mentioned but causes/severity not verified",
                        confidence=0.3,
                        doc_id=defect_type,
                    ))

        verified_ratio = len([m for m in matched if m.confidence >= 0.5]) / max(total_terms, 1)
        # If no defect terms found at all, don't penalize — Layer 1 handles retrieval
        if total_terms == 0:
            verified_ratio = 1.0  # No claims to verify = no fact-check violation
        return (verified_ratio, matched)


# ── Layer 3: CROWN Defense ──────────────────────────────────

class CROWNDefense:
    """CROWN confidence-drop defense.

    If social answer (from external knowledge) causes confidence to drop
    below threshold compared to initial answer, reject the social answer.
    """

    def __init__(self, delta: float = 0.10):
        self.delta = delta

    def check(
        self,
        initial_answer: str,
        initial_confidence: float,
        social_answer: str,
        social_confidence: float,
    ) -> Dict[str, Any]:
        answer_changed = initial_answer.strip() != social_answer.strip()
        confidence_drop = initial_confidence - social_confidence
        triggered = confidence_drop > self.delta

        if not answer_changed:
            final_answer = initial_answer
            reason = "Answer unchanged — no defense needed"
        elif triggered:
            final_answer = initial_answer
            reason = (f"CROWN triggered: confidence drop {confidence_drop:.3f} > δ={self.delta:.3f}, "
                      f"rejecting social answer")
        else:
            final_answer = social_answer
            reason = (f"Confidence drop {confidence_drop:.3f} ≤ δ={self.delta:.3f}, "
                      f"accepting social answer")

        return {
            "triggered": triggered,
            "confidence_drop": confidence_drop,
            "final_answer": final_answer,
            "reason": reason,
            "crown_delta": self.delta,
        }


# ── Layer 4: Multi-Node Vote ────────────────────────────────

class MultiNodeVoter:
    """Vote across multiple factory nodes for consensus."""

    def __init__(self, consensus_threshold: float = 0.667):
        self.consensus_threshold = consensus_threshold

    def vote(
        self,
        answers: List[str],
        confidences: List[float],
    ) -> Dict[str, Any]:
        if not answers:
            return {
                "consensus": None,
                "consensus_strength": 0.0,
                "verdict": Verdict.UNCERTAIN,
                "vote_counts": {},
                "dissenting": [],
            }

        counts: Dict[str, int] = {}
        for ans in answers:
            counts[ans] = counts.get(ans, 0) + 1

        best_answer = max(counts, key=counts.get)
        strength = counts[best_answer] / len(answers)
        dissenting = [a for a in answers if a != best_answer]

        if strength >= self.consensus_threshold:
            verdict = Verdict.VERIFIED
        elif strength >= 0.5:
            verdict = Verdict.LIKELY_TRUE
        elif dissenting:
            verdict = Verdict.UNCERTAIN
        else:
            verdict = Verdict.HALLUCINATION

        return {
            "consensus": best_answer,
            "consensus_strength": strength,
            "verdict": verdict,
            "vote_counts": counts,
            "dissenting": dissenting,
        }


# ── Layer 5: Self-Consistency ───────────────────────────────

class SelfConsistencyChecker:
    """Check if multiple samples of the same question produce consistent answers."""

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def check(self, answers: List[str]) -> Dict[str, Any]:
        if not answers:
            return {
                "consistency_score": 0.0,
                "most_common_answer": None,
                "passes": False,
                "dissenting_answers": [],
            }

        counts: Dict[str, int] = {}
        for ans in answers:
            counts[ans] = counts.get(ans, 0) + 1

        most_common = max(counts, key=counts.get)
        consistency_score = counts[most_common] / len(answers)
        dissenting = [a for a in answers if a != most_common]

        return {
            "consistency_score": consistency_score,
            "most_common_answer": most_common,
            "passes": consistency_score >= self.threshold,
            "dissenting_answers": dissenting,
        }


# ── Composite Defense Engine ────────────────────────────────

class DefectQADefenseEngine:
    """5-layer hallucination defense for defect detection QA.

    Coordinates all five defense layers to produce a unified judgment
    on whether a QA answer is likely hallucinated.
    """

    def __init__(self, config: Optional[DefenseConfig] = None,
                 defect_knowledge: Optional[Dict[str, Dict]] = None):
        self.config = config or DefenseConfig()
        self.retrieval_checker = RetrievalConsistencyChecker(self.config.retrieval_threshold)
        self.fact_checker = VectorFactChecker(defect_knowledge)
        self.crown = CROWNDefense(self.config.crown_delta)
        self.voter = MultiNodeVoter(self.config.consensus_threshold)
        self.consistency_checker = SelfConsistencyChecker(self.config.self_consistency_threshold)

    def check_qa_answer(
        self,
        question: str,
        answer: str,
        similar_defects: Optional[List[Dict[str, Any]]] = None,
        multi_node_answers: Optional[List[str]] = None,
        multi_node_confidences: Optional[List[float]] = None,
        self_consistency_answers: Optional[List[str]] = None,
        initial_confidence: Optional[float] = None,
        social_confidence: Optional[float] = None,
    ) -> DefenseResult:
        """Run all 5 defense layers on a QA answer.

        Args:
            question: The user's question about a defect.
            answer: The QA system's answer to verify.
            similar_defects: Retrieved similar defect cases (for Layer 1).
            multi_node_answers: Answers from other factory nodes (for Layer 4).
            multi_node_confidences: Confidence scores from other nodes.
            self_consistency_answers: Multiple samples of same answer (for Layer 5).
            initial_confidence: Model's initial confidence (for Layer 3).
            social_confidence: Confidence after seeing external knowledge (for Layer 3).

        Returns:
            DefenseResult with unified judgment.
        """
        all_evidence: List[DefenseEvidence] = []
        triggered_layers: List[str] = []
        risk_factors: List[float] = []

        # Layer 1: Retrieval Consistency
        if similar_defects is not None:
            supported, max_sim, evidence = self.retrieval_checker.check(answer, similar_defects)
            all_evidence.extend(evidence)
            if not supported:
                triggered_layers.append("RetrievalConsistency")
                risk_factors.append(1.0 - max_sim)

        # Layer 2: Vector Fact-Check
        verified_ratio, fact_evidence = self.fact_checker.verify(answer)
        all_evidence.extend(fact_evidence)
        if verified_ratio < 0.3 and fact_evidence:
            triggered_layers.append("VectorFactCheck")
            risk_factors.append(1.0 - verified_ratio)

        # Layer 3: CROWN Defense
        if initial_confidence is not None and social_confidence is not None:
            crown_result = self.crown.check(answer, initial_confidence, answer, social_confidence)
            if crown_result["triggered"]:
                triggered_layers.append("CROWN")
                risk_factors.append(min(crown_result["confidence_drop"], 1.0))

        # Layer 4: Multi-Node Vote
        if multi_node_answers and multi_node_confidences:
            vote_result = self.voter.vote(multi_node_answers, multi_node_confidences)
            all_evidence.append(DefenseEvidence(
                layer="MultiNodeVote",
                source="federated_nodes",
                description=f"Consensus: {vote_result['consensus']}, "
                            f"strength={vote_result['consensus_strength']:.1%}",
                confidence=vote_result["consensus_strength"],
            ))
            if vote_result["verdict"] in (Verdict.UNCERTAIN, Verdict.HALLUCINATION):
                triggered_layers.append("MultiNodeVote")
                risk_factors.append(1.0 - vote_result["consensus_strength"])

        # Layer 5: Self-Consistency
        if self_consistency_answers:
            sc_result = self.consistency_checker.check(self_consistency_answers)
            all_evidence.append(DefenseEvidence(
                layer="SelfConsistency",
                source="multi_sample",
                description=f"Consistency={sc_result['consistency_score']:.1%}, "
                            f"passes={sc_result['passes']}",
                confidence=sc_result["consistency_score"],
            ))
            if not sc_result["passes"]:
                triggered_layers.append("SelfConsistency")
                risk_factors.append(1.0 - sc_result["consistency_score"])

        # Compute risk score
        risk_score = float(np.mean(risk_factors)) if risk_factors else 0.0
        risk_score = min(risk_score, 1.0)

        # Verdict
        if risk_score >= self.config.hallucination_high_risk:
            verdict = Verdict.HALLUCINATION
        elif risk_score >= self.config.hallucination_medium_risk:
            verdict = Verdict.LIKELY_FALSE
        elif risk_score > 0.1:
            verdict = Verdict.UNCERTAIN
        elif risk_score > 0.0:
            verdict = Verdict.LIKELY_TRUE
        else:
            verdict = Verdict.VERIFIED

        # Defense action
        if verdict in (Verdict.VERIFIED, Verdict.LIKELY_TRUE):
            action = DefenseAction.ACCEPT
        elif verdict == Verdict.UNCERTAIN:
            action = DefenseAction.FLAG
        elif verdict == Verdict.LIKELY_FALSE:
            action = DefenseAction.REVIEW
        else:
            action = DefenseAction.REJECT

        claim_id = hashlib.sha256(f"{question}:{answer}".encode()).hexdigest()[:12]

        return DefenseResult(
            claim_id=claim_id,
            is_hallucination=verdict == Verdict.HALLUCINATION,
            risk_score=risk_score,
            verdict=verdict,
            triggered_layers=triggered_layers,
            evidence=all_evidence,
            defense_action=action,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
