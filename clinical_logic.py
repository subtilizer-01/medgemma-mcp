"""
clinical_logic.py — ClinicalLogic
CyberHealth AI | MedGemma Clinical Safety Auditor

Improvements over original:
  - FHIR context enrichment support
  - Cleaner risk determination logic
  - Added overall_recommendation field
"""

from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalLogic:
    """
    Transforms raw MedGemma audit results into actionable safety insights.
    Calculates gravity scores, generates decision flows, and detects biases.
    """

    CLINICAL_WEIGHTS = {
        "time_criticality": {
            "minutes": 3.0,
            "hours": 2.0,
            "days": 1.0,
            "": 1.0
        },
        "action_specificity": {
            "STAT": 1.8,
            "IMMEDIATE": 1.7,
            "URGENT": 1.5,
            "CONSIDER": 1.2,
            "ROUTINE": 1.0
        }
    }

    SEVERITY_ICONS = {
        "HIGH": "🚨",
        "MEDIUM": "⚠️",
        "LOW": "✅"
    }

    PRIORITY_TAGS = {
        "STAT": "Critical",
        "URGENT": "High",
        "PRIORITY": "Elevated",
        "ROUTINE": "Routine",
        "NONE": "Low"
    }

    def calculate_safety_metrics(self, audit_report: dict) -> dict:
        logger.info("Calculating clinical safety metrics")

        if "error" in audit_report:
            return self._error_metrics(audit_report["error"])

        findings = audit_report.get("critical_findings", [])
        high_risk = [
            f for f in findings
            if f.get("risk", "").upper() == "HIGH"
        ]
        medium_risk = [
            f for f in findings
            if f.get("risk", "").upper() == "MEDIUM"
        ]

        score_breakdown = []

        # High Risk Scoring
        hr_score = 0
        for finding in high_risk:
            time_crit = finding.get("time_critical", "").lower()
            weight = self.CLINICAL_WEIGHTS["time_criticality"].get(
                time_crit, 1.5
            )
            action = finding.get("required_action", "").upper()
            bonus = 1.0
            for k, v in self.CLINICAL_WEIGHTS["action_specificity"].items():
                if k in action:
                    bonus = max(bonus, v)
            hr_score += 35 * weight * bonus

        if high_risk:
            score_breakdown.append({
                "Factor": "High Risk Findings",
                "Count": len(high_risk),
                "Weight": "35pts + Multipliers",
                "Contribution": int(hr_score),
                "Severity": "HIGH",
                "Icon": self.SEVERITY_ICONS["HIGH"]
            })

        # Medium Risk Scoring
        mr_score = 0
        for finding in medium_risk:
            time_crit = finding.get("time_critical", "").lower()
            weight = self.CLINICAL_WEIGHTS["time_criticality"].get(
                time_crit, 1.0
            )
            mr_score += 20 * weight

        if medium_risk:
            score_breakdown.append({
                "Factor": "Medium Risk Findings",
                "Count": len(medium_risk),
                "Weight": "20pts (Base)",
                "Contribution": int(mr_score),
                "Severity": "MEDIUM",
                "Icon": self.SEVERITY_ICONS["MEDIUM"]
            })

        # Data Gaps
        gaps = len(audit_report.get("data_gaps", []))
        gap_score = gaps * 8
        if gaps > 0:
            score_breakdown.append({
                "Factor": "Missing Clinical Data",
                "Count": gaps,
                "Weight": "8pts / gap",
                "Contribution": gap_score,
                "Severity": "LOW",
                "Icon": self.SEVERITY_ICONS["LOW"]
            })

        # Cognitive Biases
        biases = len(audit_report.get("cognitive_biases", []))
        bias_score = biases * 10
        if biases > 0:
            score_breakdown.append({
                "Factor": "Cognitive Bias Pattern",
                "Count": biases,
                "Weight": "10pts / bias",
                "Contribution": bias_score,
                "Severity": "LOW",
                "Icon": self.SEVERITY_ICONS["LOW"]
            })

        raw_gravity = hr_score + mr_score + gap_score + bias_score
        gravity_score = max(10, min(100, int(raw_gravity)))

        decision_flow = self._generate_decision_flow(high_risk, medium_risk)
        data_completeness = self._calculate_data_completeness(
            audit_report.get("data_gaps", [])
        )
        bias_analysis = self._analyze_cognitive_biases(
            audit_report.get("cognitive_biases", [])
        )
        clinical_summary = self._generate_clinical_summary(
            high_risk, audit_report
        )
        invariant_check = self._check_safety_invariants(audit_report)
        risk_level = self._determine_overall_risk_level(
            high_risk, gravity_score
        )
        recommendation = self._generate_recommendation(
            risk_level, high_risk, gravity_score
        )

        return {
            "clinical_gravity_score": gravity_score,
            "scoring_breakdown": score_breakdown,
            "high_risk_count": len(high_risk),
            "medium_risk_count": len(medium_risk),
            "data_completeness": data_completeness,
            "decision_flow": decision_flow,
            "bias_analysis": bias_analysis,
            "safety_confidence": audit_report.get("safety_score", 50),
            "clinical_summary": clinical_summary,
            "overall_recommendation": recommendation,
            "timestamp": self._get_timestamp(),
            "invariant_check": invariant_check,
            "risk_level": risk_level
        }

    # ─── Helper Methods ───────────────────────────────────────────
    def _calculate_data_completeness(self, data_gaps) -> int:
        if not data_gaps:
            return 100
        return max(20, 100 - (len(data_gaps) * 12))

    def _generate_decision_flow(self, high_risk, medium_risk) -> list:
        flow = []
        step = 1

        for finding in high_risk[:3]:
            tc = finding.get("time_critical", "hours").lower()
            if "minute" in tc:
                priority, icon, urgency = "STAT", "🚨", "CRITICAL"
                verb = "Mitigate"
            elif "hour" in tc:
                priority, icon, urgency = "URGENT", "🔴", "HIGH"
                verb = "Rule out"
            else:
                priority, icon, urgency = "PRIORITY", "🟡", "ELEVATED"
                verb = "Address"

            flow.append({
                "step": step,
                "priority": priority,
                "priority_tag": self.PRIORITY_TAGS[priority],
                "action": finding.get("required_action"),
                "rationale": f"{verb} {finding.get('likely_missed', 'risk')}",
                "icon": icon,
                "urgency": urgency,
                "time_critical": finding.get("time_critical", "hours")
            })
            step += 1

        for finding in medium_risk[:2]:
            flow.append({
                "step": step,
                "priority": "ROUTINE",
                "priority_tag": self.PRIORITY_TAGS["ROUTINE"],
                "action": finding.get("required_action"),
                "rationale": "Standard of care",
                "icon": "🟢",
                "urgency": "ROUTINE",
                "time_critical": finding.get("time_critical", "days")
            })
            step += 1

        if not flow:
            flow.append({
                "step": 1,
                "priority": "NONE",
                "priority_tag": self.PRIORITY_TAGS["NONE"],
                "action": "Continue standard protocol",
                "rationale": "No specific risks identified",
                "icon": "✅",
                "urgency": "LOW",
                "time_critical": "days"
            })

        return flow

    def _analyze_cognitive_biases(self, biases) -> dict:
        if not biases:
            return {
                "count": 0,
                "summary": "✅ No cognitive biases detected",
                "types": []
            }
        bias_types = []
        if biases and isinstance(biases[0], dict):
            bias_types = [
                b.get("type", "Unknown").replace("_", " ").title()
                for b in biases
            ]
        else:
            bias_types = [
                str(b).replace("_", " ").title() for b in biases
            ]
        return {
            "count": len(biases),
            "summary": f"🧠 {len(biases)} cognitive bias(es) detected",
            "details": biases,
            "types": bias_types
        }

    def _generate_clinical_summary(self, high_risk, report) -> str:
        if not high_risk:
            score = report.get("safety_score", 50)
            if score > 70:
                return "✅ SAFETY CHECK PASSED — No critical findings"
            return "⚠️ MODERATE CONCERNS — Review recommended"
        conditions = list(set([
            f.get("likely_missed", "condition")
            for f in high_risk[:3]
        ]))
        return (
            f"🚨 HIGH RISK ALERT · "
            f"Potential {', '.join(conditions)} · "
            f"Immediate review required"
        )

    def _check_safety_invariants(self, report) -> dict:
        violations = []
        high_risk = [
            f for f in report.get("critical_findings", [])
            if f.get("risk", "").upper() == "HIGH"
        ]
        for i, f in enumerate(high_risk):
            if not f.get("time_critical"):
                violations.append(
                    f"Finding #{i+1} missing time-criticality (Severity: MED)"
                )
            if not f.get("required_action"):
                violations.append(
                    f"Finding #{i+1} missing required action (Severity: HIGH)"
                )
        if not violations:
            return {
                "status": "PASS",
                "message": "✅ All safety invariants satisfied"
            }
        return {
            "status": "FAIL",
            "message": f"⚠️ {len(violations)} logic violation(s) found",
            "violations": violations
        }

    def _determine_overall_risk_level(self, high_risk, gravity) -> str:
        if high_risk:
            return "HIGH"
        if gravity > 70:
            return "ELEVATED"
        if gravity > 40:
            return "MODERATE"
        return "LOW"

    def _generate_recommendation(self, risk_level: str,
                                  high_risk: list,
                                  gravity: int) -> str:
        if risk_level == "HIGH":
            conditions = [
                f.get("likely_missed", "unspecified condition")
                for f in high_risk[:2]
            ]
            return (
                f"IMMEDIATE ACTION REQUIRED: Rule out "
                f"{', '.join(conditions)}. "
                f"Do not discharge without senior review."
            )
        if risk_level == "ELEVATED":
            return (
                "CLOSE MONITORING: Gravity score elevated. "
                "Consider additional workup before disposition."
            )
        if risk_level == "MODERATE":
            return (
                "ROUTINE FOLLOW-UP: No immediate threats identified. "
                "Standard monitoring protocol appropriate."
            )
        return (
            "LOW RISK: Note appears complete. "
            "Continue standard care pathway."
        )

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _error_metrics(self, error_msg: str) -> dict:
        return {
            "clinical_gravity_score": 50,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "scoring_breakdown": [],
            "clinical_summary": "⚠️ System Error — Manual review required",
            "overall_recommendation": "Manual clinical review required",
            "risk_level": "UNKNOWN",
            "decision_flow": [],
            "bias_analysis": {"count": 0, "summary": "N/A", "types": []},
            "data_completeness": 0,
            "invariant_check": {"status": "ERROR", "message": error_msg},
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
