"""
fhir_client.py — FHIR Client
CyberHealth AI | MedGemma Clinical Safety Auditor

Fetches synthetic patient data from HAPI FHIR public sandbox.
All data is synthetic — safe for hackathon use.
HAPI FHIR public server: https://hapi.fhir.org/baseR4
"""

import logging
import httpx

logger = logging.getLogger(__name__)

FHIR_BASE = "https://hapi.fhir.org/baseR4"
TIMEOUT = 10.0


def get_patient_summary(patient_fhir_id: str) -> str:
    """
    Fetches a complete patient summary from FHIR.
    Returns formatted plain text for MedGemma consumption.
    """
    sections = []

    conditions = _get_conditions(patient_fhir_id)
    if conditions:
        sections.append(f"Prior Conditions: {', '.join(conditions)}")

    medications = _get_medications(patient_fhir_id)
    if medications:
        sections.append(f"Active Medications: {', '.join(medications)}")

    allergies = _get_allergies(patient_fhir_id)
    if allergies:
        sections.append(f"Known Allergies: {', '.join(allergies)}")

    observations = _get_recent_observations(patient_fhir_id)
    if observations:
        sections.append(f"Recent Vitals/Labs: {', '.join(observations)}")

    if not sections:
        return "No FHIR history found for this patient ID."

    return "=== FHIR Patient History ===\n" + "\n".join(sections)


def _get_conditions(patient_id: str) -> list:
    try:
        r = httpx.get(
            f"{FHIR_BASE}/Condition",
            params={"patient": patient_id, "_count": "10"},
            timeout=TIMEOUT
        )
        r.raise_for_status()
        bundle = r.json()
        conditions = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            code = resource.get("code", {})
            display = (
                code.get("text") or
                (code.get("coding", [{}])[0].get("display", ""))
            )
            if display:
                conditions.append(display)
        return conditions
    except Exception as e:
        logger.warning(f"FHIR conditions fetch failed: {e}")
        return []


def _get_medications(patient_id: str) -> list:
    try:
        r = httpx.get(
            f"{FHIR_BASE}/MedicationRequest",
            params={"patient": patient_id, "status": "active", "_count": "10"},
            timeout=TIMEOUT
        )
        r.raise_for_status()
        bundle = r.json()
        meds = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            med = resource.get("medicationCodeableConcept", {})
            name = (
                med.get("text") or
                (med.get("coding", [{}])[0].get("display", ""))
            )
            if name:
                meds.append(name)
        return meds
    except Exception as e:
        logger.warning(f"FHIR medications fetch failed: {e}")
        return []


def _get_allergies(patient_id: str) -> list:
    try:
        r = httpx.get(
            f"{FHIR_BASE}/AllergyIntolerance",
            params={"patient": patient_id, "_count": "10"},
            timeout=TIMEOUT
        )
        r.raise_for_status()
        bundle = r.json()
        allergies = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            code = resource.get("code", {})
            name = (
                code.get("text") or
                (code.get("coding", [{}])[0].get("display", ""))
            )
            if name:
                allergies.append(name)
        return allergies
    except Exception as e:
        logger.warning(f"FHIR allergies fetch failed: {e}")
        return []


def _get_recent_observations(patient_id: str) -> list:
    try:
        r = httpx.get(
            f"{FHIR_BASE}/Observation",
            params={
                "patient": patient_id,
                "_count": "10",
                "_sort": "-date"
            },
            timeout=TIMEOUT
        )
        r.raise_for_status()
        bundle = r.json()
        obs = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            code = resource.get("code", {})
            name = (
                code.get("text") or
                (code.get("coding", [{}])[0].get("display", ""))
            )
            value = resource.get("valueQuantity", {})
            val_str = ""
            if value:
                val_str = (
                    f": {value.get('value', '')} "
                    f"{value.get('unit', '')}"
                )
            if name:
                obs.append(f"{name}{val_str}".strip())
        return obs
    except Exception as e:
        logger.warning(f"FHIR observations fetch failed: {e}")
        return []


def write_audit_to_fhir(
    patient_id: str,
    audit_summary: str,
    gravity_score: int
) -> dict:
    """
    Writes audit findings back to FHIR as a Clinical Impression resource.
    This demonstrates FHIR write-back capability.
    """
    try:
        resource = {
            "resourceType": "ClinicalImpression",
            "status": "completed",
            "subject": {"reference": f"Patient/{patient_id}"},
            "description": audit_summary,
            "note": [
                {
                    "text": (
                        f"MedGemma Clinical Safety Audit | "
                        f"Gravity Score: {gravity_score}/100 | "
                        f"Generated by CyberHealth AI"
                    )
                }
            ]
        }

        r = httpx.post(
            f"{FHIR_BASE}/ClinicalImpression",
            json=resource,
            headers={"Content-Type": "application/fhir+json"},
            timeout=TIMEOUT
        )
        r.raise_for_status()
        result = r.json()
        return {
            "status": "success",
            "fhir_id": result.get("id", "unknown"),
            "message": "Audit written to FHIR as ClinicalImpression"
        }

    except Exception as e:
        logger.warning(f"FHIR write-back failed: {e}")
        return {
            "status": "failed",
            "message": f"FHIR write-back unavailable: {str(e)}"
        }
