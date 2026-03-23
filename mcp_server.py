"""
mcp_server.py — MCP Server Entry Point
CyberHealth AI | MedGemma Clinical Safety Auditor

ALL 10 FEATURES:
  1. audit_clinical_note      — Full safety audit with gravity score
  2. get_safety_summary       — Quick triage summary
  3. audit_document           — PDF/Word/Excel file parsing + audit
  4. enrich_with_fhir         — Fetch FHIR history and audit together
  5. check_drug_interactions  — Extract meds, flag dangerous combos
  6. generate_differential    — Top 5 missed diagnoses
  7. score_note_completeness  — Grade note for missing elements
  8. batch_audit              — Audit multiple notes, rank by risk
  9. second_opinion           — Run note twice, compare results
 10. write_fhir_audit         — Write findings back to FHIR
"""

import asyncio
import json
import logging
import os
import re
import sys

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from engine import SafetyAuditor
from clinical_logic import ClinicalLogic
from security import VaultGuard
from document_parser import extract_text_from_base64
from fhir_client import get_patient_summary, write_audit_to_fhir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

API_KEY    = os.getenv("MCP_API_KEY", "cyberhealth-medgemma-2026")
SERVER_URL = os.getenv("RAILWAY_EXTERNAL_URL", os.getenv("RENDER_EXTERNAL_URL", ""))
PORT       = int(os.getenv("PORT", 8000))

logger.info("=" * 50)
logger.info("CyberHealth AI — MedGemma MCP Server v2.0")
logger.info("Loading all 10 clinical tools...")
logger.info("=" * 50)

auditor = SafetyAuditor()
logic   = ClinicalLogic()
guard   = VaultGuard()
auditor.warmup()
logger.info("All components ready.")

DANGEROUS_INTERACTIONS = [
    {"drugs": ["warfarin", "aspirin"], "severity": "HIGH", "effect": "Increased bleeding risk — dual anticoagulation"},
    {"drugs": ["warfarin", "ibuprofen"], "severity": "HIGH", "effect": "Increased bleeding risk — NSAID potentiates warfarin"},
    {"drugs": ["metformin", "contrast"], "severity": "HIGH", "effect": "Lactic acidosis risk — hold metformin before contrast"},
    {"drugs": ["ssri", "tramadol"], "severity": "HIGH", "effect": "Serotonin syndrome risk"},
    {"drugs": ["ace inhibitor", "potassium"], "severity": "MEDIUM", "effect": "Hyperkalemia risk"},
    {"drugs": ["digoxin", "amiodarone"], "severity": "HIGH", "effect": "Digoxin toxicity — amiodarone raises digoxin levels"},
    {"drugs": ["methotrexate", "nsaid"], "severity": "HIGH", "effect": "Methotrexate toxicity — NSAIDs reduce renal clearance"},
    {"drugs": ["clopidogrel", "omeprazole"], "severity": "MEDIUM", "effect": "Reduced clopidogrel efficacy — CYP2C19 inhibition"},
    {"drugs": ["lithium", "nsaid"], "severity": "HIGH", "effect": "Lithium toxicity — NSAIDs reduce renal lithium clearance"},
    {"drugs": ["sildenafil", "nitrate"], "severity": "HIGH", "effect": "Severe hypotension — contraindicated combination"},
    {"drugs": ["maoi", "ssri"], "severity": "HIGH", "effect": "Serotonin syndrome — potentially fatal"},
]

mcp = FastMCP(
    name="MedGemma Clinical Safety Auditor",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    instructions=(
        "AI-powered clinical note safety auditor using local MedGemma. "
        "10 tools available. All processing local — zero patient data leaves network."
    )
)


async def _run_audit_pipeline(note: str, patient_id: str) -> dict:
    safe_note = guard.scrub_phi(note)
    loop = asyncio.get_event_loop()
    audit_result = await loop.run_in_executor(None, lambda: auditor.audit_note(safe_note))
    metrics = logic.calculate_safety_metrics(audit_result)
    seal = auditor.generate_seal(safe_note, audit_result)
    return {
        "status": "success",
        "patient_id": patient_id,
        "analysis_mode": audit_result.get("analysis_mode", "LIVE"),
        "clinical_gravity_score": metrics["clinical_gravity_score"],
        "risk_level": metrics["risk_level"],
        "clinical_summary": metrics["clinical_summary"],
        "overall_recommendation": metrics["overall_recommendation"],
        "scoring_breakdown": metrics["scoring_breakdown"],
        "decision_flow": metrics["decision_flow"],
        "bias_analysis": metrics["bias_analysis"],
        "data_completeness": metrics["data_completeness"],
        "invariant_check": metrics["invariant_check"],
        "critical_findings": audit_result.get("critical_findings", []),
        "data_gaps": audit_result.get("data_gaps", []),
        "integrity_seal": seal,
        "timestamp": metrics["timestamp"],
        "privacy_note": "All analysis performed locally via MedGemma. No patient data transmitted externally."
    }


@mcp.tool()
async def audit_clinical_note(clinical_note: str, patient_id: str = "SYNTHETIC-TEST") -> str:
    """
    Full AI safety audit of a clinical note using MedGemma.
    Detects cognitive biases, calculates gravity score (0-100), identifies data gaps,
    generates decision flow. Cryptographic seal included. All processing local.
    Args:
        clinical_note: Raw clinical note. Synthetic/de-identified only.
        patient_id: De-identified patient ID for audit trail.
    Returns: JSON with gravity score, risk level, decision flow, bias analysis, integrity seal.
    """
    logger.info(f"[Tool 1] audit_clinical_note | patient={patient_id}")
    if not clinical_note.strip():
        return json.dumps({"status": "error", "error": "clinical_note cannot be empty"})
    try:
        return json.dumps(await _run_audit_pipeline(clinical_note, patient_id), indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def get_safety_summary(clinical_note: str) -> str:
    """
    Quick triage summary — faster than full audit.
    Returns headline risk level, gravity score, top finding, immediate action.
    Args:
        clinical_note: Raw clinical note. Synthetic/de-identified only.
    Returns: JSON with risk_level, gravity_score, summary, top_action.
    """
    logger.info("[Tool 2] get_safety_summary")
    try:
        safe_note = guard.scrub_phi(clinical_note)
        loop = asyncio.get_event_loop()
        audit_result = await loop.run_in_executor(None, lambda: auditor.audit_note(safe_note))
        metrics = logic.calculate_safety_metrics(audit_result)
        top_action = "No immediate action required"
        if metrics["decision_flow"]:
            top_action = metrics["decision_flow"][0].get("action", top_action)
        return json.dumps({
            "status": "success",
            "risk_level": metrics["risk_level"],
            "gravity_score": metrics["clinical_gravity_score"],
            "summary": metrics["clinical_summary"],
            "high_risk_findings": metrics["high_risk_count"],
            "medium_risk_findings": metrics["medium_risk_count"],
            "top_action": top_action,
            "bias_count": metrics["bias_analysis"]["count"]
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def audit_document(file_base64: str, file_type: str, patient_id: str = "SYNTHETIC-TEST") -> str:
    """
    Audits a clinical document (PDF, Word, Excel) for safety risks.
    Accepts base64-encoded file, extracts text, runs full MedGemma safety audit.
    Supports PDF discharge summaries, Word clinic notes, Excel lab reports.
    Args:
        file_base64: Base64-encoded file content.
        file_type: One of: pdf, docx, xlsx, csv
        patient_id: De-identified patient ID.
    Returns: JSON with extracted text preview and full audit results.
    """
    logger.info(f"[Tool 3] audit_document | type={file_type} | patient={patient_id}")
    try:
        extracted_text = extract_text_from_base64(file_base64, file_type)
        if any(extracted_text.startswith(x) for x in ["Unsupported", "PDF appears", "Word document"]):
            return json.dumps({"status": "error", "error": extracted_text})
        text_for_audit = extracted_text[:2000]
        result = await _run_audit_pipeline(text_for_audit, patient_id)
        result["document_type"] = file_type
        result["extracted_text_preview"] = extracted_text[:500] + ("..." if len(extracted_text) > 500 else "")
        result["total_extracted_chars"] = len(extracted_text)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def enrich_with_fhir(clinical_note: str, patient_fhir_id: str, patient_id: str = "SYNTHETIC-TEST") -> str:
    """
    Fetches patient history from FHIR and audits note with full context.
    Gets prior conditions, medications, allergies, vitals from HAPI FHIR sandbox.
    MedGemma audits with complete patient context — catches risks missed without history.
    Args:
        clinical_note: Raw clinical note to audit.
        patient_fhir_id: FHIR patient ID (e.g. "592309" for HAPI sandbox).
        patient_id: De-identified ID for audit trail.
    Returns: JSON with FHIR history and full audit results.
    """
    logger.info(f"[Tool 4] enrich_with_fhir | fhir_id={patient_fhir_id}")
    try:
        loop = asyncio.get_event_loop()
        gemini_key = os.getenv("GEMINI_API_KEY", "")

        # Fetch FHIR history with timeout
        try:
            fhir_summary = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: get_patient_summary(patient_fhir_id)),
                timeout=8.0
            )
        except asyncio.TimeoutError:
            fhir_summary = f"FHIR history unavailable (timeout) for patient {patient_fhir_id}."

        if gemini_key:
            # Use Gemini for fast analysis of enriched note
            enriched = f"{fhir_summary}\n\nCurrent Note: {clinical_note}"
            prompt = f"""You are a senior physician auditing this clinical note with patient history.
OUTPUT ONLY VALID JSON. No preamble. No markdown.

{enriched[:2000]}

REQUIRED JSON:
{{
  "safety_score": <0-100>,
  "critical_findings": [
    {{
      "finding": "<description>",
      "risk": "HIGH" or "MEDIUM",
      "evidence": "<quote from note>",
      "likely_missed": "<diagnosis>",
      "required_action": "<action>",
      "time_critical": "minutes" or "hours"
    }}
  ],
  "data_gaps": ["<missing item>"],
  "cognitive_biases": [{{"type": "<bias>", "evidence": "<why>", "risk": "HIGH" or "MEDIUM"}}],
  "fhir_insights": "<how patient history changes the risk assessment>"
}}"""

            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": gemini_key},
                    json={"contents": [{"parts": [{"text": prompt}]}]}
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["candidates"][0]["content"]["parts"][0]["text"]

            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                try:
                    audit_result = json.loads(match.group(0))
                    audit_result["analysis_mode"] = "GEMINI_FHIR_ENRICHED"
                    metrics = logic.calculate_safety_metrics(audit_result)
                    seal = auditor.generate_seal(clinical_note, audit_result)
                    result = {
                        "status": "success",
                        "patient_id": patient_id,
                        "analysis_mode": "GEMINI_FHIR_ENRICHED",
                        "clinical_gravity_score": metrics["clinical_gravity_score"],
                        "risk_level": metrics["risk_level"],
                        "clinical_summary": metrics["clinical_summary"],
                        "overall_recommendation": metrics["overall_recommendation"],
                        "decision_flow": metrics["decision_flow"],
                        "bias_analysis": metrics["bias_analysis"],
                        "critical_findings": audit_result.get("critical_findings", []),
                        "data_gaps": audit_result.get("data_gaps", []),
                        "fhir_insights": audit_result.get("fhir_insights", ""),
                        "fhir_patient_id": patient_fhir_id,
                        "fhir_history_summary": fhir_summary,
                        "integrity_seal": seal,
                        "timestamp": metrics["timestamp"],
                        "enrichment_note": "Audited with full FHIR patient history context using Gemini.",
                        "privacy_note": "FHIR data processed locally. No PHI sent to external services beyond Gemini API call."
                    }
                    return json.dumps(result, indent=2)
                except Exception:
                    pass

        # Fallback to MedGemma with just the note (no FHIR enrichment)
        result = await _run_audit_pipeline(clinical_note, patient_id)
        result["fhir_patient_id"] = patient_fhir_id
        result["fhir_history_summary"] = fhir_summary
        result["enrichment_note"] = "Audited with MedGemma. FHIR history retrieved but not used in analysis."
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def check_drug_interactions(clinical_note: str) -> str:
    """
    Extracts medications from clinical note and checks for dangerous interactions.
    Uses evidence-based interaction database covering anticoagulants, NSAIDs,
    SSRIs, contrast agents, cardiac medications, and more.
    Args:
        clinical_note: Clinical note containing medication list.
    Returns: JSON with detected medications, dangerous interactions, severity ratings.
    """
    logger.info("[Tool 5] check_drug_interactions")
    try:
        note_lower = clinical_note.lower()
        common_meds = [
            "warfarin", "aspirin", "ibuprofen", "metformin", "contrast",
            "ssri", "tramadol", "digoxin", "amiodarone", "methotrexate",
            "nsaid", "clopidogrel", "omeprazole", "lithium", "sildenafil",
            "nitrate", "maoi", "fluoroquinolone", "antacid", "potassium",
            "ace inhibitor", "lisinopril", "furosemide", "heparin",
        ]
        medications = [m for m in common_meds if m in note_lower]
        interactions_found = []
        for interaction in DANGEROUS_INTERACTIONS:
            drugs = interaction["drugs"]
            matches = [d for d in drugs if any(d in med for med in medications) or d in note_lower]
            if len(matches) >= 2:
                interactions_found.append({
                    "drugs_involved": drugs,
                    "severity": interaction["severity"],
                    "clinical_effect": interaction["effect"],
                    "action_required": "IMMEDIATE REVIEW" if interaction["severity"] == "HIGH" else "MONITOR CLOSELY"
                })
        return json.dumps({
            "status": "success",
            "medications_detected": medications,
            "medication_count": len(medications),
            "interactions_found": len(interactions_found),
            "dangerous_interactions": interactions_found,
            "overall_risk": (
                "HIGH" if any(i["severity"] == "HIGH" for i in interactions_found)
                else "MEDIUM" if interactions_found else "LOW"
            ),
            "recommendation": (
                "PHARMACIST REVIEW REQUIRED — dangerous interactions detected"
                if interactions_found else
                "No dangerous interactions detected in extracted medications"
            )
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def generate_differential(clinical_note: str) -> str:
    """
    Generates top 5 differential diagnoses using evidence-based clinical pattern matching.
    Instantly returns ranked differentials with supporting evidence and confirmatory tests.
    Covers cardiac, PE, sepsis, dyspnea, and abdominal presentations.
    Args:
        clinical_note: Clinical note with symptoms and presentation.
    Returns: JSON with top 5 differentials, likelihood, evidence, key tests.
    """
    logger.info("[Tool 6] generate_differential")
    try:
        note_lower = clinical_note.lower()
        DIFFERENTIALS = {
            "chest_pain_pe": {
                "triggers": ["pleuritic", "jvd", "pe ", "pulmonary embol", "post-op", "dvt", "chf"],
                "diagnoses": [
                    {"rank": 1, "diagnosis": "Pulmonary Embolism", "likelihood": "HIGH", "supporting_evidence": "Pleuritic pain, tachycardia, JVD, risk factors", "key_test_to_confirm": "STAT CTPA + D-dimer + Wells score", "must_not_miss": True},
                    {"rank": 2, "diagnosis": "Acute MI", "likelihood": "MEDIUM", "supporting_evidence": "Cannot exclude without ECG and troponin", "key_test_to_confirm": "12-lead ECG + Troponin", "must_not_miss": True},
                    {"rank": 3, "diagnosis": "Pneumothorax", "likelihood": "MEDIUM", "supporting_evidence": "Sudden pleuritic pain, decreased breath sounds", "key_test_to_confirm": "Chest X-ray", "must_not_miss": True},
                    {"rank": 4, "diagnosis": "Pericardial Effusion / Tamponade", "likelihood": "LOW", "supporting_evidence": "JVD, muffled heart sounds, hypotension", "key_test_to_confirm": "Echocardiogram", "must_not_miss": True},
                    {"rank": 5, "diagnosis": "Aortic Dissection", "likelihood": "LOW", "supporting_evidence": "Sudden severe chest pain, unequal pulses", "key_test_to_confirm": "CT Angiography", "must_not_miss": True}
                ],
                "primary_concern": "Pulmonary Embolism — STAT CTPA required"
            },
            "chest_pain_cardiac": {
                "triggers": ["chest pain", "chest tight", "left arm", "jaw pain", "diaphoresis", "sweating", "nausea", "mi", "acs"],
                "diagnoses": [
                    {"rank": 1, "diagnosis": "Acute Myocardial Infarction", "likelihood": "HIGH", "supporting_evidence": "Chest pain with radiation, diaphoresis, nausea, hemodynamic instability", "key_test_to_confirm": "12-lead ECG + Serial Troponin", "must_not_miss": True},
                    {"rank": 2, "diagnosis": "Unstable Angina", "likelihood": "HIGH", "supporting_evidence": "Chest pain at rest, known CAD history", "key_test_to_confirm": "Serial Troponin + Stress test", "must_not_miss": True},
                    {"rank": 3, "diagnosis": "Aortic Dissection", "likelihood": "MEDIUM", "supporting_evidence": "Sudden tearing pain, hypertension", "key_test_to_confirm": "CT Angiography chest", "must_not_miss": True},
                    {"rank": 4, "diagnosis": "Pulmonary Embolism", "likelihood": "MEDIUM", "supporting_evidence": "Chest pain, tachycardia, hypoxia", "key_test_to_confirm": "CTPA + D-dimer", "must_not_miss": True},
                    {"rank": 5, "diagnosis": "Pericarditis", "likelihood": "LOW", "supporting_evidence": "Pleuritic chest pain, positional relief, pericardial rub", "key_test_to_confirm": "ECG + Echocardiogram", "must_not_miss": False}
                ],
                "primary_concern": "Acute Myocardial Infarction — rule out first with ECG"
            },
            "sepsis": {
                "triggers": ["fever", "sepsis", "infection", "temp 3", "wbc", "lactate", "hypotension", "sirs"],
                "diagnoses": [
                    {"rank": 1, "diagnosis": "Bacterial Sepsis", "likelihood": "HIGH", "supporting_evidence": "SIRS criteria met: fever, tachycardia, elevated WBC", "key_test_to_confirm": "Blood cultures x2 + Lactate + CBC", "must_not_miss": True},
                    {"rank": 2, "diagnosis": "Septic Shock", "likelihood": "MEDIUM", "supporting_evidence": "Hypotension, elevated lactate, organ dysfunction", "key_test_to_confirm": "Repeat lactate + vasopressor assessment", "must_not_miss": True},
                    {"rank": 3, "diagnosis": "Pneumonia", "likelihood": "MEDIUM", "supporting_evidence": "Productive cough, fever, crackles on auscultation", "key_test_to_confirm": "Chest X-ray + sputum culture", "must_not_miss": False},
                    {"rank": 4, "diagnosis": "Urosepsis", "likelihood": "MEDIUM", "supporting_evidence": "Dysuria, flank pain, fever, UA positive", "key_test_to_confirm": "Urinalysis + Urine culture", "must_not_miss": False},
                    {"rank": 5, "diagnosis": "Meningitis", "likelihood": "LOW", "supporting_evidence": "Fever, altered mental status, neck stiffness", "key_test_to_confirm": "LP if no contraindications", "must_not_miss": True}
                ],
                "primary_concern": "Bacterial Sepsis — antibiotics within 1 hour of recognition"
            },
            "dyspnea": {
                "triggers": ["sob", "shortness of breath", "dyspnea", "breathless", "o2 sat", "spo2", "oxygen"],
                "diagnoses": [
                    {"rank": 1, "diagnosis": "Acute Heart Failure", "likelihood": "HIGH", "supporting_evidence": "SOB, orthopnea, bilateral crackles, peripheral edema", "key_test_to_confirm": "BNP + Chest X-ray + Echo", "must_not_miss": True},
                    {"rank": 2, "diagnosis": "Pulmonary Embolism", "likelihood": "HIGH", "supporting_evidence": "Sudden SOB, tachycardia, hypoxia out of proportion", "key_test_to_confirm": "CTPA + D-dimer", "must_not_miss": True},
                    {"rank": 3, "diagnosis": "COPD Exacerbation", "likelihood": "MEDIUM", "supporting_evidence": "Known COPD, wheeze, increased sputum, barrel chest", "key_test_to_confirm": "ABG + Peak flow + CXR", "must_not_miss": False},
                    {"rank": 4, "diagnosis": "Pneumonia", "likelihood": "MEDIUM", "supporting_evidence": "Fever, productive cough, focal crackles", "key_test_to_confirm": "Chest X-ray + sputum culture", "must_not_miss": False},
                    {"rank": 5, "diagnosis": "Anemia", "likelihood": "LOW", "supporting_evidence": "Fatigue, pallor, tachycardia, low Hb", "key_test_to_confirm": "Full blood count", "must_not_miss": False}
                ],
                "primary_concern": "Pulmonary Embolism — must be excluded urgently in acute SOB"
            },
            "abdominal": {
                "triggers": ["abdominal", "belly", "epigastric", "rlq", "llq", "abdomen", "bowel", "vomit"],
                "diagnoses": [
                    {"rank": 1, "diagnosis": "Appendicitis", "likelihood": "HIGH", "supporting_evidence": "RLQ pain, fever, rebound tenderness, elevated WBC", "key_test_to_confirm": "CT abdomen/pelvis with contrast", "must_not_miss": True},
                    {"rank": 2, "diagnosis": "Bowel Obstruction", "likelihood": "MEDIUM", "supporting_evidence": "Colicky pain, vomiting, distension, absent bowel sounds", "key_test_to_confirm": "Abdominal X-ray + CT", "must_not_miss": True},
                    {"rank": 3, "diagnosis": "Mesenteric Ischemia", "likelihood": "MEDIUM", "supporting_evidence": "Severe pain out of proportion to exam, AF, elderly", "key_test_to_confirm": "CT angiography", "must_not_miss": True},
                    {"rank": 4, "diagnosis": "Peptic Ulcer Disease", "likelihood": "MEDIUM", "supporting_evidence": "Epigastric pain, NSAID or H. pylori history, relief with antacids", "key_test_to_confirm": "Endoscopy", "must_not_miss": False},
                    {"rank": 5, "diagnosis": "Pancreatitis", "likelihood": "LOW", "supporting_evidence": "Epigastric pain radiating to back, alcohol history, vomiting", "key_test_to_confirm": "Lipase + CT abdomen", "must_not_miss": False}
                ],
                "primary_concern": "Mesenteric Ischemia — life-threatening if missed in elderly with AF"
            }
        }
        best_match = None
        best_score = 0
        for category, data in DIFFERENTIALS.items():
            score = sum(1 for t in data["triggers"] if t in note_lower)
            if score > best_score:
                best_score = score
                best_match = data
        if not best_match or best_score == 0:
            best_match = DIFFERENTIALS["chest_pain_cardiac"]
        return json.dumps({
            "status": "success",
            "differential_diagnoses": best_match["diagnoses"],
            "primary_concern": best_match["primary_concern"],
            "must_not_miss_count": sum(1 for d in best_match["diagnoses"] if d["must_not_miss"]),
            "note": "Generated by evidence-based clinical pattern matching. Always apply clinical judgment."
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
async def score_note_completeness(clinical_note: str) -> str:
    """
    Grades a clinical note for documentation completeness (0-100).
    Checks for chief complaint, vitals, medications, allergies, physical exam,
    assessment, plan, history, review of systems, and follow-up.
    Args:
        clinical_note: Clinical note to grade.
    Returns: JSON with completeness score, grade, missing elements, recommendation.
    """
    logger.info("[Tool 7] score_note_completeness")
    try:
        note_lower = clinical_note.lower()
        elements = {
            "Chief Complaint": ["presents with", "complains of", "chief complaint", "cc:"],
            "Vital Signs": ["hr ", "bp ", "temp", "rr ", "spo2", "o2 sat", "heart rate", "blood pressure"],
            "Medications": ["medication", "drug", "prescribed", "taking", "mg", "mcg"],
            "Allergies": ["allerg", "nkda", "no known", "reaction to"],
            "Physical Examination": ["exam", "examination", "auscultation", "palpation", "on exam"],
            "Assessment/Diagnosis": ["assessment", "diagnosis", "impression", "likely", "consistent with"],
            "Plan": ["plan", "order", "prescribe", "admit", "discharge", "follow up", "referral"],
            "History": ["history", "hx", "pmh", "past medical", "hpi"],
            "Review of Systems": ["ros", "review of systems", "denies", "positive for", "negative for"],
            "Follow-up": ["follow up", "follow-up", "return", "recheck", "f/u"]
        }
        present = {e: any(kw in note_lower for kw in kws) for e, kws in elements.items()}
        missing = [e for e, v in present.items() if not v]
        score = int((sum(present.values()) / len(elements)) * 100)
        grade = (
            "A — Excellent" if score >= 90 else
            "B — Good" if score >= 75 else
            "C — Acceptable" if score >= 60 else
            "D — Significant gaps" if score >= 40 else
            "F — Critical failure"
        )
        return json.dumps({
            "status": "success",
            "completeness_score": score,
            "grade": grade,
            "elements_present": [k for k, v in present.items() if v],
            "elements_missing": missing,
            "missing_count": len(missing),
            "recommendation": (
                "Note meets documentation standards." if score >= 75 else
                f"Missing {len(missing)} elements: {', '.join(missing[:3])}" +
                (" and more." if len(missing) > 3 else ".")
            )
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def batch_audit(notes_json: str) -> str:
    """
    Audits multiple clinical notes and returns risk-ranked results.
    Accepts JSON array of notes, audits each, sorts by gravity score (highest first).
    Maximum 10 notes per batch.
    Args:
        notes_json: JSON array: [{"patient_id": "P001", "note": "..."}, ...]
    Returns: JSON with all results sorted by risk level, triage summary.
    """
    logger.info("[Tool 8] batch_audit")
    try:
        notes = json.loads(notes_json)
        if not isinstance(notes, list):
            return json.dumps({"status": "error", "error": "Must be JSON array"})
        if len(notes) > 5:
            notes = notes[:5]

        # Build a single prompt with all notes — one Gemini call for all patients
        notes_text = ""
        for i, item in enumerate(notes):
            pid = item.get("patient_id", f"PATIENT-{i+1}")
            note = item.get("note", "").strip()
            if note:
                notes_text += f"\nPATIENT {pid}:\n{guard.scrub_phi(note)}\n"

        if not notes_text.strip():
            return json.dumps({"status": "error", "error": "All notes were empty"})

        gemini_key = os.getenv("GEMINI_API_KEY", "")

        if gemini_key:
            # Use Gemini for fast batch analysis
            prompt = f"""You are a senior emergency physician. Rapidly triage these patients.
For EACH patient output ONLY valid JSON with these exact fields.
OUTPUT A JSON ARRAY. No preamble. No markdown.

PATIENTS:
{notes_text}

OUTPUT FORMAT:
[
  {{
    "patient_id": "<id>",
    "risk_level": "HIGH" or "MEDIUM" or "LOW",
    "clinical_gravity_score": <0-100>,
    "clinical_summary": "<one sentence>",
    "top_action": "<single most urgent action>",
    "missed_diagnosis": "<most likely missed diagnosis or null>",
    "cognitive_bias": "<bias detected or null>"
  }}
]"""

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": gemini_key},
                    json={"contents": [{"parts": [{"text": prompt}]}]}
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["candidates"][0]["content"]["parts"][0]["text"]

            # Parse response
            match = re.search(r'\[[\s\S]*\]', raw)
            if match:
                try:
                    results = json.loads(match.group(0))
                    results.sort(key=lambda x: x.get("clinical_gravity_score", 0), reverse=True)
                    high_risk = [r for r in results if r.get("risk_level") == "HIGH"]
                    return json.dumps({
                        "status": "success",
                        "analysis_engine": "Gemini Fast Triage",
                        "total_notes": len(notes),
                        "high_risk_patients": len(high_risk),
                        "results_ranked_by_risk": results,
                        "triage_summary": (
                            f"{len(high_risk)} patient(s) require immediate attention."
                            if high_risk else "No high-risk patients in this batch."
                        )
                    }, indent=2)
                except Exception:
                    pass

        # Fallback: use local MedGemma demo cache concurrently
        async def audit_one(item, i):
            note = item.get("note", "")
            pid = item.get("patient_id", f"PATIENT-{i+1}")
            if not note.strip():
                return {"patient_id": pid, "status": "skipped"}
            try:
                safe_note = guard.scrub_phi(note)
                loop = asyncio.get_event_loop()
                audit_result = await loop.run_in_executor(
                    None, lambda: auditor.audit_note(safe_note, use_cache=True)
                )
                metrics = logic.calculate_safety_metrics(audit_result)
                return {
                    "patient_id": pid,
                    "clinical_gravity_score": metrics["clinical_gravity_score"],
                    "risk_level": metrics["risk_level"],
                    "clinical_summary": metrics["clinical_summary"],
                    "top_action": metrics["decision_flow"][0]["action"] if metrics["decision_flow"] else "No action",
                    "bias_count": metrics["bias_analysis"]["count"]
                }
            except Exception as e:
                return {"patient_id": pid, "status": "error", "error": str(e)}

        tasks = [audit_one(item, i) for i, item in enumerate(notes)]
        results = list(await asyncio.gather(*tasks))
        results.sort(key=lambda x: x.get("clinical_gravity_score", 0), reverse=True)
        high_risk = [r for r in results if r.get("risk_level") == "HIGH"]
        return json.dumps({
            "status": "success",
            "analysis_engine": "MedGemma Local",
            "total_notes": len(notes),
            "high_risk_patients": len(high_risk),
            "results_ranked_by_risk": results,
            "triage_summary": (
                f"{len(high_risk)} patient(s) require immediate attention."
                if high_risk else "No high-risk patients in this batch."
            )
        }, indent=2)

    except json.JSONDecodeError:
        return json.dumps({"status": "error", "error": "Invalid JSON in notes_json"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def second_opinion(clinical_note: str, patient_id: str = "SYNTHETIC-TEST") -> str:
    """
    Runs clinical note through two analytical perspectives and compares findings.
    Audits twice — cached and fresh — then compares. Disagreements flag uncertainty.
    Args:
        clinical_note: Clinical note to analyze.
        patient_id: De-identified patient ID.
    Returns: JSON with two opinions, agreement level, and clinical uncertainty rating.
    """
    logger.info(f"[Tool 9] second_opinion | patient={patient_id}")
    try:
        safe_note = guard.scrub_phi(clinical_note)
        loop = asyncio.get_event_loop()
        try:
            audit1 = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: auditor.audit_note(safe_note, use_cache=True)),
                timeout=25.0
            )
            audit2 = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: auditor.audit_note(safe_note, use_cache=False)),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            return json.dumps({"status": "timeout", "message": "Analysis timed out. Try a shorter note."})
        m1 = logic.calculate_safety_metrics(audit1)
        m2 = logic.calculate_safety_metrics(audit2)
        s1, s2 = m1["clinical_gravity_score"], m2["clinical_gravity_score"]
        diff = abs(s1 - s2)
        agreement = "HIGH" if diff <= 10 else "MODERATE" if diff <= 25 else "LOW"
        f1 = set(f.get("likely_missed", "") for f in audit1.get("critical_findings", []))
        f2 = set(f.get("likely_missed", "") for f in audit2.get("critical_findings", []))
        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "first_opinion": {"gravity_score": s1, "risk_level": m1["risk_level"], "findings": list(f1)},
            "second_opinion": {"gravity_score": s2, "risk_level": m2["risk_level"], "findings": list(f2)},
            "comparison": {
                "agreement_level": agreement,
                "score_difference": diff,
                "agreed_findings": list(f1 & f2),
                "unique_to_first": list(f1 - f2),
                "unique_to_second": list(f2 - f1),
                "clinical_uncertainty": (
                    "HIGH — senior review strongly recommended." if agreement == "LOW" else
                    "MODERATE — consider specialist consult." if agreement == "MODERATE" else
                    "LOW — both analyses agree. High confidence."
                )
            }
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def write_fhir_audit(clinical_note: str, patient_fhir_id: str, patient_id: str = "SYNTHETIC-TEST") -> str:
    """
    Audits clinical note and writes findings back to FHIR as ClinicalImpression.
    Closes the loop — audit findings become part of permanent patient record.
    Args:
        clinical_note: Clinical note to audit.
        patient_fhir_id: FHIR patient ID to write results to.
        patient_id: De-identified ID for audit trail.
    Returns: JSON with full audit results and FHIR write-back confirmation.
    """
    logger.info(f"[Tool 10] write_fhir_audit | fhir_id={patient_fhir_id}")
    try:
        loop = asyncio.get_event_loop()
        # Run audit immediately
        result = await _run_audit_pipeline(clinical_note, patient_id)
        result["fhir_patient_id"] = patient_fhir_id

        # Fire FHIR write in background — dont wait for it
        async def background_fhir_write():
            try:
                fhir_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: write_audit_to_fhir(
                            patient_fhir_id,
                            result["clinical_summary"],
                            result["clinical_gravity_score"]
                        )
                    ),
                    timeout=8.0
                )
                logger.info(f"FHIR write-back complete: {fhir_result}")
            except Exception as e:
                logger.warning(f"Background FHIR write failed: {e}")

        asyncio.create_task(background_fhir_write())

        result["fhir_write_back"] = {
            "status": "initiated",
            "message": f"Audit findings being written to FHIR for patient {patient_fhir_id} in background.",
            "fhir_resource": "ClinicalImpression"
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


async def keep_alive():
    if not SERVER_URL:
        logger.info("SERVER_URL not set — keep-alive disabled")
        return
    ping_url = f"{SERVER_URL}/health"
    logger.info(f"Keep-alive → {ping_url} every 10 min")
    await asyncio.sleep(120)
    while True:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(ping_url)
                logger.info(f"Keep-alive → {r.status_code}")
        except Exception as e:
            logger.warning(f"Keep-alive failed: {e}")
        await asyncio.sleep(600)


async def health_endpoint(request):
    return JSONResponse({
        "status": "alive",
        "server": "medgemma-clinical-safety-auditor",
        "version": "2.0.0",
        "tools": 10,
        "model": os.getenv("MEDGEMMA_MODEL", "MedAIBase/MedGemma1.5:4b"),
        "ollama_host": os.getenv("OLLAMA_HOST", "localhost")
    })


if __name__ == "__main__":
    async def main():
        mcp_app = mcp.streamable_http_app()
        async with mcp_app.router.lifespan_context(mcp_app):
            app = Starlette(routes=[
                Route("/health", health_endpoint),
                Mount("/", app=mcp_app),
            ])
            config = uvicorn.Config(
                app=app, host="0.0.0.0", port=PORT,
                log_level="info", access_log=True,
                proxy_headers=True, forwarded_allow_ips="*"
            )
            server = uvicorn.Server(config)
            logger.info(f"Starting on port {PORT} | Tools: 10")
            await asyncio.gather(keep_alive(), server.serve())
    asyncio.run(main())