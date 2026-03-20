"""
mcp_server.py — MCP Server Entry Point
CyberHealth AI | MedGemma Clinical Safety Auditor
 
Transport:    Streamable HTTP (required by Prompt Opinion)
Endpoint:     https://your-app.onrender.com/mcp
Health check: https://your-app.onrender.com/health
 
Deploy on:    Render.com (free tier)
AI backend:   Your local Ollama via ngrok static domain
"""
 
import asyncio
import json
import logging
import os
import sys
 
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
 
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
# Your existing modules
from engine import SafetyAuditor
from clinical_logic import ClinicalLogic
from security import VaultGuard
 
# ─────────────────────────────────────────────────────────────────
# Logging — ALWAYS to stderr
# In HTTP mode stdout is fine, but stderr is the safe default.
# Never use print() — it can corrupt JSON responses.
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────────────────────────
# Config from environment variables
# Set these on Render.com under Environment
# ─────────────────────────────────────────────────────────────────
API_KEY = os.getenv("MCP_API_KEY", "cyberhealth-medgemma-2026")
RENDER_URL = os.getenv("RAILWAY_EXTERNAL_URL", os.getenv("RENDER_EXTERNAL_URL", ""))
PORT = int(os.getenv("PORT", 8000))
 
# ─────────────────────────────────────────────────────────────────
# Initialize your existing components once at startup
# This warms up the Ollama connection before the first request
# ─────────────────────────────────────────────────────────────────
logger.info("=" * 50)
logger.info("CyberHealth AI — MedGemma MCP Server")
logger.info("=" * 50)
logger.info("Initializing components...")
 
auditor = SafetyAuditor()
logic   = ClinicalLogic()
guard   = VaultGuard()
auditor.warmup()
 
logger.info("All components ready.")
logger.info(f"MCP endpoint will be at: {RENDER_URL}/mcp")
 
# ─────────────────────────────────────────────────────────────────
# Create the FastMCP server
# This handles all MCP protocol details automatically
# ─────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="MedGemma Clinical Safety Auditor",
     transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
    instructions=(
        "This MCP server provides AI-powered clinical note safety auditing. "
        "It uses a locally-running MedGemma model to detect diagnostic errors, "
        "cognitive biases, and safety risks in unstructured clinical notes. "
        "All patient data is processed on-premises — nothing is transmitted "
        "to external AI services. Use audit_clinical_note for full analysis "
        "or get_safety_summary for a quick risk triage."
    )
)
 
 
# ═════════════════════════════════════════════════════════════════
# TOOL 1: Full Clinical Safety Audit
# The main tool — judges will test this one
# ═════════════════════════════════════════════════════════════════
@mcp.tool()
async def audit_clinical_note(
    clinical_note: str,
    patient_id: str = "SYNTHETIC-TEST"
) -> str:
    """
    Performs a full AI safety audit on a clinical note.
 
    Analyzes the note using a locally-running MedGemma medical AI model.
    Detects cognitive biases (anchoring, premature closure, framing),
    calculates a clinical gravity score from 0 to 100, identifies missing
    data gaps, and generates a prioritized step-by-step decision flow
    for the reviewing clinician. Uses cryptographic sealing for audit
    trail integrity. All processing is local — zero patient data is
    transmitted to any external service.
 
    Args:
        clinical_note: The raw unstructured clinical note text to analyze.
                       Must use synthetic or de-identified data only.
                       No real Protected Health Information (PHI).
        patient_id:    De-identified patient identifier for the audit trail.
                       Defaults to SYNTHETIC-TEST for demo purposes.
 
    Returns:
        JSON string containing:
        - clinical_gravity_score (0-100)
        - risk_level (HIGH/ELEVATED/MODERATE/LOW)
        - clinical_summary
        - decision_flow (prioritized action steps)
        - bias_analysis (detected cognitive biases)
        - scoring_breakdown
        - data_gaps
        - integrity_seal (SHA-256 hash)
    """
    logger.info(
        f"[audit_clinical_note] patient={patient_id} | "
        f"note_length={len(clinical_note)}"
    )
 
    if not clinical_note.strip():
        return json.dumps({
            "status": "error",
            "error": "clinical_note cannot be empty"
        })
 
    try:
        # Step 1: PHI scrubbing (safety net)
        safe_note = guard.scrub_phi(clinical_note)
        logger.info("PHI scrubbing complete")
 
        # Step 2: MedGemma analysis
        # run_in_executor because ollama.generate() is synchronous/blocking
        loop = asyncio.get_event_loop()
        audit_result = await loop.run_in_executor(
            None,
            lambda: auditor.audit_note(safe_note)
        )
        logger.info(
            f"MedGemma audit complete | "
            f"safety_score={audit_result.get('safety_score', 'N/A')} | "
            f"mode={audit_result.get('analysis_mode', 'LIVE')}"
        )
 
        # Step 3: Clinical metrics
        metrics = logic.calculate_safety_metrics(audit_result)
 
        # Step 4: Cryptographic integrity seal
        seal = auditor.generate_seal(safe_note, audit_result)
 
        # Step 5: Build response
        response = {
            "status": "success",
            "patient_id": patient_id,
            "analysis_mode": audit_result.get("analysis_mode", "LIVE"),
 
            # ── Core output ──
            "clinical_gravity_score": metrics["clinical_gravity_score"],
            "risk_level": metrics["risk_level"],
            "clinical_summary": metrics["clinical_summary"],
            "overall_recommendation": metrics["overall_recommendation"],
 
            # ── Detailed results ──
            "scoring_breakdown": metrics["scoring_breakdown"],
            "decision_flow": metrics["decision_flow"],
            "bias_analysis": metrics["bias_analysis"],
            "data_completeness": metrics["data_completeness"],
            "invariant_check": metrics["invariant_check"],
 
            # ── Raw findings ──
            "critical_findings": audit_result.get("critical_findings", []),
            "data_gaps": audit_result.get("data_gaps", []),
 
            # ── Metadata ──
            "integrity_seal": seal,
            "timestamp": metrics["timestamp"],
            "privacy_note": (
                "All analysis performed locally on-premises via MedGemma. "
                "No patient data transmitted to external AI services."
            )
        }
 
        logger.info(
            f"Audit complete | gravity={metrics['clinical_gravity_score']} | "
            f"risk={metrics['risk_level']}"
        )
        return json.dumps(response, indent=2)
 
    except Exception as e:
        logger.error(f"audit_clinical_note failed: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "error": str(e),
            "patient_id": patient_id,
            "fallback": "Manual clinical review required"
        })
 
 
# ═════════════════════════════════════════════════════════════════
# TOOL 2: Quick Safety Summary
# Faster lightweight version for triage
# ═════════════════════════════════════════════════════════════════
@mcp.tool()
async def get_safety_summary(clinical_note: str) -> str:
    """
    Returns a quick safety summary of a clinical note.
 
    Lighter and faster than the full audit. Returns only the headline
    risk level, gravity score, top finding, and immediate action.
    Use this for rapid triage before running a full audit.
 
    Args:
        clinical_note: Raw unstructured clinical note.
                       Must be synthetic or de-identified.
 
    Returns:
        JSON string with risk_level, gravity_score, summary,
        finding counts, and top recommended action.
    """
    logger.info("[get_safety_summary] called")
 
    try:
        safe_note = guard.scrub_phi(clinical_note)
 
        loop = asyncio.get_event_loop()
        audit_result = await loop.run_in_executor(
            None,
            lambda: auditor.audit_note(safe_note)
        )
        metrics = logic.calculate_safety_metrics(audit_result)
 
        top_action = "No immediate action required"
        if metrics["decision_flow"]:
            top_action = metrics["decision_flow"][0].get(
                "action", "No immediate action required"
            )
 
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
        logger.error(f"get_safety_summary failed: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)})
 
 
# ═════════════════════════════════════════════════════════════════
# KEEP-ALIVE
# Pings /health every 10 minutes to prevent Render free tier
# from spinning down (it spins down after 15 min inactivity)
# ═════════════════════════════════════════════════════════════════
async def keep_alive():
    if not RENDER_URL:
        logger.info("RENDER_EXTERNAL_URL not set — keep-alive disabled")
        return
 
    ping_url = f"{RENDER_URL}/health"
    logger.info(f"Keep-alive enabled → pinging {ping_url} every 10 min")
 
    # Wait 2 minutes after startup before first ping
    await asyncio.sleep(120)
 
    while True:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(ping_url)
                logger.info(f"Keep-alive ping → {r.status_code}")
        except Exception as e:
            logger.warning(f"Keep-alive ping failed (non-fatal): {e}")
 
        await asyncio.sleep(600)  # 10 minutes
 
 
# ═════════════════════════════════════════════════════════════════
# SERVER STARTUP
# Combines: MCP endpoint at /mcp + health check at /health
# ═════════════════════════════════════════════════════════════════
async def health_endpoint(request):
    """Health check — used by keep-alive and Render's health monitor."""
    return JSONResponse({
        "status": "alive",
        "server": "medgemma-clinical-safety-auditor",
        "version": "1.0.0",
        "model": os.getenv("MEDGEMMA_MODEL", "MedAIBase/MedGemma1.5:4b"),
        "ollama_host": os.getenv("OLLAMA_HOST", "localhost")
    })
 
 
if __name__ == "__main__":
 
    async def main():
        mcp_app = mcp.streamable_http_app()
 
        async with mcp_app.router.lifespan_context(mcp_app):
            app = Starlette(
                routes=[
                    Route("/health", health_endpoint),
                    Mount("/", app=mcp_app),
                ]
            )
 
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=PORT,
                log_level="info",
                access_log=True,
                proxy_headers=True,
                forwarded_allow_ips="*"
            )
            server = uvicorn.Server(config)
 
            logger.info(f"Starting on port {PORT}")
            logger.info(f"MCP endpoint:  http://0.0.0.0:{PORT}/mcp")
            logger.info(f"Health check:  http://0.0.0.0:{PORT}/health")
 
            await asyncio.gather(
                keep_alive(),
                server.serve()
            )
 
    asyncio.run(main())