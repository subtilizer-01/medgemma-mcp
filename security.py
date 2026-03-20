"""
security.py — VaultGuard
CyberHealth AI | MedGemma Clinical Safety Auditor

Improvements over original:
  - bcrypt replaces SHA-256 (industry standard password hashing)
  - Expanded PHI scrubbing patterns (emails, phones, MRNs, dates of birth)
  - Timing-safe password comparison
  - Audit log with action tracking
"""

import sqlite3
import hashlib
import re
import bcrypt
from datetime import datetime


class VaultGuard:
    def __init__(self):
        self.db = 'sovereign_vault.db'
        self._init_db()

    def _init_db(self):
        """Initialize SQLite tables."""
        with sqlite3.connect(self.db) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS users
                            (username TEXT PRIMARY KEY,
                             password TEXT,
                             destruction_key TEXT,
                             hint TEXT,
                             created_at TEXT)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS records
                            (doc_id TEXT, pid TEXT, timestamp TEXT,
                             findings TEXT, seal TEXT)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS audit
                            (doc_id TEXT, action TEXT, timestamp TEXT,
                             detail TEXT)''')

    # ─────────────────────────────────────────────────────────────
    # PHI SCRUBBING
    # Removes identifying information before any AI processing.
    # This is a safety net — real data should never enter at all.
    # ─────────────────────────────────────────────────────────────
    def scrub_phi(self, text: str) -> str:
        """
        Redacts common PHI patterns from clinical text.
        Covers: names, CNICs, MRNs, phone numbers, emails,
                dates of birth, SSNs, IP addresses.
        """
        if not text:
            return text

        # Pakistani CNIC formats (12345-1234567-1 or 13 digits)
        text = re.sub(
            r'\b\d{5}-\d{7}-\d\b|\b\d{13}\b',
            '[REDACTED-CNIC]', text
        )

        # US Social Security Numbers (123-45-6789)
        text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[REDACTED-SSN]', text
        )

        # Medical Record Numbers (MRN: followed by digits)
        text = re.sub(
            r'\bMRN\s*:?\s*\d+\b',
            '[REDACTED-MRN]', re.IGNORECASE
        ) if False else re.sub(
            r'(?i)\bMRN\s*:?\s*\d+\b',
            '[REDACTED-MRN]', text
        )

        # Phone numbers (various formats)
        text = re.sub(
            r'\b(\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            '[REDACTED-PHONE]', text
        )

        # Email addresses
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[REDACTED-EMAIL]', text
        )

        # Dates of birth (common formats: 01/01/1990, Jan 1 1990, 1990-01-01)
        text = re.sub(
            r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b',
            '[REDACTED-DOB]', text
        )
        text = re.sub(
            r'\b(19|20)\d{2}-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b',
            '[REDACTED-DOB]', text
        )

        # IP addresses
        text = re.sub(
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            '[REDACTED-IP]', text
        )

        # Full names (First Last or First Middle Last)
        # Conservative pattern — only catches capitalized name patterns
        text = re.sub(
            r'\b[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,2}\b',
            '[REDACTED-NAME]', text
        )

        return text

    # ─────────────────────────────────────────────────────────────
    # USER MANAGEMENT
    # ─────────────────────────────────────────────────────────────
    def _hash_password(self, password: str) -> str:
        """
        bcrypt hashing — industry standard for passwords.
        Unlike SHA-256, bcrypt is slow by design (prevents brute force)
        and includes a random salt automatically.
        """
        return bcrypt.hashpw(
            password.encode('utf-8'),
            bcrypt.gensalt(rounds=12)
        ).decode('utf-8')

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Timing-safe password comparison using bcrypt."""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed.encode('utf-8')
            )
        except Exception:
            return False

    def manage_user(self, username, password,
                    destruction_key=None, hint=None, mode="login"):
        """Register or login users."""
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()

            if mode.lower() in ["register", "signup"]:
                try:
                    hashed = self._hash_password(password)
                    c.execute(
                        "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                        (username, hashed, destruction_key, hint,
                         datetime.now().isoformat())
                    )
                    conn.commit()
                    self._log_audit(username, "REGISTER", "Account created")
                    return True, "Account created successfully."
                except sqlite3.IntegrityError:
                    return False, "Username already exists."

            # Login
            c.execute(
                "SELECT password FROM users WHERE username = ?",
                (username,)
            )
            res = c.fetchone()
            if res and self._verify_password(password, res[0]):
                self._log_audit(username, "LOGIN", "Successful login")
                return True, "Access granted."
            self._log_audit(username, "LOGIN_FAIL", "Invalid credentials")
            return False, "Invalid credentials."

    def get_recovery_hint(self, username):
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT hint FROM users WHERE username = ?", (username,)
            )
            res = c.fetchone()
            return res[0] if res else "No recovery hint set."

    def verify_wipe_authority(self, username, destruction_key_input):
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT destruction_key FROM users WHERE username = ?",
                (username,)
            )
            res = c.fetchone()
            return res and res[0] == destruction_key_input

    def check_pid_exists(self, pid):
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT pid FROM records WHERE pid = ?", (pid,)
            )
            return c.fetchone() is not None

    def purge_vault(self, doc_id):
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "DELETE FROM records WHERE doc_id = ?", (doc_id,)
            )
            conn.commit()
        self._log_audit(doc_id, "PURGE", "Records deleted")

    def save_audit_record(self, doc_id: str, pid: str,
                          findings: str, seal: str):
        """Persist an audit record to the vault."""
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                (doc_id, pid, datetime.now().isoformat(), findings, seal)
            )
            conn.commit()

    def _log_audit(self, doc_id: str, action: str, detail: str = ""):
        """Internal audit trail logging."""
        try:
            with sqlite3.connect(self.db) as conn:
                conn.execute(
                    "INSERT INTO audit VALUES (?, ?, ?, ?)",
                    (doc_id, action, datetime.now().isoformat(), detail)
                )
                conn.commit()
        except Exception:
            pass  # Audit logging must never crash the main flow
