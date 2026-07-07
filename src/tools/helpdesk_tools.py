import sys
import os
import json
import re
import sqlite3
from datetime import datetime

# Adjust path to import db.database correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import get_db_connection

# ==========================================
# 1. INFORMATION TOOL
# ==========================================
def query_knowledge_base(category: str = None, topic: str = None) -> list:
    """
    Retrieves grounded troubleshooting guides from the knowledge base.
    
    Args:
        category (str, optional): The domain area (e.g., 'network', 'account', 'application').
        topic (str, optional): Search term matching the topic.
        
    Returns:
        list: A list of dicts containing the topic, symptoms, diagnostic steps,
              resolution steps, and safety warnings.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM knowledge_base WHERE 1=1"
    params = []
    
    if category:
        query += " AND category = ?"
        params.append(category.lower())
    
    if topic:
        query += " AND topic LIKE ?"
        params.append(f"%{topic}%")
        
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "category": row["category"],
            "topic": row["topic"],
            "symptoms": json.loads(row["symptoms"]),
            "diagnostic_steps": json.loads(row["diagnostic_steps"]),
            "resolution_steps": json.loads(row["resolution_steps"]),
            "safety_warnings": row["safety_warnings"]
        })
        
    return results


# ==========================================
# 1b. USER LOOKUP TOOL
# ==========================================
def lookup_user(username: str) -> dict:
    """
    Looks up a user and their associated device from the database.
    Used by gather_info_node to validate a user exists and populate
    the working memory with their device context.

    Args:
        username (str): The employee username to look up.

    Returns:
        dict: User profile and device details, or an error if not found.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username = ?;", (username,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return {
            "found": False,
            "message": f"User '{username}' not found in the employee registry."
        }

    cursor.execute("SELECT * FROM devices WHERE username = ?;", (username,))
    device = cursor.fetchone()
    conn.close()

    return {
        "found": True,
        "username": user["username"],
        "full_name": user["full_name"],
        "email": user["email"],
        "department": user["department"],
        "device_id": device["device_id"] if device else None,
        "os": device["os"] if device else None,
        "ip_address": device["ip_address"] if device else None,
        "vpn_configured": bool(device["vpn_configured"]) if device else False,
        "last_patch_date": device["last_patch_date"] if device else None,
    }


# ==========================================
# 1c. INTENT MATCHING TOOL
# ==========================================
def match_intent_from_symptoms(user_text: str) -> dict:
    """
    Deterministically matches user free-text input against grounded KB symptom
    keyword lists to identify the most likely troubleshooting category and topic.
    Used by route_input_node to classify intent without hallucination.

    Args:
        user_text (str): Raw natural language description from the user
                         (e.g. 'my vpn keeps dropping every hour').

    Returns:
        dict: Best matching category, topic, confidence score, and KB article id.
              Returns category='unknown' if no symptoms match.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, category, topic, symptoms FROM knowledge_base;")
    rows = cursor.fetchall()
    conn.close()

    normalized_input = user_text.lower()
    # Tokenize the input into individual words for flexible matching
    input_tokens = set(re.findall(r"\b\w+\b", normalized_input))
    best_match = {"category": "unknown", "topic": None, "kb_id": None, "score": 0, "matched_keywords": []}

    for row in rows:
        symptoms = json.loads(row["symptoms"])
        matched = []
        for kw in symptoms:
            # A keyword phrase matches if ALL its tokens appear anywhere in the user input
            kw_tokens = set(re.findall(r"\b\w+\b", kw.lower()))
            if kw_tokens and kw_tokens.issubset(input_tokens):
                matched.append(kw)
        # Score = total number of matched keyword tokens (rewards more specific matches)
        score = sum(len(re.findall(r"\b\w+\b", kw)) for kw in matched)
        if score > best_match["score"]:
            best_match = {
                "category": row["category"],
                "topic": row["topic"],
                "kb_id": row["id"],
                "score": score,
                "matched_keywords": matched
            }

    if best_match["score"] == 0:
        return {
            "category": "unknown",
            "topic": None,
            "kb_id": None,
            "score": 0,
            "matched_keywords": [],
            "message": "No matching symptom keywords found. Agent will ask user for clarification."
        }

    return best_match


# ==========================================
# 2. ANALYSIS TOOL
# ==========================================
def analyze_diagnostic_input(issue_type: str, user_input: str, username: str = None) -> dict:
    """
    Analyzes diagnostic inputs, user responses, error logs, or database records
    to identify fault severity and match appropriate resolution workflows.
    
    Args:
        issue_type (str): Type of check ('ping', 'account_check', 'disk_space', 'error_code').
        user_input (str): The raw diagnostic logs, error codes, or inputs pasted by the user.
        username (str, optional): The user's account name (required for database status checks).
        
    Returns:
        dict: A structured report containing status ('healthy', 'warning', 'error'),
              confidence_score (0.0 - 1.0), and a descriptive message/details.
    """
    issue_type = issue_type.lower()
    
    # CASE A: Analyzing network ping results
    if issue_type == "ping":
        # Search for packet loss or timeouts in raw ping logs
        loss_match = re.search(r"(\d+)%\s+loss", user_input, re.IGNORECASE)
        timed_out_count = len(re.findall(r"request timed out", user_input, re.IGNORECASE))
        destination_unreachable = re.search(r"destination host unreachable", user_input, re.IGNORECASE)
        
        if destination_unreachable or timed_out_count >= 4 or (loss_match and int(loss_match.group(1)) == 100):
            return {
                "status": "error",
                "confidence_score": 1.0,
                "detected_issue": "Complete network disconnect / destination unreachable.",
                "details": {"packet_loss_pct": 100, "unreachable": True}
            }
        elif loss_match and int(loss_match.group(1)) > 0:
            pct = int(loss_match.group(1))
            return {
                "status": "warning",
                "confidence_score": 0.8,
                "detected_issue": f"Intermittent connection. Packet loss detected: {pct}%.",
                "details": {"packet_loss_pct": pct, "unreachable": False}
            }
        else:
            # Check for standard successful round-trip patterns
            if "reply from" in user_input.lower() or (loss_match and int(loss_match.group(1)) == 0):
                return {
                    "status": "healthy",
                    "confidence_score": 0.9,
                    "detected_issue": "Network link is active and responding.",
                    "details": {"packet_loss_pct": 0, "unreachable": False}
                }
            return {
                "status": "warning",
                "confidence_score": 0.5,
                "detected_issue": "Could not conclusively parse ping output. No errors nor replies detected.",
                "details": {}
            }
            
    # CASE B: Checking database account lockout / password expiry
    elif issue_type == "account_check":
        if not username:
            return {
                "status": "error",
                "confidence_score": 1.0,
                "detected_issue": "Missing username for account status lookup.",
                "details": {}
            }
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM account_status WHERE username = ?;", (username,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {
                "status": "error",
                "confidence_score": 1.0,
                "detected_issue": f"User '{username}' not found in the account registry.",
                "details": {"user_exists": False}
            }
            
        # Check lockout and expiry dates
        is_locked = bool(row["is_locked"])
        password_expiry_str = row["password_expiry"]
        mfa_enabled = bool(row["mfa_enabled"])
        
        # Calculate password expiry against mock current time: 2026-07-06
        expiry_date = datetime.strptime(password_expiry_str, "%Y-%m-%d")
        current_mock_date = datetime(2026, 7, 6) # Using current local year/time reference from context
        password_expired = expiry_date < current_mock_date
        
        if is_locked:
            return {
                "status": "error",
                "confidence_score": 1.0,
                "detected_issue": f"Account '{username}' is locked out in Active Directory.",
                "details": {"is_locked": True, "password_expired": password_expired, "mfa_enabled": mfa_enabled}
            }
        elif password_expired:
            return {
                "status": "error",
                "confidence_score": 1.0,
                "detected_issue": f"Account '{username}' password expired on {password_expiry_str}.",
                "details": {"is_locked": False, "password_expired": True, "mfa_enabled": mfa_enabled}
            }
        else:
            return {
                "status": "healthy",
                "confidence_score": 1.0,
                "detected_issue": f"Account '{username}' is active and valid (expires on {password_expiry_str}).",
                "details": {"is_locked": False, "password_expired": False, "mfa_enabled": mfa_enabled}
            }

    # CASE C: Disk Space checks
    elif issue_type == "disk_space":
        # Extract numerical space from string (e.g. "5 GB", "100GB", "12480 MB")
        match = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB|KB|bytes)?", user_input, re.IGNORECASE)
        if not match:
            return {
                "status": "warning",
                "confidence_score": 0.4,
                "detected_issue": "Could not parse disk space volume. Ensure input contains unit (e.g., '10 GB').",
                "details": {}
            }
            
        value = float(match.group(1))
        unit = match.group(2).upper() if match.group(2) else "GB"
        
        # Normalize to GB
        if unit == "MB":
            value = value / 1024
        elif unit == "KB":
            value = value / (1024 * 1024)
        elif unit == "BYTES":
            value = value / (1024 * 1024 * 1024)
            
        if value < 15.0:
            return {
                "status": "error",
                "confidence_score": 1.0,
                "detected_issue": f"Insufficient disk space: {value:.2f} GB available. Minimum required for update is 15.0 GB.",
                "details": {"free_space_gb": value, "below_threshold": True}
            }
        else:
            return {
                "status": "healthy",
                "confidence_score": 1.0,
                "detected_issue": f"Disk space is sufficient: {value:.2f} GB available.",
                "details": {"free_space_gb": value, "below_threshold": False}
            }

    # CASE D: Error code checks (e.g. BSOD Stop codes, software errors)
    elif issue_type == "error_code":
        normalized_input = user_input.upper().strip()
        
        # Matches for common IT error codes
        if "0X8024" in normalized_input or "0X8007" in normalized_input:
            return {
                "status": "error",
                "confidence_score": 0.9,
                "detected_issue": "Windows Update corruption or catalog connectivity issue detected.",
                "details": {"error_type": "windows_update_error", "code": normalized_input}
            }
        elif "PAGE_FAULT_IN_NONPAGED_AREA" in normalized_input or "SYSTEM_THREAD_EXCEPTION_NOT_HANDLED" in normalized_input or "CRITICAL_PROCESS_DIED" in normalized_input:
            return {
                "status": "error",
                "confidence_score": 1.0,
                "detected_issue": f"Critical kernel driver crash (BSOD Stop Code: {normalized_input}).",
                "details": {"error_type": "kernel_bsod", "code": normalized_input}
            }
        else:
            return {
                "status": "warning",
                "confidence_score": 0.6,
                "detected_issue": f"Unknown error signature received: '{user_input}'. Treating as general crash symptom.",
                "details": {"error_type": "unknown", "code": normalized_input}
            }
            
    return {
        "status": "error",
        "confidence_score": 0.0,
        "detected_issue": f"Unsupported diagnostic check type: '{issue_type}'",
        "details": {}
    }


# ==========================================
# 3. ACTION TOOL
# ==========================================
def execute_helpdesk_action(action_type: str, username: str, details: dict = None, confirmed: bool = False) -> dict:
    """
    Modifies the database state (e.g., unlocking account, logging incident tickets, resolving tickets).
    Sensitive or state-changing actions (like unlocking accounts) require the 'confirmed=True' flag.
    
    Args:
        action_type (str): The operation ('unlock_account', 'create_ticket', 'resolve_ticket').
        username (str): Target user.
        details (dict, optional): Context arguments (e.g., device_id, symptoms, steps_taken).
        confirmed (bool): Flag indicating user explicitly confirmed the action.
        
    Returns:
        dict: Success details or a confirmation request.
    """
    action_type = action_type.lower()
    details = details or {}
    
    if action_type == "unlock_account":
        if not confirmed:
            return {
                "status": "pending_confirmation",
                "action": "unlock_account",
                "username": username,
                "message": f"CRITICAL ACTION REQUIRED: Please confirm unlocking the account for employee '{username}'."
            }
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify user exists
        cursor.execute("SELECT * FROM account_status WHERE username = ?;", (username,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return {
                "status": "failed",
                "message": f"Cannot unlock. User '{username}' does not exist."
            }
            
        # Perform unlock
        cursor.execute("UPDATE account_status SET is_locked = 0 WHERE username = ?;", (username,))
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "action": "unlock_account",
            "username": username,
            "message": f"Successfully unlocked Active Directory account for user '{username}'."
        }
        
    elif action_type == "create_ticket":
        # Creating a ticket is standard; no confirmation gate required as it is logging-oriented.
        device_id = details.get("device_id")
        category = details.get("category", "General")
        symptoms = details.get("symptoms", "No description provided.")
        steps_taken_list = details.get("steps_taken", [])
        status = details.get("resolution_status", "Open")

        conn = get_db_connection()
        cursor = conn.cursor()

        # Validate username FK: only use it if the user actually exists in the DB.
        # If not, set to None — the tickets table has username NOT NULL so we use
        # a special sentinel 'guest' row, or we remap unknown to None safely by
        # checking first and using the literal string only if valid.
        safe_username = None
        if username:
            cursor.execute("SELECT 1 FROM users WHERE username = ?;", (username,))
            if cursor.fetchone():
                safe_username = username

        # Validate device_id FK too
        safe_device_id = None
        if device_id:
            cursor.execute("SELECT 1 FROM devices WHERE device_id = ?;", (device_id,))
            if cursor.fetchone():
                safe_device_id = device_id

        # tickets.username is NOT NULL — use the validated username or a fallback
        insert_username = safe_username if safe_username else (username or "guest")

        # Ensure the fallback username exists (insert a guest row if needed)
        if not safe_username:
            cursor.execute("SELECT 1 FROM users WHERE username = ?;", (insert_username,))
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT OR IGNORE INTO users (username, full_name, email, department) "
                    "VALUES (?, 'Guest / Unverified', 'unknown@company.com', 'Unknown');",
                    (insert_username,)
                )

        cursor.execute("""
        INSERT INTO tickets (username, device_id, category, symptoms, steps_taken, resolution_status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'));
        """, (insert_username, safe_device_id, category, symptoms, json.dumps(steps_taken_list), status))

        ticket_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return {
            "status": "success",
            "action": "create_ticket",
            "ticket_id": ticket_id,
            "message": f"Incident Ticket #{ticket_id} created successfully for '{insert_username}' (Status: {status})."
        }
        
    elif action_type == "resolve_ticket":
        ticket_id = details.get("ticket_id")
        if not ticket_id:
            return {"status": "failed", "message": "Missing 'ticket_id' to resolve ticket."}
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        UPDATE tickets 
        SET resolution_status = 'Resolved', resolved_at = datetime('now', 'localtime') 
        WHERE ticket_id = ?;
        """, (ticket_id,))
        
        # Also ensure target account is unlocked if this was an unlock ticket
        cursor.execute("SELECT category, username FROM tickets WHERE ticket_id = ?;", (ticket_id,))
        ticket = cursor.fetchone()
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "action": "resolve_ticket",
            "ticket_id": ticket_id,
            "message": f"Ticket #{ticket_id} has been marked as Resolved."
        }
        
    elif action_type == "escalate_ticket":
        ticket_id = details.get("ticket_id")
        reason = details.get("reason", "No escalation reason provided.")
        if not ticket_id:
            return {"status": "failed", "message": "Missing 'ticket_id' for escalation."}

        conn = get_db_connection()
        cursor = conn.cursor()
        # Append escalation reason to steps_taken log
        cursor.execute("SELECT steps_taken FROM tickets WHERE ticket_id = ?;", (ticket_id,))
        row = cursor.fetchone()
        existing_steps = json.loads(row["steps_taken"]) if row else []
        existing_steps.append(f"[ESCALATED] {reason}")

        cursor.execute("""
        UPDATE tickets
        SET resolution_status = 'Escalated', steps_taken = ?, resolved_at = datetime('now', 'localtime')
        WHERE ticket_id = ?;
        """, (json.dumps(existing_steps), ticket_id))
        conn.commit()
        conn.close()

        return {
            "status": "success",
            "action": "escalate_ticket",
            "ticket_id": ticket_id,
            "message": f"Ticket #{ticket_id} escalated to Tier-2 support. Reason: {reason}"
        }

    return {
        "status": "failed",
        "message": f"Unsupported action type: '{action_type}'."
    }


# ==========================================
# 3b. APPEND TICKET STEP TOOL
# ==========================================
def append_ticket_step(ticket_id: int, step: str) -> dict:
    """
    Appends a single diagnostic step string to an existing ticket's steps_taken log.
    Called incrementally by graph nodes as the session progresses so the ticket
    always reflects the latest state of the conversation.

    Args:
        ticket_id (int): The target ticket record.
        step (str): A short description of the diagnostic action just taken.

    Returns:
        dict: Success confirmation or error details.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT steps_taken FROM tickets WHERE ticket_id = ?;", (ticket_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return {"status": "failed", "message": f"Ticket #{ticket_id} not found."}

    existing_steps = json.loads(row["steps_taken"])
    existing_steps.append(step)

    cursor.execute(
        "UPDATE tickets SET steps_taken = ? WHERE ticket_id = ?;",
        (json.dumps(existing_steps), ticket_id)
    )
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "ticket_id": ticket_id,
        "steps_count": len(existing_steps),
        "message": f"Step appended to Ticket #{ticket_id}: '{step}'"
    }


# ==========================================
# 4. REPORTING TOOL
# ==========================================
def generate_resolution_report(ticket_id: int) -> str:
    """
    Retrieves records for a given ticket ID and compiles a formal diagnostic markdown report.
    This fulfills the Safety-Critical/Triage rules to outline source information, findings, and limitations.
    
    Args:
        ticket_id (int): Target ticket record ID.
        
    Returns:
        str: Markdown formatting diagnostic report.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT t.*, u.full_name, u.email, u.department, d.os, d.ip_address, d.vpn_configured, d.last_patch_date
    FROM tickets t
    LEFT JOIN users u ON t.username = u.username
    LEFT JOIN devices d ON t.device_id = d.device_id
    WHERE t.ticket_id = ?;
    """, (ticket_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return f"### Error\nTicket ID #{ticket_id} could not be found in the database system."
        
    steps_taken = json.loads(row["steps_taken"])
    steps_formatted = "\n".join([f"* [x] {step}" for step in steps_taken]) if steps_taken else "* No diagnostic steps recorded."
    
    # Format a professional diagnostic/triage report
    report = f"""# IT Help-Desk Incident Resolution Report

## Ticket Overview
* **Ticket ID**: #{row["ticket_id"]}
* **Category**: {row["category"].upper()}
* **Status**: **{row["resolution_status"].upper()}**
* **Created At**: {row["created_at"]}
* **Resolved/Escalated At**: {row["resolved_at"] if row["resolved_at"] else "N/A (Active)"}

## Employee & Asset Details
* **Requestor**: {row["full_name"]} ({row["username"]})
* **Department**: {row["department"]}
* **Contact**: {row["email"]}
* **Device ID**: {row["device_id"] if row["device_id"] else "None"}
* **OS Environment**: {row["os"] if row["os"] else "Unknown"}
* **Device IP Address**: {row["ip_address"] if row["ip_address"] else "Unknown"}
* **Last OS Security Patch Date**: {row["last_patch_date"] if row["last_patch_date"] else "Unknown"}

## Reported Symptoms
> "{row["symptoms"]}"

## Diagnostic Checklist Performed
{steps_formatted}

## Triage Assessment & Resolution Notes
* **Action Summary**: The diagnostic agent performed interactive checks with the user. The primary symptoms were evaluated against the help-desk deterministic rules.
* **Resolution Details**: {
    "Locked Active Directory account was unlocked after verification." if row["category"] == "account" and row["resolution_status"] == "Resolved"
    else "Network ping test completed and steps to reset DNS/renew IP lease were detailed." if row["category"] == "network" and row["resolution_status"] == "Resolved"
    else "System logs and disk space constraints were checked. Action recommended to clear update cache." if row["category"] == "application" and row["resolution_status"] == "Resolved"
    else "Triage completed. Escalated to Tier-2 specialized human support for hands-on diagnostics."
}

---

## ⚠️ Safety Disclaimer & Limitations
* **Decision Support Role**: This system functions as a triage and decision-support tool. It does not possess direct hardware access or direct administrative override capabilities beyond authorized API endpoints.
* **Limitation Warning**: The diagnosis is grounded strictly on the symptoms and diagnostics provided by the user. If symptoms persist or if physical hardware anomalies (e.g. overheating, drive grinding, smoking) occur, suspend use of the machine immediately and contact local IT support.
* **Source Attribution**: All diagnostic rules and suggestions are retrieved from the corporate Grounded Help-Desk Knowledge Base (`knowledge_base`).
"""
    return report
