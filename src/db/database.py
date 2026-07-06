import sqlite3
import json
import os

# Default database path (can be overridden by environment variable for Docker volumes)
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "helpdesk.db"))

def get_db_connection():
    """Returns a connection to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def initialize_database():
    """Creates tables if they do not exist."""
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        department TEXT NOT NULL
    );
    """)

    # 2. Devices table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS devices (
        device_id TEXT PRIMARY KEY,
        username TEXT NOT NULL,
        os TEXT NOT NULL,
        ip_address TEXT NOT NULL,
        vpn_configured INTEGER DEFAULT 0,
        last_patch_date TEXT NOT NULL,
        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
    );
    """)

    # 3. Account Status table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS account_status (
        username TEXT PRIMARY KEY,
        is_locked INTEGER DEFAULT 0,
        mfa_enabled INTEGER DEFAULT 0,
        mfa_method TEXT,
        password_expiry TEXT NOT NULL,
        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
    );
    """)

    # 4. Knowledge Base table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        topic TEXT NOT NULL UNIQUE,
        symptoms TEXT NOT NULL,  -- JSON list of symptom strings
        diagnostic_steps TEXT NOT NULL,  -- JSON list of diagnostic step strings
        resolution_steps TEXT NOT NULL,  -- JSON list of resolution step strings
        safety_warnings TEXT NOT NULL  -- Warning details / safety limits / uncertainty disclaimer
    );
    """)

    # 5. Tickets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        device_id TEXT,
        category TEXT NOT NULL,
        symptoms TEXT NOT NULL,
        steps_taken TEXT NOT NULL,  -- JSON log of diagnostic steps performed
        resolution_status TEXT NOT NULL CHECK(resolution_status IN ('Open', 'Resolved', 'Escalated')),
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
        resolved_at TEXT,
        FOREIGN KEY (username) REFERENCES users(username),
        FOREIGN KEY (device_id) REFERENCES devices(device_id)
    );
    """)

    conn.commit()
    conn.close()

def seed_database():
    """Seeds the database with initial users, devices, account statuses, and knowledge base articles."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Clear existing data to allow clean re-seeding
    cursor.execute("DELETE FROM tickets;")
    cursor.execute("DELETE FROM account_status;")
    cursor.execute("DELETE FROM devices;")
    cursor.execute("DELETE FROM users;")
    cursor.execute("DELETE FROM knowledge_base;")

    # --- Seed Users ---
    users_data = [
        ("alice_smith", "Alice Smith", "alice.smith@company.com", "Engineering"),
        ("bob_jones", "Bob Jones", "bob.jones@company.com", "Sales"),
        ("charlie_brown", "Charlie Brown", "charlie.brown@company.com", "HR"),
        ("diana_prince", "Diana Prince", "diana.prince@company.com", "Security Operations"),
    ]
    cursor.executemany("INSERT INTO users (username, full_name, email, department) VALUES (?, ?, ?, ?);", users_data)

    # --- Seed Devices ---
    devices_data = [
        ("LAPTOP-ALICE11", "alice_smith", "Windows 11 Enterprise", "192.168.1.45", 1, "2026-06-15"),
        ("MAC-BOB02", "bob_jones", "macOS Sequoia", "192.168.1.110", 0, "2026-05-10"),
        ("LAPTOP-CHARLIE09", "charlie_brown", "Windows 10 Pro", "10.0.0.12", 1, "2025-11-20"), # Outdated patches
        ("LAPTOP-DIANA07", "diana_prince", "Windows 11 Enterprise", "10.0.0.88", 1, "2026-07-01"),
    ]
    cursor.executemany("INSERT INTO devices (device_id, username, os, ip_address, vpn_configured, last_patch_date) VALUES (?, ?, ?, ?, ?, ?);", devices_data)

    # --- Seed Account Status ---
    # locked out accounts, expired passwords, etc.
    account_status_data = [
        ("alice_smith", 0, 1, "Authenticator App", "2026-10-01"),
        ("bob_jones", 1, 1, "SMS", "2026-08-15"), # Locked account
        ("charlie_brown", 0, 0, None, "2026-01-15"), # Password expired (relative to 2026-07-06)
        ("diana_prince", 0, 1, "FIDO2 Key", "2026-12-31"),
    ]
    cursor.executemany("INSERT INTO account_status (username, is_locked, mfa_enabled, mfa_method, password_expiry) VALUES (?, ?, ?, ?, ?);", account_status_data)

    # --- Seed Knowledge Base Articles ---
    kb_articles = [
        {
            "category": "network",
            "topic": "VPN Connection Failure",
            "symptoms": json.dumps(["vpn fail", "vpn disconnect", "cannot connect vpn", "anyconnect error", "forticlient error"]),
            "diagnostic_steps": json.dumps([
                "Check user internet connection by pinging a public host (e.g. ping 8.8.8.8).",
                "Verify if the device is running the latest VPN client software version.",
                "Verify the VPN Gateway address configuration is correct (e.g., gw.company.com).",
                "Verify user credentials and check if account is locked out."
            ]),
            "resolution_steps": json.dumps([
                "Restart the VPN client and reconnect.",
                "Verify split-tunneling is functioning by running traceroute to internal resources.",
                "If credentials are bad, assist the user in checking account lock or resetting their password.",
                "Escalate to Network Team if the gateway is unreachable but public ping succeeds."
            ]),
            "safety_warnings": "Warning: Ensure the user is not attempting to connect from a high-risk country flagged by geo-IP blocking. Escalation required if security blocks are triggered."
        },
        {
            "category": "network",
            "topic": "Slow Internet or Wi-Fi Disconnects",
            "symptoms": json.dumps(["slow wifi", "wifi disconnect", "dropped ping", "slow page load", "captive portal"]),
            "diagnostic_steps": json.dumps([
                "Run a ping loop to gateway to identify packet loss rates.",
                "Check if public DNS resolution fails (e.g., nslookup google.com).",
                "Verify if connected to guest Wi-Fi and pending captive portal authorization."
            ]),
            "resolution_steps": json.dumps([
                "Release and renew IP lease (ipconfig /release && ipconfig /renew).",
                "Forget Wi-Fi network and reconnect.",
                "Set DNS servers explicitly to 8.8.8.8 and 8.8.4.4 if corporate DNS is unreachable."
            ]),
            "safety_warnings": "Warning: Do not recommend disabling corporate firewalls or VPNs to fix local speed issues unless temporary diagnostics require it."
        },
        {
            "category": "account",
            "topic": "Active Directory Account Lockout",
            "symptoms": json.dumps(["account locked", "ad lock", "too many passwords", "wrong password lock", "user lock"]),
            "diagnostic_steps": json.dumps([
                "Query domain controller for bad password count and account lock status.",
                "Identify source device generating bad password attempts (usually cached credentials on mobile/laptop).",
                "Verify user identity via secondary authentication method (MFA verification)."
            ]),
            "resolution_steps": json.dumps([
                "Unlock account in Active Directory domain controller (requires admin/agent confirmation).",
                "Instruct user to update cached passwords in Windows Credential Manager and mobile email clients.",
                "If unlock fails repeatedly, escalate to Identity & Access Management (IAM) team."
            ]),
            "safety_warnings": "Important: Do not unlock accounts without verifying user identity first. Repeated locks indicate credential hijacking or automated brute-forcing. Escalate immediately if suspicious activity is detected."
        },
        {
            "category": "account",
            "topic": "MFA and Password Setup",
            "symptoms": json.dumps(["reset password", "mfa reset", "new authenticator", "backup codes", "expired password"]),
            "diagnostic_steps": json.dumps([
                "Check password expiration date in account status.",
                "Verify MFA device enrollment status.",
                "Determine if password change needs self-service password reset (SSPR)."
            ]),
            "resolution_steps": json.dumps([
                "Trigger MFA device registration link or temporary bypass code (requires MFA validation).",
                "Provide self-service password reset URL (https://sspr.company.com).",
                "Verify status after password reset."
            ]),
            "safety_warnings": "Important: Multi-Factor Authentication resets are highly sensitive. You must request explicit manager approval or visual ID check if secondary contact information is changed. Escalation required for MFA token replacement."
        },
        {
            "category": "application",
            "topic": "Operating System / Blue Screen Crashes",
            "symptoms": json.dumps(["blue screen", "bsod", "system crash", "kernel panic", "critical process died"]),
            "diagnostic_steps": json.dumps([
                "Retrieve Stop Code from user report (e.g., SYSTEM_THREAD_EXCEPTION_NOT_HANDLED, PAGE_FAULT_IN_NONPAGED_AREA).",
                "Check device patch level and last update installation history.",
                "Inspect device crash dump logs (C:\\Windows\\Minidump) if accessible."
            ]),
            "resolution_steps": json.dumps([
                "Reboot in Safe Mode and check if stable.",
                "Roll back recently installed device drivers or cumulative updates.",
                "Run built-in hardware diagnostics (SFC /scannow and DISM tool).",
                "Escalate to Hardware Repair if hardware diagnostic reports bad memory or disk blocks."
            ]),
            "safety_warnings": "Warning: Do NOT advise opening the computer case or performing physical hardware modification. Triage only. High-risk crashes with smoke or physical heat signs must be escalated to site services immediately."
        },
        {
            "category": "application",
            "topic": "Software Update Failures",
            "symptoms": json.dumps(["update fail", "windows update error", "patch failed", "disk space error", "0x8024"]),
            "diagnostic_steps": json.dumps([
                "Check system drive free disk space.",
                "Retrieve specific update error code (e.g., 0x80244007).",
                "Check Windows Update service status."
            ]),
            "resolution_steps": json.dumps([
                "Run disk cleanup tools to free at least 15GB of space.",
                "Stop Windows Update service, clear SoftwareDistribution folder cache, and restart service.",
                "Run manual KB download installer if online check fails."
            ]),
            "safety_warnings": "Decision Support Rule: Verify if the system is critical server or production database. Update failures on production servers must be handled by the Server Ops team, not helpdesk."
        }
    ]

    for article in kb_articles:
        cursor.execute("""
        INSERT INTO knowledge_base (category, topic, symptoms, diagnostic_steps, resolution_steps, safety_warnings)
        VALUES (:category, :topic, :symptoms, :diagnostic_steps, :resolution_steps, :safety_warnings);
        """, article)

    # --- Seed Initial Tickets (for testing history query) ---
    tickets_data = [
        ("alice_smith", "LAPTOP-ALICE11", "network", "VPN disconnects every 10 minutes", 
         json.dumps(["Checked internet: Stable ping", "VPN configuration: Verified"]), "Resolved", "2026-07-01 09:30:00", "2026-07-01 10:15:00"),
        ("bob_jones", "MAC-BOB02", "account", "Account lockout lock notification shown", 
         json.dumps(["Account checked: is_locked = 1"]), "Open", "2026-07-06 11:00:00", None)
    ]
    cursor.executemany("""
    INSERT INTO tickets (username, device_id, category, symptoms, steps_taken, resolution_status, created_at, resolved_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, tickets_data)

    conn.commit()
    conn.close()
    print("Database initialized and seeded successfully.")

if __name__ == "__main__":
    initialize_database()
    seed_database()
