import unittest
import os
import sys
import sqlite3
import json

# Setup sys.path to resolve src/ packages
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Setup test DB path environment variable
TEST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_helpdesk_tools.db")
os.environ["DB_PATH"] = TEST_DB_PATH

from db.database import initialize_database, seed_database, get_db_connection
from tools.helpdesk_tools import (
    query_knowledge_base,
    lookup_user,
    match_intent_from_symptoms,
    analyze_diagnostic_input,
    execute_helpdesk_action,
    append_ticket_step,
    generate_resolution_report
)

class TestHelpdeskTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        initialize_database()

    def setUp(self):
        # Fresh seed for every test to guarantee clean state
        seed_database()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    # 1. Test Information Tool
    def test_query_kb_by_category(self):
        """Test retrieving KB articles filtered by category."""
        articles = query_knowledge_base(category="network")
        self.assertGreater(len(articles), 0)
        topics = [a["topic"] for a in articles]
        self.assertIn("VPN Connection Failure", topics)
        self.assertIn("Slow Internet or Wi-Fi Disconnects", topics)

    def test_query_kb_by_topic_search(self):
        """Test search by keyword in topics."""
        articles = query_knowledge_base(topic="Active Directory")
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["topic"], "Active Directory Account Lockout")
        self.assertGreater(len(articles[0]["diagnostic_steps"]), 0)

    # 2. Test Analysis Tool
    def test_analysis_ping_loss(self):
        """Test parsing ping output with loss and success."""
        ping_failure_log = """
        Pinging 8.8.8.8 with 32 bytes of data:
        Request timed out.
        Request timed out.
        Request timed out.
        Request timed out.
        Ping statistics for 8.8.8.8:
            Packets: Sent = 4, Received = 0, Lost = 4 (100% loss)
        """
        res = analyze_diagnostic_input("ping", ping_failure_log)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["details"]["packet_loss_pct"], 100)

        ping_success_log = """
        Pinging 8.8.8.8 with 32 bytes of data:
        Reply from 8.8.8.8: bytes=32 time=12ms TTL=118
        Reply from 8.8.8.8: bytes=32 time=10ms TTL=118
        Ping statistics for 8.8.8.8:
            Packets: Sent = 2, Received = 2, Lost = 0 (0% loss)
        """
        res2 = analyze_diagnostic_input("ping", ping_success_log)
        self.assertEqual(res2["status"], "healthy")
        self.assertEqual(res2["details"]["packet_loss_pct"], 0)

    def test_analysis_account_checks(self):
        """Test account lock and expiration checks in database."""
        # Test locked user (bob_jones is locked in mock data)
        res = analyze_diagnostic_input("account_check", "", username="bob_jones")
        self.assertEqual(res["status"], "error")
        self.assertTrue(res["details"]["is_locked"])

        # Test expired password (charlie_brown is expired)
        res2 = analyze_diagnostic_input("account_check", "", username="charlie_brown")
        self.assertEqual(res2["status"], "error")
        self.assertTrue(res2["details"]["password_expired"])

        # Test healthy user (alice_smith)
        res3 = analyze_diagnostic_input("account_check", "", username="alice_smith")
        self.assertEqual(res3["status"], "healthy")
        self.assertFalse(res3["details"]["is_locked"])
        self.assertFalse(res3["details"]["password_expired"])

    def test_analysis_disk_space(self):
        """Test disk space capacity checking."""
        # Under threshold (15GB)
        res = analyze_diagnostic_input("disk_space", "5.4 GB free")
        self.assertEqual(res["status"], "error")
        self.assertTrue(res["details"]["below_threshold"])

        # Over threshold
        res2 = analyze_diagnostic_input("disk_space", "120 GB remaining")
        self.assertEqual(res2["status"], "healthy")
        self.assertFalse(res2["details"]["below_threshold"])

    def test_analysis_error_codes(self):
        """Test parsing of update or kernel crash error signatures."""
        res = analyze_diagnostic_input("error_code", "PAGE_FAULT_IN_NONPAGED_AREA")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["details"]["error_type"], "kernel_bsod")

        res2 = analyze_diagnostic_input("error_code", "Windows error code 0x80244007")
        self.assertEqual(res2["status"], "error")
        self.assertEqual(res2["details"]["error_type"], "windows_update_error")

    # 3. Test Action Tool
    def test_action_unlock_confirmation_gate(self):
        """Test that sensitive unlock operations require explicit confirmation."""
        # Unconfirmed
        res = execute_helpdesk_action("unlock_account", "bob_jones", confirmed=False)
        self.assertEqual(res["status"], "pending_confirmation")
        self.assertIn("Please confirm", res["message"])

        # Confirmed
        res2 = execute_helpdesk_action("unlock_account", "bob_jones", confirmed=True)
        self.assertEqual(res2["status"], "success")

        # Verify DB lock state updated
        conn = get_db_connection()
        status = conn.execute("SELECT is_locked FROM account_status WHERE username='bob_jones';").fetchone()
        conn.close()
        self.assertEqual(status["is_locked"], 0)

    def test_action_create_ticket(self):
        """Test logging diagnostic incidents as database tickets."""
        details = {
            "device_id": "LAPTOP-ALICE11",
            "category": "network",
            "symptoms": "VPN connection drops",
            "steps_taken": ["Ping test run: OK", "Credentials checked: OK"]
        }
        res = execute_helpdesk_action("create_ticket", "alice_smith", details=details)
        self.assertEqual(res["status"], "success")
        self.assertIsNotNone(res["ticket_id"])

        # Verify ticket details saved
        conn = get_db_connection()
        ticket = conn.execute("SELECT * FROM tickets WHERE ticket_id=?;", (res["ticket_id"],)).fetchone()
        conn.close()
        self.assertEqual(ticket["username"], "alice_smith")
        self.assertEqual(ticket["resolution_status"], "Open")
        self.assertIn("Ping test run", ticket["steps_taken"])

    # 4. Test Reporting Tool
    def test_generate_resolution_report(self):
        """Test generating diagnostic summary reports."""
        # Create ticket
        details = {
            "device_id": "MAC-BOB02",
            "category": "account",
            "symptoms": "Locked out of email client",
            "steps_taken": ["AD lockout check: Lockout detected", "User verified via SMS MFA"]
        }
        res = execute_helpdesk_action("create_ticket", "bob_jones", details=details)
        ticket_id = res["ticket_id"]

        # Resolve ticket
        execute_helpdesk_action("resolve_ticket", "bob_jones", details={"ticket_id": ticket_id})

        # Generate report
        report = generate_resolution_report(ticket_id)
        self.assertIn("IT Help-Desk Incident Resolution Report", report)
        self.assertIn("Ticket ID**: #", report)
        self.assertIn("Status**: **RESOLVED**", report)
        self.assertIn("Bob Jones", report)
        self.assertIn("macOS Sequoia", report) # Joins device table OS
        self.assertIn("Safety Disclaimer", report) # Renders safety section

    # 5. Test lookup_user
    def test_lookup_user_found(self):
        """Test looking up a valid user returns their profile and device."""
        res = lookup_user("alice_smith")
        self.assertTrue(res["found"])
        self.assertEqual(res["username"], "alice_smith")
        self.assertEqual(res["full_name"], "Alice Smith")
        self.assertEqual(res["department"], "Engineering")
        self.assertEqual(res["device_id"], "LAPTOP-ALICE11")
        self.assertEqual(res["os"], "Windows 11 Enterprise")
        self.assertTrue(res["vpn_configured"])

    def test_lookup_user_not_found(self):
        """Test looking up a non-existent user returns found=False."""
        res = lookup_user("ghost_user")
        self.assertFalse(res["found"])
        self.assertIn("not found", res["message"])

    # 6. Test match_intent_from_symptoms
    def test_intent_matching_vpn(self):
        """Test symptom keyword matching correctly identifies VPN intent."""
        res = match_intent_from_symptoms("I cannot connect to vpn it keeps disconnecting")
        self.assertEqual(res["category"], "network")
        self.assertIn("VPN", res["topic"])
        self.assertGreater(res["score"], 0)

    def test_intent_matching_lockout(self):
        """Test symptom keyword matching correctly identifies account lockout intent."""
        res = match_intent_from_symptoms("my account is locked and I cannot log in")
        self.assertEqual(res["category"], "account")
        self.assertIn("Lockout", res["topic"])

    def test_intent_matching_unknown(self):
        """Test unknown input returns category=unknown."""
        res = match_intent_from_symptoms("the coffee machine is broken")
        self.assertEqual(res["category"], "unknown")
        self.assertIsNone(res["topic"])

    # 7. Test append_ticket_step
    def test_append_ticket_step(self):
        """Test that diagnostic steps are incrementally appended to a ticket."""
        # Create initial ticket with empty steps
        res = execute_helpdesk_action("create_ticket", "diana_prince", details={
            "device_id": "LAPTOP-DIANA07",
            "category": "application",
            "symptoms": "Blue screen on boot",
            "steps_taken": []
        })
        ticket_id = res["ticket_id"]

        # Append two steps incrementally
        r1 = append_ticket_step(ticket_id, "Queried BSOD KB article.")
        r2 = append_ticket_step(ticket_id, "Error code PAGE_FAULT_IN_NONPAGED_AREA detected.")

        self.assertEqual(r1["status"], "success")
        self.assertEqual(r1["steps_count"], 1)
        self.assertEqual(r2["steps_count"], 2)

        # Verify DB reflects both steps
        conn = get_db_connection()
        row = conn.execute("SELECT steps_taken FROM tickets WHERE ticket_id=?;", (ticket_id,)).fetchone()
        conn.close()
        steps = json.loads(row["steps_taken"])
        self.assertIn("Queried BSOD KB article.", steps)
        self.assertIn("Error code PAGE_FAULT_IN_NONPAGED_AREA detected.", steps)

    # 8. Test escalate_ticket action
    def test_escalate_ticket(self):
        """Test that a ticket can be escalated with a reason appended to steps log."""
        res = execute_helpdesk_action("create_ticket", "charlie_brown", details={
            "device_id": "LAPTOP-CHARLIE09",
            "category": "application",
            "symptoms": "Windows update fails repeatedly",
            "steps_taken": ["Disk space checked: OK", "Update cache cleared: no change"]
        })
        ticket_id = res["ticket_id"]

        esc_res = execute_helpdesk_action("escalate_ticket", "charlie_brown", details={
            "ticket_id": ticket_id,
            "reason": "Issue persists after 5 diagnostic iterations. Requires Tier-2 OS specialist."
        })

        self.assertEqual(esc_res["status"], "success")
        self.assertIn("Tier-2", esc_res["message"])

        # Verify DB status changed to Escalated
        conn = get_db_connection()
        row = conn.execute("SELECT resolution_status, steps_taken FROM tickets WHERE ticket_id=?;", (ticket_id,)).fetchone()
        conn.close()
        self.assertEqual(row["resolution_status"], "Escalated")
        steps = json.loads(row["steps_taken"])
        self.assertTrue(any("[ESCALATED]" in s for s in steps))

if __name__ == "__main__":
    unittest.main()
