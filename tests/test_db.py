import unittest
import sqlite3
import json
import os
import sys

# Add src/ to sys.path so we can import db.database
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Override DB_PATH for testing
TEST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_helpdesk.db")
os.environ["DB_PATH"] = TEST_DB_PATH

from db.database import initialize_database, seed_database, get_db_connection

class TestDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Clean up any residual test database before starting
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        initialize_database()
        seed_database()

    @classmethod
    def tearDownClass(cls):
        # Clean up after all tests complete
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    def test_database_tables_exist(self):
        """Verify that all required tables exist in the schema."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]
        conn.close()

        expected_tables = ["users", "devices", "account_status", "knowledge_base", "tickets"]
        for table in expected_tables:
            self.assertIn(table, tables)

    def test_seeded_users(self):
        """Verify that mock users were inserted correctly."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users;")
        users = cursor.fetchall()
        conn.close()

        self.assertEqual(len(users), 4)
        usernames = [u["username"] for u in users]
        self.assertIn("alice_smith", usernames)
        self.assertIn("bob_jones", usernames)

    def test_seeded_devices(self):
        """Verify device records and their relation to users."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM devices WHERE username='alice_smith';")
        alice_device = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(alice_device)
        self.assertEqual(alice_device["device_id"], "LAPTOP-ALICE11")
        self.assertEqual(alice_device["vpn_configured"], 1)

    def test_seeded_account_statuses(self):
        """Verify account lock and password expiry details."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM account_status WHERE username='bob_jones';")
        bob_status = cursor.fetchone()
        cursor.execute("SELECT * FROM account_status WHERE username='charlie_brown';")
        charlie_status = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(bob_status)
        self.assertEqual(bob_status["is_locked"], 1) # Bob should be locked out
        
        self.assertIsNotNone(charlie_status)
        self.assertEqual(charlie_status["mfa_enabled"], 0) # Charlie has no MFA enabled

    def test_knowledge_base_grounding(self):
        """Verify that we can retrieve grounding steps for a topic."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM knowledge_base WHERE topic='VPN Connection Failure';")
        vpn_article = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(vpn_article)
        self.assertEqual(vpn_article["category"], "network")
        
        # Deserialize JSON fields
        symptoms = json.loads(vpn_article["symptoms"])
        diagnostic_steps = json.loads(vpn_article["diagnostic_steps"])
        
        self.assertIn("vpn fail", symptoms)
        self.assertGreater(len(diagnostic_steps), 0)
        self.assertIn("Check user internet connection by pinging a public host (e.g. ping 8.8.8.8).", diagnostic_steps)

    def test_foreign_key_violations(self):
        """Verify foreign key constraints restrict invalid records."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Attempt to insert a device with a non-existent username
        with self.assertRaises(sqlite3.IntegrityError):
            cursor.execute("""
            INSERT INTO devices (device_id, username, os, ip_address, vpn_configured, last_patch_date)
            VALUES ('LAPTOP-BAD', 'ghost_user', 'Windows 11', '10.0.0.1', 0, '2026-01-01');
            """)
            conn.commit()
            
        conn.close()

    def test_create_and_resolve_ticket(self):
        """Verify ticket creation and status update cycle."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create a new ticket
        cursor.execute("""
        INSERT INTO tickets (username, device_id, category, symptoms, steps_taken, resolution_status)
        VALUES ('alice_smith', 'LAPTOP-ALICE11', 'network', 'VPN failed to authenticate', '[]', 'Open');
        """)
        ticket_id = cursor.lastrowid
        
        # Check it exists
        cursor.execute("SELECT * FROM tickets WHERE ticket_id=?;", (ticket_id,))
        ticket = cursor.fetchone()
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket["resolution_status"], "Open")
        
        # Resolve the ticket
        cursor.execute("""
        UPDATE tickets 
        SET resolution_status='Resolved', resolved_at=datetime('now', 'localtime') 
        WHERE ticket_id=?;
        """, (ticket_id,))
        
        cursor.execute("SELECT * FROM tickets WHERE ticket_id=?;", (ticket_id,))
        ticket = cursor.fetchone()
        self.assertEqual(ticket["resolution_status"], "Resolved")
        self.assertIsNotNone(ticket["resolved_at"])
        
        conn.close()

if __name__ == "__main__":
    unittest.main()
