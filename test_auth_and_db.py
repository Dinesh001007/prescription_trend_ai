import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.db import (
    init_db,
    register_user,
    authenticate_user,
    get_user_by_id,
    create_conversation,
    get_user_conversations,
    get_conversation,
    update_conversation_title,
    delete_conversation,
    add_message,
    get_conversation_messages,
    save_memory,
    get_user_memories,
    delete_memory,
    clear_all_memories,
    create_analysis_session,
    get_user_analysis_sessions,
    get_analysis_session,
    update_analysis_session_data,
    delete_analysis_session,
    add_analysis_message,
    get_analysis_messages,
)
from utils.memory_engine import (
    build_conversation_prompt,
    generate_chat_title,
    extract_and_save_clinical_memory
)
from utils.llm import query_chat_llm

class TestAuthAndDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_registration_and_auth(self):
        unique_suffix = str(uuid.uuid4())[:8]
        test_user = f"doc_{unique_suffix}"
        test_email = f"doc_{unique_suffix}@hospital.org"
        test_pwd = "SecurePassword123!"

        # Register
        ok, msg, udata = register_user(test_user, test_email, test_pwd, full_name="Dr. Alex Rivera", role="Cardiologist")
        self.assertTrue(ok, f"Registration failed: {msg}")
        self.assertIsNotNone(udata)
        self.assertEqual(udata["username"], test_user)

        # Authenticate with username
        ok, msg, user = authenticate_user(test_user, test_pwd)
        self.assertTrue(ok)
        self.assertEqual(user["email"], test_email)

        # Authenticate with email
        ok, msg, user2 = authenticate_user(test_email, test_pwd)
        self.assertTrue(ok)
        self.assertEqual(user2["id"], user["id"])

        # Authenticate with bad password
        ok, msg, _ = authenticate_user(test_user, "WrongPassword")
        self.assertFalse(ok)

    def test_conversation_and_messages(self):
        unique_suffix = str(uuid.uuid4())[:8]
        # Register user
        _, _, user = register_user(f"conv_{unique_suffix}", f"conv_{unique_suffix}@med.org", "password123", full_name="Dr. Sarah")
        uid = user["id"]
        
        # Create conversation
        conv = create_conversation(uid, title="Hypertension Consultation")
        conv_id = conv["id"]
        self.assertEqual(conv["title"], "Hypertension Consultation")

        # Add messages
        msg1 = add_message(conv_id, "user", "What is the recommended starting dose of Amlodipine?")
        msg2 = add_message(conv_id, "assistant", "The standard initial dose is 5 mg orally once daily.")
        
        # Retrieve messages
        history = get_conversation_messages(conv_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

        # Update title
        update_conversation_title(conv_id, "Amlodipine Dosage Study")
        updated_conv = get_conversation(conv_id, uid)
        self.assertEqual(updated_conv["title"], "Amlodipine Dosage Study")

        # List user conversations
        convs = get_user_conversations(uid)
        self.assertTrue(any(c["id"] == conv_id for c in convs))

        # Delete conversation
        del_ok = delete_conversation(conv_id, uid)
        self.assertTrue(del_ok)
        empty_history = get_conversation_messages(conv_id)
        self.assertEqual(len(empty_history), 0)

    def test_memories_and_prompt_building(self):
        unique_suffix = str(uuid.uuid4())[:8]
        _, _, user = register_user(f"mem_{unique_suffix}", f"mem_{unique_suffix}@med.org", "password123")
        uid = user["id"]
        clear_all_memories(uid)

        # Save memory
        mem1 = save_memory(uid, "Patient is 62yo female with CKD Stage 3b and Sulfa allergy")
        self.assertIsNotNone(mem1["id"])

        memories = get_user_memories(uid)
        self.assertEqual(len(memories), 1)
        self.assertIn("Sulfa allergy", memories[0]["content"])

        # Test Prompt Builder
        messages = [
            {"role": "user", "content": "Can I prescribe Bactrim?"},
            {"role": "assistant", "content": "Bactrim contains sulfamethoxazole and trimethoprim."},
            {"role": "user", "content": "What alternatives can I use instead?"}
        ]
        prompt = build_conversation_prompt(messages, user_memories=memories)
        self.assertIn("Sulfa allergy", prompt)
        self.assertIn("Bactrim contains", prompt)
        self.assertIn("What alternatives can I use instead?", prompt)

        # Test query_chat_llm (fallback or live)
        response = query_chat_llm(messages, user_memories=memories)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 10)

    def test_analysis_sessions_and_safe_delete(self):
        unique_suffix = str(uuid.uuid4())[:8]
        _, _, user = register_user(f"analyst_{unique_suffix}", f"analyst_{unique_suffix}@clinic.org", "pass123456", role="Medical Data Scientist")
        uid = user["id"]

        # 1. Create Dataset Analysis session
        ds_session = create_analysis_session(
            user_id=uid,
            session_type="dataset_analysis",
            title="Q3 Prescription Audit",
            filename="prescriptions_q3.csv",
            data_dict={"row_count": 1500, "risk_high_pct": 14.2}
        )
        self.assertIsNotNone(ds_session["id"])
        self.assertEqual(ds_session["session_type"], "dataset_analysis")
        self.assertEqual(ds_session["title"], "Q3 Prescription Audit")

        # 2. Add analysis contextual chat messages
        add_analysis_message(ds_session["id"], "user", "What is the primary risk driver in this dataset?")
        add_analysis_message(ds_session["id"], "assistant", "High dosage of opioid analgesics combined with benzodiazepines.")

        msgs = get_analysis_messages(ds_session["id"])
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")

        # 3. Update session data
        update_analysis_session_data(
            session_id=ds_session["id"],
            title="Q3 Prescription Audit (Validated)",
            data_dict={"validated": True}
        )
        fetched = get_analysis_session(ds_session["id"], uid)
        self.assertEqual(fetched["title"], "Q3 Prescription Audit (Validated)")

        # 4. Create Image Report session
        img_session = create_analysis_session(
            user_id=uid,
            session_type="image_report",
            title="Chest CT Scan",
            filename="scan_patient_99.png",
            data_dict={"modality": "CT", "findings": "No acute consolidation."}
        )
        self.assertIsNotNone(img_session["id"])

        # 5. List sessions
        ds_list = get_user_analysis_sessions(uid, "dataset_analysis")
        img_list = get_user_analysis_sessions(uid, "image_report")
        self.assertEqual(len(ds_list), 1)
        self.assertEqual(len(img_list), 1)

        # 6. Safely delete session and verify complete removal
        del_res = delete_analysis_session(ds_session["id"], uid)
        self.assertTrue(del_res)

        # Ensure session and messages are deleted
        self.assertIsNone(get_analysis_session(ds_session["id"], uid))
        self.assertEqual(len(get_analysis_messages(ds_session["id"])), 0)
        self.assertEqual(len(get_user_analysis_sessions(uid, "dataset_analysis")), 0)


if __name__ == "__main__":
    unittest.main()
