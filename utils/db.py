import sqlite3
import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medical_platform.db")


def get_db_connection():
    """Create and return a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables with appropriate schemas and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT DEFAULT 'Clinician',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. Conversations Table (ChatGPT style threads)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)

    # 3. Messages Table (Individual turns inside a conversation)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )
    """)

    # 4. User Memories Table (Persistent cross-chat long-term memory)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        conversation_id TEXT,
        memory_type TEXT DEFAULT 'clinical_fact',
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)

    # 5. Analysis Sessions Table (for Dataset Analysis & Image Report history)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_sessions (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        session_type TEXT NOT NULL,
        title TEXT NOT NULL,
        filename TEXT DEFAULT '',
        data_json TEXT DEFAULT '{}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)

    # 6. Analysis Messages Table (Contextual chat turns on specific datasets or scan reports)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES analysis_sessions(id) ON DELETE CASCADE
    )
    """)

    # Create Indexes for fast querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at ASC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON user_memories(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_sessions_user ON analysis_sessions(user_id, session_type, updated_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_messages_session ON analysis_messages(session_id, created_at ASC)")

    conn.commit()
    conn.close()


# ─── Password Security ────────────────────────────────────────────────────────

def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash password using PBKDF2 with SHA-256 and secure random salt."""
    if not salt:
        salt = secrets.token_hex(16)
    
    pwd_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    key = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, 100000)
    return key.hex(), salt


def _verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Verify a plain password against the stored hash and salt."""
    pwd_hash, _ = _hash_password(password, salt)
    return secrets.compare_digest(pwd_hash, expected_hash)


# ─── User Authentication Functions ───────────────────────────────────────────

def register_user(
    username: str, 
    email: str, 
    password: str, 
    full_name: str = "", 
    role: str = "Clinician"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Register a new user in the database.
    Returns: (success: bool, message: str, user_dict: Optional[dict])
    """
    username = username.strip().lower()
    email = email.strip().lower()
    full_name = full_name.strip() or username.title()

    if not username:
        return False, "Username is required.", None
    if len(username) < 3:
        return False, "Username must be at least 3 characters.", None
    if not email or "@" not in email:
        return False, "A valid email address is required.", None
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long.", None

    pwd_hash, salt = _hash_password(password)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash, salt, full_name, role)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username, email, pwd_hash, salt, full_name, role)
        )
        conn.commit()
        user_id = cursor.lastrowid

        user_data = {
            "id": user_id,
            "username": username,
            "email": email,
            "full_name": full_name,
            "role": role,
            "created_at": datetime.now().isoformat()
        }

        # Create a default welcome conversation for the new user
        create_conversation(user_id, title="Welcome to Prescription Trend AI")
        
        return True, "Account registered successfully! Welcome aboard.", user_data

    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "users.username" in err_msg or "unique constraint failed: users.username" in err_msg:
            return False, f"Username '{username}' is already taken. Please choose another.", None
        elif "users.email" in err_msg or "unique constraint failed: users.email" in err_msg:
            return False, f"Email '{email}' is already registered. Please sign in.", None
        return False, "An account with these details already exists.", None
    except Exception as e:
        return False, f"Registration error: {str(e)}", None
    finally:
        conn.close()


def authenticate_user(username_or_email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticate user with username or email and password.
    Returns: (success: bool, message: str, user_dict: Optional[dict])
    """
    identifier = username_or_email.strip().lower()
    if not identifier or not password:
        return False, "Please enter your username/email and password.", None

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, username, email, password_hash, salt, full_name, role, created_at
            FROM users
            WHERE username = ? OR email = ?
            """,
            (identifier, identifier)
        )
        row = cursor.fetchone()
        if not row:
            return False, "User not found. Please check your credentials or create an account.", None

        if _verify_password(password, row["salt"], row["password_hash"]):
            user_data = {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "role": row["role"],
                "created_at": row["created_at"]
            }
            return True, "Authentication successful.", user_data
        else:
            return False, "Incorrect password. Please try again.", None
    except Exception as e:
        return False, f"Login error: {str(e)}", None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve user details by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, username, email, full_name, role, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ─── Conversation Management (ChatGPT-style) ─────────────────────────────────

def create_conversation(user_id: int, title: str = "New Chat") -> Dict[str, Any]:
    """Create a new conversation session for a user."""
    conv_id = str(uuid.uuid4())
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO conversations (id, user_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (conv_id, user_id, title, now, now)
    )
    conn.commit()
    conn.close()
    return {
        "id": conv_id,
        "user_id": user_id,
        "title": title,
        "created_at": now,
        "updated_at": now
    }


def get_user_conversations(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve all conversations for a specific user, sorted newest updated first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_conversation(conv_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Get conversation details by ID for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_conversation_title(conv_id: str, new_title: str):
    """Update conversation title."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (new_title.strip(), now, conv_id)
    )
    conn.commit()
    conn.close()


def touch_conversation(conv_id: str):
    """Update conversation timestamp when a new message is added."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conv_id)
    )
    conn.commit()
    conn.close()


def delete_conversation(conv_id: str, user_id: int) -> bool:
    """Delete a conversation and all its messages."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


# ─── Message Management ───────────────────────────────────────────────────────

def add_message(conv_id: str, role: str, content: str) -> Dict[str, Any]:
    """Add a message to a conversation and update conversation timestamp."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO messages (conversation_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (conv_id, role, content, now)
    )
    msg_id = cursor.lastrowid
    conn.commit()
    conn.close()

    touch_conversation(conv_id)
    return {
        "id": msg_id,
        "conversation_id": conv_id,
        "role": role,
        "content": content,
        "created_at": now
    }


def get_conversation_messages(conv_id: str) -> List[Dict[str, Any]]:
    """Retrieve all messages in a conversation in chronological order."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (conv_id,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── Memory Storage & Management ─────────────────────────────────────────────

def save_memory(
    user_id: int, 
    content: str, 
    memory_type: str = "clinical_fact", 
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Store a long-term memory fact for the user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if identical memory already exists for user to avoid duplication
    cursor.execute(
        "SELECT id FROM user_memories WHERE user_id = ? AND content = ?",
        (user_id, content.strip())
    )
    existing = cursor.fetchone()
    if existing:
        conn.close()
        return {"id": existing["id"], "user_id": user_id, "content": content, "status": "existing"}

    cursor.execute(
        """
        INSERT INTO user_memories (user_id, conversation_id, memory_type, content, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, conversation_id, memory_type, content.strip(), now)
    )
    mem_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {
        "id": mem_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "memory_type": memory_type,
        "content": content.strip(),
        "created_at": now,
        "status": "created"
    }


def get_user_memories(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve all saved memories for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, user_id, conversation_id, memory_type, content, created_at
            FROM user_memories
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_memory(memory_id: int, user_id: int) -> bool:
    """Delete a specific memory by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user_memories WHERE id = ? AND user_id = ?", (memory_id, user_id))
    rows = cursor.rowcount
    conn.commit()
    conn.close()
    return rows > 0


def clear_all_memories(user_id: int) -> int:
    """Clear all memories for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))
    rows = cursor.rowcount
    conn.commit()
    conn.close()
    return rows


# ─── Analysis Session & Dataset / Scan Chat Management ──────────────────────

def make_json_serializable(obj: Any) -> Any:
    """Recursively converts objects to JSON-serializable primitives (dicts, lists, scalars)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        import math
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "to_dict"):
        try:
            if hasattr(obj, "columns"):
                return make_json_serializable(obj.to_dict(orient="records"))
            return make_json_serializable(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return make_json_serializable(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            val = obj.item()
            if isinstance(val, float):
                import math
                return None if (math.isnan(val) or math.isinf(val)) else val
            return val
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(item) for item in obj]
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)


def _safe_json_dumps(data: Any) -> str:
    """Safely dumps any data structure into JSON string without throwing TypeError."""
    try:
        clean_data = make_json_serializable(data)
        return json.dumps(clean_data)
    except Exception:
        try:
            return json.dumps(data, default=lambda o: str(o))
        except Exception:
            return "{}"


def create_analysis_session(
    user_id: int, 
    session_type: str, 
    title: str = "New Analysis", 
    filename: str = "", 
    data_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a new analysis session (dataset_analysis or image_report)."""
    session_id = str(uuid.uuid4())
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_str = _safe_json_dumps(data_dict or {})
    cursor.execute(
        """
        INSERT INTO analysis_sessions (id, user_id, session_type, title, filename, data_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, user_id, session_type, title, filename, data_str, now, now)
    )
    conn.commit()
    conn.close()
    return {
        "id": session_id,
        "user_id": user_id,
        "session_type": session_type,
        "title": title,
        "filename": filename,
        "data_json": data_str,
        "created_at": now,
        "updated_at": now
    }


def get_user_analysis_sessions(user_id: int, session_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve analysis sessions for a user, filtered optionally by session_type, sorted newest updated first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if session_type:
            cursor.execute(
                """
                SELECT id, user_id, session_type, title, filename, data_json, created_at, updated_at
                FROM analysis_sessions
                WHERE user_id = ? AND session_type = ?
                ORDER BY updated_at DESC
                """,
                (user_id, session_type)
            )
        else:
            cursor.execute(
                """
                SELECT id, user_id, session_type, title, filename, data_json, created_at, updated_at
                FROM analysis_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,)
            )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_analysis_session(session_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Get analysis session details by ID for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, user_id, session_type, title, filename, data_json, created_at, updated_at
            FROM analysis_sessions
            WHERE id = ? AND user_id = ?
            """,
            (session_id, user_id)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_analysis_session_data(
    session_id: str, 
    title: Optional[str] = None, 
    filename: Optional[str] = None, 
    data_dict: Optional[Dict[str, Any]] = None
) -> bool:
    """Update session title, filename, and/or saved analysis data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("SELECT title, filename, data_json FROM analysis_sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False
        
    new_title = title.strip() if title is not None and title.strip() else row["title"]
    new_filename = filename if filename is not None else row["filename"]
    
    if data_dict is not None:
        try:
            curr_data = json.loads(row["data_json"] or "{}")
        except Exception:
            curr_data = {}
        curr_data.update(data_dict)
        new_data_str = _safe_json_dumps(curr_data)
    else:
        new_data_str = row["data_json"]
        
    cursor.execute(
        """
        UPDATE analysis_sessions
        SET title = ?, filename = ?, data_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (new_title, new_filename, new_data_str, now, session_id)
    )
    conn.commit()
    conn.close()
    return True


def delete_analysis_session(session_id: str, user_id: int) -> bool:
    """Safely delete an analysis session, its saved data, and its messages."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM analysis_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def add_analysis_message(session_id: str, role: str, content: str) -> Dict[str, Any]:
    """Add a chat message linked to a specific analysis session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO analysis_messages (session_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, role, content, now)
    )
    msg_id = cursor.lastrowid
    cursor.execute(
        "UPDATE analysis_sessions SET updated_at = ? WHERE id = ?",
        (now, session_id)
    )
    conn.commit()
    conn.close()
    return {
        "id": msg_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": now
    }


def get_analysis_messages(session_id: str) -> List[Dict[str, Any]]:
    """Get all chat messages for a specific analysis session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, session_id, role, content, created_at
            FROM analysis_messages
            WHERE session_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (session_id,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# Ensure database tables exist at import time
init_db()
