#  user_db.py — управление пользователями (авторизация/аутентификация)
import sqlite3
import hashlib

USERS_DB_PATH = "users.db"

ROLE_ADMIN = 0
ROLE_USER = 1

def init_user_db():
    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(username: str, password: str):
    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    stored_hash, role = result
    return hash_password(password) == stored_hash, role

def create_user(username: str, password: str, role: int):
    password_hash = hash_password(password)
    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                       (username, password_hash, role))
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("Пользователь с таким именем уже существует")
    finally:
        conn.close()

# :
# init_user_db()
# create_user("starkspecter7", "securepass", ROLE_ADMIN)
# print(check_credentials("starkspecter7", "securepass"))
