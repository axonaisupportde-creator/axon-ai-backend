import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "axon_memory.db"
KNOWLEDGE_FILE = BASE_DIR / "knowledge.json"
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("AXON_MODEL", "gemini-flash-latest")
client = genai.Client(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="Axon AI Support", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

AXON_PERSONALITY = """
Du bist Axon, die KI von Axon AI Support.
Antworte standardmäßig auf Deutsch und passe dich der Sprache des Nutzers an.
Sei freundlich, intelligent, direkt, modern und hilfreich. Erkläre verständlich,
gib bei technischen Fragen konkrete Schritte und erfinde keine Fakten.
Du bist Axon und behauptest nicht, ChatGPT zu sein. Gib niemals API-Keys,
Passwörter, Tokens oder geheime Konfigurationswerte aus.
"""

class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str

class MemoryRequest(BaseModel):
    user_id: str
    memory: str


def database() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def create_database() -> None:
    with database() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            memory TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")


def load_knowledge() -> dict[str, Any]:
    if not KNOWLEDGE_FILE.exists():
        return {}
    try:
        value = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"items": value}
    except Exception:
        return {}


def get_memories(user_id: str) -> list[str]:
    with database() as db:
        rows = db.execute(
            "SELECT memory FROM memories WHERE user_id = ? ORDER BY id DESC LIMIT 30",
            (user_id,),
        ).fetchall()
    return [row["memory"] for row in rows]


def save_memory(user_id: str, memory: str) -> None:
    with database() as db:
        db.execute(
            "INSERT INTO memories (user_id, memory, created_at) VALUES (?, ?, ?)",
            (user_id, memory, datetime.utcnow().isoformat()),
        )


def clear_memory(user_id: str) -> None:
    with database() as db:
        db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))


def save_message(user_id: str, conversation_id: str, role: str, content: str) -> None:
    with database() as db:
        db.execute(
            "INSERT INTO messages (user_id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, conversation_id, role, content, datetime.utcnow().isoformat()),
        )


def get_history(user_id: str, conversation_id: str) -> list[dict[str, str]]:
    with database() as db:
        rows = db.execute(
            "SELECT role, content FROM messages WHERE user_id = ? AND conversation_id = ? ORDER BY id ASC LIMIT 20",
            (user_id, conversation_id),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def create_system_prompt(user_id: str) -> str:
    memories = get_memories(user_id)
    knowledge = load_knowledge()
    memory_text = "\n".join(f"- {item}" for item in memories) if memories else "Keine gespeicherten Erinnerungen."
    knowledge_text = json.dumps(knowledge, ensure_ascii=False, indent=2)
    return f"{AXON_PERSONALITY}\n\nAXON MEMORY:\n{memory_text}\n\nAXON KNOWLEDGE BASE:\n{knowledge_text}"


def check_for_memory(message: str) -> str | None:
    text = message.lower()
    triggers = ["ich heiße", "mein name ist", "ich mag", "ich liebe", "mein lieblings", "meine lieblings", "ich spiele", "ich benutze", "ich nutze"]
    return message.strip() if any(trigger in text for trigger in triggers) else None


create_database()

@app.get("/")
def home():
    return {"name": "Axon AI Support", "status": "online", "version": "1.1"}

@app.get("/api/health")
def health():
    return {"status": "online", "ai_connected": client is not None, "model": MODEL}

@app.get("/api/memory/{user_id}")
def memory(user_id: str):
    return {"memories": get_memories(user_id)}

@app.post("/api/memory")
def add_memory(data: MemoryRequest):
    if not data.memory.strip():
        raise HTTPException(status_code=400, detail="Memory darf nicht leer sein.")
    save_memory(data.user_id, data.memory.strip())
    return {"success": True, "message": "Memory gespeichert."}

@app.delete("/api/memory/{user_id}")
def delete_memory(user_id: str):
    clear_memory(user_id)
    return {"success": True, "message": "Memory gelöscht."}

@app.post("/api/chat")
def chat(data: ChatRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="Gemini ist nicht verbunden. Prüfe GEMINI_API_KEY in Render → Environment.")
    message = data.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Nachricht darf nicht leer sein.")
    if len(message) > 4000:
        raise HTTPException(status_code=400, detail="Nachricht ist zu lang.")

    memory_candidate = check_for_memory(message)
    if memory_candidate:
        save_memory(data.user_id, memory_candidate)

    history = get_history(data.user_id, data.conversation_id)
    contents = []
    for item in history:
        role = "model" if item["role"] in ("assistant", "model") else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=item["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=create_system_prompt(data.user_id),
                temperature=0.7,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("Gemini hat keine Textantwort geliefert.")
    except Exception as error:
        print(f"AXON GEMINI ERROR: {type(error).__name__}: {error}", flush=True)
        raise HTTPException(status_code=502, detail="Axon konnte gerade keine Gemini-Antwort erzeugen. Prüfe Render Logs und API-Key/Modell.")

    save_message(data.user_id, data.conversation_id, "user", message)
    save_message(data.user_id, data.conversation_id, "assistant", answer)
    return {"success": True, "answer": answer, "conversation_id": data.conversation_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
