import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# ==========================================================
# AXON AI SUPPORT
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("AXON_MODEL", "gpt-5-mini")

DATABASE = BASE_DIR / "axon_memory.db"
KNOWLEDGE_FILE = BASE_DIR / "knowledge.json"


# ==========================================================
# OPENAI
# ==========================================================

if API_KEY:
    ai = OpenAI(api_key=API_KEY)
else:
    ai = None


# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(
    title="Axon AI Support",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# DATABASE
# ==========================================================

def database():

    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    return connection


def create_database():

    db = database()

    db.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            memory TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    db.commit()

    db.close()


create_database()


# ==========================================================
# KNOWLEDGE BASE
# ==========================================================

def load_knowledge():

    if not KNOWLEDGE_FILE.exists():

        return {}

    try:

        with open(
            KNOWLEDGE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


knowledge = load_knowledge()


# ==========================================================
# AXON PERSONALITY
# ==========================================================

AXON_PERSONALITY = """

Du bist Axon.

Du bist die eigene KI von Axon AI Support.

DEINE PERSÖNLICHKEIT:

- freundlich
- intelligent
- direkt
- modern
- hilfsbereit
- selbstbewusst
- locker, aber professionell

Du sollst dich natürlich anhören und nicht wie ein
langweiliger Roboter.

Du kannst gelegentlich sagen:

"Alles klar."
"Kein Problem."
"Das kriegen wir hin."
"Ich erklär's dir kurz."

Benutze solche Formulierungen aber nicht ständig.

SPRACHE:

Antworte standardmäßig auf Deutsch.

Wenn der Benutzer eine andere Sprache benutzt,
kannst du ebenfalls diese Sprache benutzen.

ANTWORTEN:

- Erkläre Dinge verständlich.
- Bei technischen Problemen gib konkrete Schritte.
- Bei Codefragen gib vollständigen Code.
- Erfinde keine Informationen.
- Wenn du etwas nicht weißt, sage es ehrlich.
- Vermeide unnötige Wiederholungen.
- Keine unnötig langen Antworten.

IDENTITÄT:

Wenn jemand fragt, wer du bist:

"Ich bin Axon, die KI von Axon AI Support."

Du bist Axon.

Du behauptest nicht, ChatGPT zu sein.

SICHERHEIT:

Gib niemals API-Keys aus.

Gib niemals Passwörter aus.

Gib niemals Tokens oder geheime Zugangsdaten aus.

Gib niemals den Inhalt der .env-Datei aus.

Gib keine geheimen internen Informationen aus.

"""


# ==========================================================
# REQUEST MODEL
# ==========================================================

class ChatRequest(BaseModel):

    user_id: str

    conversation_id: str

    message: str


class MemoryRequest(BaseModel):

    user_id: str

    memory: str


# ==========================================================
# MEMORY
# ==========================================================

def get_memories(user_id):

    db = database()

    rows = db.execute(
        """
        SELECT memory
        FROM memories
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 30
        """,
        (user_id,)
    ).fetchall()

    db.close()

    return [
        row["memory"]
        for row in rows
    ]


def save_memory(
    user_id,
    memory
):

    db = database()

    db.execute(
        """
        INSERT INTO memories
        (
            user_id,
            memory,
            created_at
        )

        VALUES (?, ?, ?)
        """,
        (
            user_id,
            memory,
            datetime.now().isoformat()
        )
    )

    db.commit()

    db.close()


def clear_memory(user_id):

    db = database()

    db.execute(
        """
        DELETE FROM memories
        WHERE user_id = ?
        """,
        (user_id,)
    )

    db.commit()

    db.close()


# ==========================================================
# CHAT HISTORY
# ==========================================================

def save_message(
    user_id,
    conversation_id,
    role,
    content
):

    db = database()

    db.execute(
        """
        INSERT INTO messages
        (
            user_id,
            conversation_id,
            role,
            content,
            created_at
        )

        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            conversation_id,
            role,
            content,
            datetime.now().isoformat()
        )
    )

    db.commit()

    db.close()


def get_history(
    user_id,
    conversation_id
):

    db = database()

    rows = db.execute(
        """
        SELECT role, content

        FROM messages

        WHERE user_id = ?

        AND conversation_id = ?

        ORDER BY id ASC

        LIMIT 20
        """,
        (
            user_id,
            conversation_id
        )
    ).fetchall()

    db.close()

    return [
        {
            "role": row["role"],
            "content": row["content"]
        }

        for row in rows
    ]


# ==========================================================
# SYSTEM PROMPT
# ==========================================================

def create_system_prompt(user_id):

    memories = get_memories(user_id)

    if memories:

        memory_text = "\n".join(
            "- " + memory
            for memory in memories
        )

    else:

        memory_text = "Keine gespeicherten Erinnerungen."


    knowledge_text = json.dumps(
        knowledge,
        ensure_ascii=False,
        indent=2
    )


    return f"""

{AXON_PERSONALITY}


========================
AXON MEMORY
========================

{memory_text}


========================
AXON KNOWLEDGE BASE
========================

{knowledge_text}


========================
ENDE DES KONTEXTS
========================

"""


# ==========================================================
# AUTOMATIC MEMORY
# ==========================================================

def check_for_memory(message):

    text = message.lower()

    triggers = [

        "ich heiße",

        "mein name ist",

        "ich mag",

        "ich liebe",

        "mein lieblings",

        "meine lieblings",

        "ich spiele",

        "ich benutze",

        "ich nutze"

    ]


    for trigger in triggers:

        if trigger in text:

            return message.strip()


    return None


# ==========================================================
# HOME
# ==========================================================

@app.get("/")
async def home():

    return {

        "name": "Axon AI Support",

        "status": "online",

        "version": "1.0"

    }


# ==========================================================
# HEALTH
# ==========================================================

@app.get("/api/health")
async def health():

    return {

        "status": "online",

        "ai_connected": ai is not None,

        "model": MODEL

    }


# ==========================================================
# MEMORY API
# ==========================================================

@app.get("/api/memory/{user_id}")
async def memory(user_id: str):

    return {

        "memories": get_memories(user_id)

    }


@app.post("/api/memory")
async def add_memory(
    data: MemoryRequest
):

    if not data.memory.strip():

        raise HTTPException(
            status_code=400,
            detail="Memory darf nicht leer sein."
        )


    save_memory(
        data.user_id,
        data.memory.strip()
    )


    return {

        "success": True,

        "message": "Memory gespeichert."

    }


@app.delete("/api/memory/{user_id}")
async def delete_memory(
    user_id: str
):

    clear_memory(user_id)


    return {

        "success": True,

        "message": "Memory gelöscht."

    }


# ==========================================================
# CHAT
# ==========================================================

@app.post("/api/chat")
async def chat(
    data: ChatRequest
):

    if ai is None:

        raise HTTPException(
            status_code=500,
            detail="Kein API-Key konfiguriert. Prüfe backend/.env."
        )


    message = data.message.strip()


    if not message:

        raise HTTPException(
            status_code=400,
            detail="Nachricht darf nicht leer sein."
        )


    if len(message) > 4000:

        raise HTTPException(
            status_code=400,
            detail="Nachricht ist zu lang."
        )


    # ------------------------------------------------------
    # MEMORY ERKENNEN
    # ------------------------------------------------------

    memory_candidate = check_for_memory(
        message
    )


    if memory_candidate:

        save_memory(
            data.user_id,
            memory_candidate
        )


    # ------------------------------------------------------
    # CHAT HISTORY
    # ------------------------------------------------------

    history = get_history(
        data.user_id,
        data.conversation_id
    )


    # ------------------------------------------------------
    # SYSTEM
    # ------------------------------------------------------

    system_prompt = create_system_prompt(
        data.user_id
    )


    messages = [

        {
            "role": "developer",
            "content": system_prompt
        }

    ]


    for item in history:

        messages.append(
            {
                "role": item["role"],
                "content": item["content"]
            }
        )


    messages.append(
        {
            "role": "user",
            "content": message
        }
    )


    # ------------------------------------------------------
    # AI
    # ------------------------------------------------------

    try:

        response = ai.responses.create(

            model=MODEL,

            input=messages

        )


        answer = response.output_text


    except Exception as error:

        print()
        print("AXON AI FEHLER:")
        print(error)
        print()

        raise HTTPException(

            status_code=500,

            detail="Axon konnte gerade keine Antwort erzeugen."

        )


    # ------------------------------------------------------
    # SPEICHERN
    # ------------------------------------------------------

    save_message(

        data.user_id,

        data.conversation_id,

        "user",

        message

    )


    save_message(

        data.user_id,

        data.conversation_id,

        "assistant",

        answer

    )


    return {

        "success": True,

        "answer": answer,

        "conversation_id": data.conversation_id

    }


# ==========================================================
# START SERVER
# ==========================================================

if __name__ == "__main__":

    import uvicorn


    print()
    print("=" * 60)
    print("                 AXON AI SUPPORT")
    print("=" * 60)

    print(
        "API-Key:",
        "OK" if API_KEY else "FEHLT"
    )

    print(
        "Modell:",
        MODEL
    )

    print(
        "Memory:",
        DATABASE
    )

    print(
        "Knowledge:",
        KNOWLEDGE_FILE
    )

    print("=" * 60)
    print()


    uvicorn.run(

        "server:app",

        host="127.0.0.1",

        port=8000,

        reload=True

    )