# =====================================================
# Pathway Pipeline: Chunking + Embedding + Characters + Events
# =====================================================

import pathway as pw
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------

BOOKS_DIR = "./Dataset/Books/*.txt"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

embedding_model = SentenceTransformer(MODEL_NAME)

# -----------------------------------------------------
# FIXED CHARACTER UNIVERSE (from you)
# -----------------------------------------------------

CHARACTERS = [
    "Thalcave",
    "Faria",
    "Kai-Koumou",
    "Noirtier",
    "Tom Ayrton",
    "Ben Joyce",
    "Jacques Paganel"
]

CHAR_COLUMNS = {
    "Thalcave": "Thalcave",
    "Faria": "Faria",
    "Kai-Koumou": "Kai_Koumou",
    "Noirtier": "Noirtier",
    "Tom Ayrton": "Tom_Ayrton",
    "Ben Joyce": "Ben_Joyce",
    "Jacques Paganel": "Jacques_Paganel"
}

# -----------------------------------------------------
# EVENT LEXICON (lemmas → surface forms)
# -----------------------------------------------------

EVENT_LEXICON = {
    "arrest": ["arrest", "arrested", "re-arrested", "detained"],
    "imprison": ["imprison", "imprisoned", "prison", "cell", "dungeon"],
    "ship": ["ship", "shipped", "transported", "sent to", "aboard"],
    "kill": ["kill", "killed", "murder", "murdered", "slain", "cut his throat"],
    "drown": ["drown", "drowned", "drowning"],
    "burn": ["burn", "burned", "burnt", "set fire"],
    "poison": ["poison", "poisoned"],
    "escape": ["escape", "escaped", "fled", "slipped away"],
    "marry": ["marry", "married", "wed", "betrothal"],
    "trial": ["trial", "court", "prosecuted"],
    "seize": ["seize", "seized", "confiscated"],
    "rescue": ["rescue", "rescued", "saved"],
    "raid": ["raid", "raided"],
    "smuggle": ["smuggle", "smuggled"],
    "betray": ["betray", "betrayed", "treason"],
    "execute": ["execute", "executed", "guillotined"],
    "transport": ["transported", "deported", "exiled"],
    "meet": ["met", "meeting"],
    "confess": ["confess", "confessed"]
}

# -----------------------------------------------------
# Utilities
# -----------------------------------------------------

def chunk_text(text):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_text(text):
    return embedding_model.encode(text).tolist()

# -----------------------------------------------------
# Character presence
# -----------------------------------------------------

def character_presence(chunk):
    text = chunk.lower()
    out = {}
    for name, col in CHAR_COLUMNS.items():
        found = 1 if name.lower() in text else 0
        out[col] = found
    return out

# -----------------------------------------------------
# Event presence
# -----------------------------------------------------

def event_presence(chunk):
    text = chunk.lower()
    out = {}
    for lemma, forms in EVENT_LEXICON.items():
        found = 0
        for f in forms:
            if f in text:
                found = 1
                break
        out["event_" + lemma] = found
    return out

# -----------------------------------------------------
# Load novels
# -----------------------------------------------------

novels = pw.io.fs.read(
    BOOKS_DIR,
    format="plaintext_by_file",
    with_metadata=True,
    mode="static"
)

novels = novels.select(
    book_name=pw.apply(lambda p: p.split("/")[-1].replace(".txt",""),
                       pw.apply(str, pw.this._metadata["path"])),
    text=pw.this.data
)

# =========================
# Event extractor UDFs
# =========================

@pw.udf
def get_Noirtier(j) -> int:
    return int(j.get("Noirtier", 0))

@pw.udf
def get_Tom_Ayrton(j) -> int:
    return int(j.get("Tom_Ayrton", 0))

@pw.udf
def get_Ben_Joyce(j) -> int:
    return int(j.get("Ben_Joyce", 0))

@pw.udf
def get_Jacques_Paganel(j) -> int:
    return int(j.get("Jacques_Paganel", 0))

@pw.udf
def get_event_arrest(d: dict) -> int:
    return int(d.get("event_arrest", 0))

@pw.udf
def get_event_imprison(d: dict) -> int:
    return int(d.get("event_imprison", 0))

@pw.udf
def get_event_ship(d: dict) -> int:
    return int(d.get("event_ship", 0))

@pw.udf
def get_event_kill(d: dict) -> int:
    return int(d.get("event_kill", 0))

@pw.udf
def get_event_drown(d: dict) -> int:
    return int(d.get("event_drown", 0))

@pw.udf
def get_event_burn(d: dict) -> int:
    return int(d.get("event_burn", 0))

@pw.udf
def get_event_poison(d: dict) -> int:
    return int(d.get("event_poison", 0))

@pw.udf
def get_event_escape(d: dict) -> int:
    return int(d.get("event_escape", 0))

@pw.udf
def get_event_marry(d: dict) -> int:
    return int(d.get("event_marry", 0))

@pw.udf
def get_event_trial(d: dict) -> int:
    return int(d.get("event_trial", 0))

@pw.udf
def get_event_seize(d: dict) -> int:
    return int(d.get("event_seize", 0))

@pw.udf
def get_event_rescue(d: dict) -> int:
    return int(d.get("event_rescue", 0))

@pw.udf
def get_event_raid(d: dict) -> int:
    return int(d.get("event_raid", 0))

@pw.udf
def get_event_smuggle(d: dict) -> int:
    return int(d.get("event_smuggle", 0))

@pw.udf
def get_event_betray(d: dict) -> int:
    return int(d.get("event_betray", 0))

@pw.udf
def get_event_execute(d: dict) -> int:
    return int(d.get("event_execute", 0))

@pw.udf
def get_event_transport(d: dict) -> int:
    return int(d.get("event_transport", 0))

@pw.udf
def get_event_meet(d: dict) -> int:
    return int(d.get("event_meet", 0))

@pw.udf
def get_event_confess(d: dict) -> int:
    return int(d.get("event_confess", 0))

# -----------------------------------------------------
# Chunk
# -----------------------------------------------------


chunks = novels.select(
    book_name=pw.this.book_name,
    chunk_text=pw.apply(chunk_text, pw.this.text)
).flatten(pw.this.chunk_text)

chunks = chunks.with_columns(
    Thalcave         = pw.apply(get_Thalcave, pw.this.char_map),
    Faria            = pw.apply(get_Faria, pw.this.char_map),
    Kai_Koumou       = pw.apply(get_Kai_Koumou, pw.this.char_map),
    Noirtier         = pw.apply(get_Noirtier, pw.this.char_map),
    Tom_Ayrton       = pw.apply(get_Tom_Ayrton, pw.this.char_map),
    Ben_Joyce        = pw.apply(get_Ben_Joyce, pw.this.char_map),
    Jacques_Paganel  = pw.apply(get_Jacques_Paganel, pw.this.char_map),
)

chunks = chunks.with_columns(
    embedding=pw.apply(embed_text, pw.this.chunk_text),
    char_map=pw.apply(character_presence, pw.this.chunk_text),
    event_map=pw.apply(event_presence, pw.this.chunk_text),
)
chunks = chunks.select(
    pw.this.book_name,
    pw.this.chunk_text,
    pw.this.embedding,
    pw.this.Thalcave,
    pw.this.Faria,
    pw.this.Kai_Koumou,
    pw.this.Noirtier,
    pw.this.Tom_Ayrton,
    pw.this.Ben_Joyce,
    pw.this.Jacques_Paganel,

    # characters
    Thalcave        = pw.apply(get_Thalcave, pw.this.char_map),
    Faria           = pw.apply(get_Faria, pw.this.char_map),
    Kai_Koumou      = pw.apply(get_Kai_Koumou, pw.this.char_map),
    Noirtier        = pw.apply(get_Noirtier, pw.this.char_map),
    Tom_Ayrton      = pw.apply(get_Tom_Ayrton, pw.this.char_map),
    Ben_Joyce       = pw.apply(get_Ben_Joyce, pw.this.char_map),
    Jacques_Paganel = pw.apply(get_Jacques_Paganel, pw.this.char_map),

    # events
    event_arrest    = pw.apply(get_event_arrest, pw.this.event_map),
    event_imprison  = pw.apply(get_event_imprison, pw.this.event_map),
    event_ship      = pw.apply(get_event_ship, pw.this.event_map),
    event_kill      = pw.apply(get_event_kill, pw.this.event_map),
    event_escape    = pw.apply(get_event_escape, pw.this.event_map),
)

# -----------------------------------------------------
# Write vector store
# -----------------------------------------------------

pw.io.fs.write(chunks, "./vector_store", format="json")
pw.debug.compute_and_print(chunks)

chunk_counts = chunks.groupby(chunks.book_name).reduce(
    book_name=pw.this.book_name,
    num_chunks=pw.reducers.count()
)

pw.debug.compute_and_print(chunk_counts)
pw.run()