# =====================================================
# Pathway Pipeline: Chunking + Embedding + Characters + Events
# =====================================================

import pathway as pw
from sentence_transformers import SentenceTransformer
import re

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

# -----------------------------------------------------
# Chunk
# -----------------------------------------------------

chunks = novels.select(
    book_name=pw.this.book_name,
    chunk_text=pw.apply(chunk_text, pw.this.text)
).flatten(pw.this.chunk_text)

chunks = chunks.with_columns(
    embedding=pw.apply(embed_text, pw.this.chunk_text),
    char_map=pw.apply(character_presence, pw.this.chunk_text),
    event_map=pw.apply(event_presence, pw.this.chunk_text)
)

def explode_maps(row):
    merged = {}
    merged.update(row["char_map"])
    merged.update(row["event_map"])
    return merged

chunks = chunks.select(
    book_name=pw.this.book_name,
    chunk_text=pw.this.chunk_text,
    embedding=pw.this.embedding,
    **pw.apply(explode_maps, pw.this)
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
