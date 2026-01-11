import pathway as pw
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIGURATION
# =====================================================

VECTOR_STORE_PATH = "./vector_store"
TRAIN_PATH = "./Dataset/train_with_claims_and_contradictions.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CLAIMS = 4

model = SentenceTransformer(MODEL_NAME)

EVENT_LEMMAS = [
    "arrest","imprison","ship","kill","drown","burn","poison","escape",
    "marry","trial","seize","rescue","raid","smuggle","betray","execute",
    "transport","meet","confess"
]

# =====================================================
# UDFs
# =====================================================

@pw.udf
def embed(text: str) -> list[float]:
    return model.encode(text).tolist()

@pw.udf
def cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.array(a)
    vb = np.array(b)
    if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
        return 0.0
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

@pw.udf
def bm25_like(query: str, doc: str) -> float:
    q = query.lower().split()
    d = doc.lower()
    score = 0
    for w in q:
        if len(w) > 2 and w in d:
            score += 1
    return float(score)

@pw.udf
def parse_claims(claims_json: str) -> list[dict]:
    try:
        claims = json.loads(claims_json)
        if not isinstance(claims, list):
            return []
        return claims[:MAX_CLAIMS]
    except:
        return []

@pw.udf
def explode_contradictions(contradictions_json: str, claim_id: int) -> list[str]:
    try:
        data = json.loads(contradictions_json)
        return data.get(str(claim_id), [])
    except:
        return []

@pw.udf
def char_filter(characters: list[str], row: dict) -> bool:
    for c in characters:
        if c not in row:
            return False
        if row[c] != 1:
            return False
    return True

@pw.udf
def extract_events(text: str) -> list[str]:
    t = text.lower()
    found = []
    for e in EVENT_LEMMAS:
        if e in t:
            found.append("event_" + e)
    return found

@pw.udf
def event_filter(event_cols: list[str], row: dict) -> bool:
    if not event_cols:
        return True
    for e in event_cols:
        if e not in row or row[e] != 1:
            return False
    return True

@pw.udf
def event_bonus(event_cols: list[str], row: dict) -> float:
    if not event_cols:
        return 0.0
    bonus = 0.0
    for e in event_cols:
        if e in row and row[e] == 1:
            bonus += 1.0
    return bonus

# =====================================================
# LOAD DATA
# =====================================================

vector_store = pw.io.fs.read(
    VECTOR_STORE_PATH,
    format="json",
    mode="static",
    schema=pw.schema_from_types(
        book_name=str,
        chunk_text=str,
        embedding=list[float],
    ),
)

train = pw.io.fs.read(
    TRAIN_PATH,
    format="csv",
    mode="static",
    schema=pw.schema_from_csv(TRAIN_PATH),
)

# =====================================================
# PREPROCESS BACKSTORIES
# =====================================================

train_claims = train.select(
    story_id=pw.this.id,
    book_name=pw.this.book_name,
    characters=pw.this.characters,
    claims_list=pw.apply(parse_claims, pw.this.claims),
    contradictions_json=pw.this.contradictions,
).flatten(pw.this.claims_list)

claims_expanded = train_claims.select(
    story_id=pw.this.story_id,
    book_name=pw.this.book_name,
    characters=pw.this.characters,
    contradictions_json=pw.this.contradictions_json,
    claim_id=pw.this.claims_list["claim_id"],
    claim_text=pw.this.claims_list["claim_text"],
    claim_embedding=pw.apply(embed, pw.this.claims_list["claim_text"]),
    event_cols=pw.apply(extract_events, pw.this.claims_list["claim_text"])
)

# =====================================================
# CLAIM RETRIEVAL
# =====================================================

filtered_chunks = vector_store.filter(
    pw.apply(char_filter, claims_expanded.characters, pw.this)
    & pw.apply(event_filter, claims_expanded.event_cols, pw.this)
)

claim_matches = claims_expanded.join(
    filtered_chunks,
    claims_expanded.book_name == filtered_chunks.book_name,
    how=pw.JoinMode.INNER
).select(
    story_id=pw.this.left.story_id,
    claim_id=pw.this.left.claim_id,
    claim_text=pw.this.left.claim_text,
    chunk_text=pw.this.right.chunk_text,

    semantic_score=pw.apply(cosine_sim,
        pw.this.left.claim_embedding,
        pw.this.right.embedding
    ),

    lexical_score=pw.apply(bm25_like,
        pw.this.left.claim_text,
        pw.this.right.chunk_text
    ),
    event_boost=pw.apply(event_bonus, pw.this.left.event_cols, pw.this.right)
).with_columns(
    final_score=0.75 * pw.this.lexical_score + 0.25 * pw.this.semantic_score + 0.5 * pw.this.event_boost
)

best_claim_chunks = claim_matches.groupby(
    pw.this.story_id, pw.this.claim_id
).reduce(
    pw.this.story_id,
    pw.this.claim_id,
    pw.this.claim_text,
    claim_chunks=pw.reducers.topk(
        pw.struct(chunk=pw.this.chunk_text, score=pw.this.final_score),
        key=pw.this.final_score,
        k=60
    )
)

# =====================================================
# CONTRADICTION RETRIEVAL
# =====================================================

contras_expanded = claims_expanded.select(
    pw.this.story_id,
    pw.this.book_name,
    pw.this.characters,
    pw.this.event_cols,
    pw.this.claim_id,
    contra_text_list=pw.apply(explode_contradictions,
        pw.this.contradictions_json,
        pw.this.claim_id
    )
).flatten(pw.this.contra_text_list)

contras_embedded = contras_expanded.select(
    pw.this.story_id,
    pw.this.book_name,
    pw.this.characters,
    pw.this.event_cols,
    pw.this.claim_id,
    contra_text=pw.this.contra_text_list,
    contra_embedding=pw.apply(embed, pw.this.contra_text_list)
)

filtered_chunks = vector_store.filter(
    pw.apply(char_filter, contras_embedded.characters, pw.this)
    & pw.apply(event_filter, contras_embedded.event_cols, pw.this)
)

contra_matches = contras_embedded.join(
    filtered_chunks,
    contras_embedded.book_name == filtered_chunks.book_name,
    how=pw.JoinMode.INNER
).select(
    story_id=pw.this.left.story_id,
    claim_id=pw.this.left.claim_id,
    contra_text=pw.this.left.contra_text,
    chunk_text=pw.this.right.chunk_text,

    semantic_score=pw.apply(cosine_sim,
        pw.this.left.contra_embedding,
        pw.this.right.embedding
    ),

    lexical_score=pw.apply(bm25_like,
        pw.this.left.contra_text,
        pw.this.right.chunk_text
    ),
    event_boost=pw.apply(event_bonus, pw.this.left.event_cols, pw.this.right)
).with_columns(
    final_score=0.75 * pw.this.lexical_score + 0.25 * pw.this.semantic_score + 0.5 * pw.this.event_boost
)

best_contra_chunks = contra_matches.groupby(
    pw.this.story_id,
    pw.this.claim_id,
    pw.this.contra_text
).reduce(
    pw.this.story_id,
    pw.this.claim_id,
    pw.this.contra_text,
    contra_chunks=pw.reducers.topk(
        pw.struct(chunk=pw.this.chunk_text, score=pw.this.final_score),
        key=pw.this.final_score,
        k=60
    )
)

# =====================================================
# FINAL AGGREGATION
# =====================================================

final_output = best_claim_chunks.join(
    best_contra_chunks,
    on=["story_id", "claim_id"],
    how=pw.JoinMode.LEFT
).groupby(
    pw.this.story_id,
    pw.this.claim_id
).reduce(
    pw.this.story_id,
    pw.this.claim_id,
    pw.this.claim_text,
    pw.this.claim_chunks,
    contradiction_evidence=pw.reducers.collect(
        pw.struct(
            contradiction=pw.this.contra_text,
            chunks=pw.this.contra_chunks
        )
    )
)

pw.debug.compute_and_print(final_output)
pw.run()
