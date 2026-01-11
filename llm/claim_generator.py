"""
Training Data Claims Generator (ChatGPT, Flexible Count)
-------------------------------------------------------
- Uses ChatGPT (OpenAI API)
- Extracts ALL factual claims (min 3)
- Outputs strict JSON
"""

import csv
import json
from openai import OpenAI


# =====================================================
# CONFIG
# =====================================================

INPUT_CSV = "./Dataset/train.csv"
OUTPUT_CSV = "./Dataset/train_with_claims.csv"
MIN_CLAIMS = 3

MODEL_NAME = "gpt-4o-mini"

# =====================================================
# PROMPT
# =====================================================

SYSTEM_PROMPT = """
You are a literary fact extraction system.
Extract only factual, verifiable claims strictly grounded in text.
Output valid JSON only.
"""

def build_prompt(character: str, book: str, backstory: str) -> str:
    return f"""
Character: {character}
Book: {book}

Backstory:
{backstory}

Task:
Extract ALL factual, testable claims about the character.

Rules:
- Extract as many claims as clearly supported
- Minimum required claims: {MIN_CLAIMS}
- Each claim must be ONE declarative sentence
- Claims must be verifiable in the novel text
- Do NOT speculate or infer beyond text
- Use factual past or present tense
- Include keywords useful for semantic retrieval
- claim_id must be sequential starting from 1
- Output ONLY valid JSON, nothing else

Required JSON format:
[
  {{
    "claim_id": 1,
    "claim_text": "Character did X...",
    "keywords": ["keyword1", "keyword2"]
  }}
]
"""

# =====================================================
# CLAIM EXTRACTION
# =====================================================

def extract_claims(character: str, book: str, backstory: str) -> list:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(character, book, backstory)}
        ],
        temperature=0.2,
        max_tokens=900
    )

    raw = response.choices[0].message.content.strip()
    claims = json.loads(raw)

    # HARD VALIDATION
    assert isinstance(claims, list)
    assert len(claims) >= MIN_CLAIMS

    for i, c in enumerate(claims, start=1):
        assert c["claim_id"] == i
        assert isinstance(c["claim_text"], str)
        assert len(c["claim_text"]) > 10
        assert isinstance(c["keywords"], list)

    return claims

# =====================================================
# CSV PIPELINE
# =====================================================

def process_csv():
    with open(INPUT_CSV, newline="", encoding="utf-8") as infile, \
         open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)

        fieldnames = [f for f in reader.fieldnames if f != "caption"]
        fieldnames.append("claims")

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            character = row["char"]
            book = row["book_name"]
            backstory = row["content"]

            print(f"🔄 Processing: {character} | {book}")

            try:
                claims = extract_claims(character, book, backstory)
                row["claims"] = json.dumps(claims, ensure_ascii=False)

            except Exception as e:
                print(f"⚠️ Failed for {character}: {e}")
                row["claims"] = "[]"

            row.pop("caption", None)
            writer.writerow(row)

    print(f"\n✅ Done. Output saved to {OUTPUT_CSV}")

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    process_csv()