import os
import json
import openai
import pathway as pw

# -----------------------------------------------------
# OpenAI setup
# -----------------------------------------------------

openai.api_key = os.environ.get("OPENAI_API_KEY")

# -----------------------------------------------------
# System Prompt
# -----------------------------------------------------

SYSTEM_PROMPT = """
You are a textual consistency verifier.

You are given:
• A backstory that happened before the novel
• Passages from the novel retrieved by a search system

Your task is to decide whether any passage contradicts the backstory.

You must only use the provided passages.
Do not assume or invent any events, timelines, or facts that are not in the text.

Rules:
1. Treat the backstory as a hypothesis.
2. Treat the novel passages as evidence.
3. A contradiction exists if any passage implies that the backstory cannot be true.
4. If the passages are silent about the backstory, that is not a contradiction.
5. If a passage clearly conflicts with the backstory, that is a contradiction.

Output ONLY valid JSON with this schema:

{
  "verdict": "CONSISTENT or CONTRADICTED",
  "contradictions": [
    {
      "passage": "...",
      "reason": "..."
    }
  ]
}
"""

# -----------------------------------------------------
# JSON-safe OpenAI call
# -----------------------------------------------------

def call_llm(backstory: str, claim_chunk: str, contra_evidence: list[dict]) -> dict:
    evidence = []

    if claim_chunk:
        evidence.append(claim_chunk)

    for c in contra_evidence or []:
        if isinstance(c, dict) and "evidence" in c and c["evidence"]:
            evidence.append(c["evidence"])

    prompt = f"""BACKSTORY:
{backstory}

EVIDENCE:
{chr(10).join(evidence)}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )

    raw = response["choices"][0]["message"]["content"].strip()

    # Enforce strict JSON
    return json.loads(raw)

# -----------------------------------------------------
# Pathway UDF wrapper
# -----------------------------------------------------

@pw.udf
def llm_verdict(backstory: str, claim_chunk: str, contra_evidence: list[dict]) -> dict:
    try:
        return call_llm(backstory, claim_chunk, contra_evidence)
    except Exception as e:
        return {
            "verdict": "ERROR",
            "contradictions": [
                {"passage": "", "reason": str(e)}
            ]
        }
