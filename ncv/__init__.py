"""Narrative Consistency Verification package."""

from .config import DEFAULT_CONFIG, NCVConfig
from .data import load_test_examples, load_train_examples
from .llm import get_llm_client
from .schemas import (
    BackstoryExample,
    BackstoryVerdict,
    Claim,
    ClaimPlan,
    ClaimVerdict,
    ContradictionFamily,
    EvidenceChunk,
    EventFrame,
    RetrievedChunk,
    RetrievalQuery,
)

__all__ = [
    "DEFAULT_CONFIG",
    "NCVConfig",
    "load_test_examples",
    "load_train_examples",
    "get_llm_client",
    "BackstoryExample",
    "BackstoryVerdict",
    "Claim",
    "ClaimPlan",
    "ClaimVerdict",
    "ContradictionFamily",
    "EvidenceChunk",
    "EventFrame",
    "RetrievedChunk",
    "RetrievalQuery",
]

__version__ = "0.1.0"
