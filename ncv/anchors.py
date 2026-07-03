"""Backstory anchor extraction for classifier and constraint features."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .schemas import BackstoryExample


KNOWN_CHARACTERS = (
    "Faria",
    "Noirtier",
    "Thalcave",
    "Kai-Koumou",
    "Tom Ayrton",
    "Ayrton",
    "Ben Joyce",
    "Jacques Paganel",
    "Paganel",
    "Lord Glenarvan",
    "Lady Glenarvan",
    "Captain Grant",
    "Mary Grant",
    "Robert Grant",
    "Edmond Dantès",
    "Edmond Dantes",
    "Edmond",
    "Villefort",
    "Mercédès",
    "Mercedes",
    "Fernand",
    "Danglars",
    "Caderousse",
)

KNOWN_LOCATIONS = (
    "Château d’If",
    "Chateau d'If",
    "Chāteau d’If",
    "London",
    "Patagonia",
    "New Zealand",
    "Tasmania",
    "Madeira",
    "Toulon",
    "Lisbon",
    "Vienna",
    "Elba",
    "Waterloo",
    "South America",
    "South-American",
    "Marseille",
    "Marseilles",
)

KNOWN_INSTITUTIONS = (
    "Britannia",
    "Duncan",
    "Royal Geographical Society",
    "Geographical Society of Paris",
    "Girondins",
    "Bonapartist",
    "slave-traders",
    "slave traders",
)

KNOWN_ROLES = (
    "first mate",
    "quartermaster",
    "guide",
    "secretary",
    "abbé",
    "abbe",
    "chief",
    "captain",
    "procureur",
    "royalist",
    "Bonapartist",
)

KNOWN_OBJECTS = (
    "dossier",
    "letter",
    "map",
    "treasure",
    "Spada",
    "dagger",
    "forearm",
    "shark-tooth necklace",
    "battle-axe",
    "logbook",
    "document",
)

EVENT_VERB_GROUPS: Mapping[str, tuple[str, ...]] = {
    "custody": ("arrested", "re-arrested", "imprisoned", "confined", "captured", "detained", "shipped"),
    "escape_release": ("escaped", "released", "freed", "slipped", "fled"),
    "movement": ("met", "travelled", "traveled", "arrived", "departed", "returned", "sailed", "boarded", "hired"),
    "violence": ("killed", "slashed", "stabbed", "wounded", "poisoned", "murdered"),
    "knowledge": ("knew", "learned", "discovered", "revealed", "studied"),
    "relationship": ("married", "argued", "rescued", "protected", "betrayed"),
}

YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
QUOTE_RE = re.compile(r"[\"“”‘’']([^\"“”‘’']{2,80})[\"“”‘’']")
CAPITALIZED_PHRASE_RE = re.compile(r"\b[A-Z][a-z]+(?:[-\s][A-Z][a-z]+){0,4}\b")


@dataclass(frozen=True, slots=True)
class AnchorSet:
    target_character: str
    secondary_characters: tuple[str, ...] = field(default_factory=tuple)
    locations: tuple[str, ...] = field(default_factory=tuple)
    dates: tuple[str, ...] = field(default_factory=tuple)
    institutions: tuple[str, ...] = field(default_factory=tuple)
    roles: tuple[str, ...] = field(default_factory=tuple)
    event_verbs: tuple[str, ...] = field(default_factory=tuple)
    event_types: tuple[str, ...] = field(default_factory=tuple)
    named_objects: tuple[str, ...] = field(default_factory=tuple)
    quoted_phrases: tuple[str, ...] = field(default_factory=tuple)
    capitalized_phrases: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_character": self.target_character,
            "secondary_characters": list(self.secondary_characters),
            "locations": list(self.locations),
            "dates": list(self.dates),
            "institutions": list(self.institutions),
            "roles": list(self.roles),
            "event_verbs": list(self.event_verbs),
            "event_types": list(self.event_types),
            "named_objects": list(self.named_objects),
            "quoted_phrases": list(self.quoted_phrases),
            "capitalized_phrases": list(self.capitalized_phrases),
        }


def extract_anchors(example: BackstoryExample | Mapping[str, object] | str, *, target_character: str | None = None) -> AnchorSet:
    """Extract high-signal names, dates, places, roles, objects, and events."""

    if isinstance(example, BackstoryExample):
        target = example.character
        text = " ".join(part for part in (example.book_name, example.character, example.caption or "", example.backstory) if part)
    elif isinstance(example, str):
        target = target_character or ""
        text = example
    else:
        target = str(example.get("character") or example.get("char") or target_character or "")
        text = " ".join(
            str(example.get(key) or "")
            for key in ("book_name", "character", "char", "caption", "content", "backstory")
        )

    target = _canonical_character(target)
    characters = tuple(dict.fromkeys(_canonical_character(item) for item in _find_terms(text, KNOWN_CHARACTERS)))
    secondary = tuple(item for item in characters if item != target)
    event_verbs, event_types = _event_matches(text)

    return AnchorSet(
        target_character=target,
        secondary_characters=secondary,
        locations=_find_terms(text, KNOWN_LOCATIONS),
        dates=tuple(sorted(set(YEAR_RE.findall(text)))),
        institutions=_find_terms(text, KNOWN_INSTITUTIONS),
        roles=_find_terms(text, KNOWN_ROLES),
        event_verbs=event_verbs,
        event_types=event_types,
        named_objects=_find_terms(text, KNOWN_OBJECTS),
        quoted_phrases=tuple(dict.fromkeys(match.strip() for match in QUOTE_RE.findall(text) if match.strip())),
        capitalized_phrases=_capitalized_phrases(text),
    )


def strong_anchor_count(anchors: AnchorSet) -> int:
    return (
        len(anchors.secondary_characters)
        + len(anchors.locations)
        + len(anchors.dates)
        + len(anchors.institutions)
        + len(anchors.roles)
        + len(anchors.named_objects)
    )


def _find_terms(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    found: list[str] = []
    for term in terms:
        if _contains_term(text, term):
            found.append(term)
    return tuple(dict.fromkeys(found))


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"[\s-]+") + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _event_matches(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lowered = text.casefold()
    verbs: list[str] = []
    types: list[str] = []
    for event_type, terms in EVENT_VERB_GROUPS.items():
        hit = False
        for term in terms:
            if re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", lowered):
                verbs.append(term)
                hit = True
        if hit:
            types.append(event_type)
    return tuple(dict.fromkeys(verbs)), tuple(dict.fromkeys(types))


def _capitalized_phrases(text: str) -> tuple[str, ...]:
    phrases = []
    for phrase in CAPITALIZED_PHRASE_RE.findall(text):
        if len(phrase) > 2 and phrase.casefold() not in {"the", "and"}:
            phrases.append(phrase)
    return tuple(dict.fromkeys(phrases))


def _canonical_character(value: str) -> str:
    text = " ".join(str(value).replace("/", " ").split())
    aliases = {
        "Ayrton": "Tom Ayrton",
        "Tom Ayrton Ben Joyce": "Tom Ayrton/Ben Joyce",
        "Ben Joyce": "Tom Ayrton/Ben Joyce",
        "Paganel": "Jacques Paganel",
        "Edmond": "Edmond Dantès",
        "Edmond Dantes": "Edmond Dantès",
        "Mercedes": "Mercédès",
    }
    return aliases.get(text, text)
