"""
Offline lexicon + regex NER and a topical gate for LuV faithfulness checking.

Tags domain entities in claims and notes with no model download:

    PERSON   trainee / staff names
    AREA     ICF competence area (schulische Basiskompetenzen, ...)
    SKILL    skill / trait / topic term, matched on morphology-robust roots
             (Förderbedarf / Förderung -> root "förder")
    DATE, MEASURE, NEG

A note can only contradict a claim if it is about the same topic. Retrieval is
scoped per student, so PERSON is shared by almost every pair and is useless for
gating; the gate uses SKILL/AREA entities only. share_topic() drives the
contradiction gate in faithcheck.py.
"""
import re

# Morphology-robust topic/skill roots (lowercase substrings; umlaut + ae/oe/ue
# variants both listed so matching is robust to spelling).
TOPIC_ROOTS = [
    "förder", "foerder", "konzentr", "selbstständ", "selbständ", "selbstaend",
    "selbsteinschätz", "selbsteinschaetz", "motiv", "pünktl", "puenktl",
    "zuverläss", "zuverlaess", "belast", "durchhalt", "gedächtnis", "gedaechtnis",
    "merkfäh", "merkfaeh", "ausdauer", "teamf", "teamarb", "kommunik", "sorgfalt",
    "arbeitstempo", "arbeitsqual", "arbeitsverhalten", "feinmotor", "lernlück",
    "lernluec", "fehlzeit", "anwesenheit", "stabil", "instabil", "routine",
    "qualifizierungsbaust", "wiederhol", "anweis", "fortschritt", "rückschritt",
    "ruecksch", "stärke", "staerke", "schwäch", "schwaech", "eigenständ",
    "eigenstaend", "frustration", "sozial", "emotional", "kompetenz", "leistung",
    "unterstütz", "unterstuetz", "schulisch", "schul", "abschluss", "prüfung",
    "pruefung", "zeugnis", "verhalten", "feedback", "rückmeld", "rueckmeld",
    "pause", "impuls", "ausgeglich", "stimmung", "eignung", "ausbildungsreif",
    "psychisch", "nervös", "nervoes", "unruh", "aggress", "ordnung", "hygiene",
    "gesundheit", "medikament", "therapie", "praktik", "werkstatt", "ausbildung",
    "orientier", "beruf", "rechn", "lesen", "schreib", "mathe", "deutsch",
    "konflikt", "kritik", "auffass", "umgang", "respekt", "hilfsbereit",
    "freundlich", "genauigkeit", "pünktlichkeit", "selbstvertrauen", "flexib",
    "verantwort", "motorik", "wahrnehmung", "merk", "planung", "struktur",
]

AREA_KEYS = [
    "personale kompetenz", "methodische kompetenz", "fachliche basiskompetenz",
    "fachliche kompetenz", "sozial-kommunikative", "schulische basiskompetenz",
    "berufliche orientierung", "umweltfaktor", "psychische stabilität",
    "emotionale funktion", "risikofaktor", "qualifizierungs", "ausbildungsbaustein",
]

DATE_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
MEASURE_RE = re.compile(r"\b\d+\s*(?:%|prozent|punkte?)\b|\b\d+\s*bis\s*\d+\b", re.IGNORECASE)
NEG_RE = re.compile(r"\b(?:nicht|kein\w*|nie|niemals|ohne)\b|\bes trifft nicht zu\b", re.IGNORECASE)
PERSON_RE = re.compile(r"\b(?:Herr|Frau)\s+[A-ZÄÖÜ][a-zäöüß]+")


def _topic_keys(text):
    t = (text or "").lower()
    keys = {r for r in TOPIC_ROOTS if r in t}
    keys |= {a for a in AREA_KEYS if a in t}
    return keys


def share_topic(claim, note):
    """Do the claim and note share >=1 SKILL/AREA topic entity?
    If the claim has no detectable topic entity, gating isn't possible, so return
    True (don't block) rather than over-fire on claims the lexicon misses."""
    ck = _topic_keys(claim)
    if not ck:
        return True
    return bool(ck & _topic_keys(note))


def tag_entities(text):
    """Full entity tagging for reporting/illustration."""
    t = text or ""
    return {
        "PERSON": PERSON_RE.findall(t),
        "AREA": sorted(a for a in AREA_KEYS if a in t.lower()),
        "SKILL": sorted(r for r in TOPIC_ROOTS if r in t.lower()),
        "DATE": DATE_RE.findall(t),
        "MEASURE": MEASURE_RE.findall(t),
        "NEG": NEG_RE.findall(t),
    }
