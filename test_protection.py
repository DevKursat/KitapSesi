#!/usr/bin/env python3
"""Quick test for term protection and TTS normalization - standalone."""
import re


# ---- Inline the protection code so we don't need full module imports ----

_PROTECTED_ACRONYMS_2L = frozenset({
    "AI", "AR", "DJ", "EU", "EQ", "FM", "HD", "HR", "IP", "IQ",
    "IT", "MC", "OK", "PC", "PR", "TV", "UK", "UN", "US", "VR",
    "VP",
})

_COMMON_TITLE_WORDS = frozenset({
    "A", "About", "After", "Again", "All", "Also", "Am", "An", "And",
    "Another", "Any", "Are", "As", "At", "Back", "Be", "Because",
    "Been", "Before", "Being", "Both", "But", "By", "Can", "Come",
    "Could", "Did", "Do", "Does", "Done", "Down", "During", "Each",
    "Either", "Even", "Every", "Few", "Find", "First", "For", "From",
    "Get", "Give", "Go", "Going", "Gone", "Good", "Got", "Great",
    "Had", "Has", "Have", "Having", "He", "Her", "Here", "Hers",
    "Herself", "Him", "Himself", "His", "How", "However", "I", "If",
    "In", "Indeed", "Instead", "Into", "Is", "It", "Its", "Itself",
    "Just", "Keep", "Kind", "Know", "Last", "Leave", "Let", "Life",
    "Like", "Little", "Long", "Look", "Made", "Make", "Man", "Many",
    "May", "Me", "Might", "Mind", "More", "Most", "Much", "Must",
    "My", "Myself", "Never", "New", "Next", "No", "None", "Nor",
    "Not", "Nothing", "Now", "Of", "Off", "Often", "Oh", "Old",
    "On", "Once", "One", "Only", "Or", "Other", "Others", "Our",
    "Ours", "Ourselves", "Out", "Over", "Own", "Part", "People",
    "Perhaps", "Place", "Point", "Put", "Quite", "Rather", "Really",
    "Right", "Said", "Same", "Say", "See", "Seem", "Set", "Shall",
    "She", "Should", "Show", "Since", "So", "Some", "Something",
    "Sometimes", "Still", "Such", "Take", "Tell", "Than", "That",
    "The", "Their", "Theirs", "Them", "Themselves", "Then", "There",
    "Therefore", "These", "They", "Thing", "Things", "Think", "This",
    "Those", "Though", "Through", "Thus", "Time", "To", "Together",
    "Too", "Toward", "Turn", "Two", "Under", "Until", "Up", "Upon",
    "Us", "Use", "Used", "Using", "Very", "Want", "Was", "Way",
    "We", "Well", "Were", "What", "Whatever", "When", "Whenever",
    "Where", "Whether", "Which", "While", "Who", "Whom", "Whose",
    "Why", "Will", "With", "Within", "Without", "Woman", "Won",
    "Work", "World", "Would", "Yes", "Yet", "You", "Your", "Yours",
    "Bir", "Bu", "\u015eu", "Ve", "Ya", "Veya", "De", "Da", "\u0130le",
    "\u0130\u00e7in", "Ama", "Fakat", "Ancak", "\u00c7\u00fcnk\u00fc", "Hem", "Ne", "Nas\u0131l",
    "Neden", "Nerede", "Kim", "Hangi", "Her", "Hi\u00e7", "\u00c7ok", "Az",
    "Daha", "En", "Baz\u0131", "B\u00fct\u00fcn", "T\u00fcm", "Ayn\u0131", "Ba\u015fka", "Di\u011fer",
    "Sonra", "\u00d6nce", "\u015eimdi", "Zaten", "Bile", "Hala", "Yine",
    "Art\u0131k", "Belki", "Sadece", "Yaln\u0131z", "Ben", "Sen", "Biz",
    "Siz", "Onlar", "Benim", "Senin", "Bizim", "Sizin", "Onun",
})


def _find_protected_spans(text):
    spans = []

    def _occupied(start, end):
        return any(s < end and e > start for s, e, _ in spans)

    def _add(start, end, term):
        if not _occupied(start, end):
            spans.append((start, end, term))

    for m in re.finditer(r'\b([A-Z][A-Z0-9]{2,})\b', text):
        _add(m.start(1), m.end(1), m.group(1))

    for m in re.finditer(r'\b([A-Z]{2})\b', text):
        if m.group(1) in _PROTECTED_ACRONYMS_2L:
            _add(m.start(1), m.end(1), m.group(1))

    for m in re.finditer(r'\b([a-z]+[A-Z][a-zA-Z]*)\b', text):
        _add(m.start(1), m.end(1), m.group(1))
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b', text):
        _add(m.start(1), m.end(1), m.group(1))

    return sorted(spans, key=lambda s: s[0])


def protect_terms(text, glossary=frozenset()):
    spans = []
    for term in sorted(glossary, key=len, reverse=True):
        for match in re.finditer(
            rf"(?<!\w){re.escape(term)}(?!\w)",
            text,
        ):
            if not any(
                start < match.end() and end > match.start()
                for start, end, _ in spans
            ):
                spans.append((match.start(), match.end(), match.group(0)))
    for start, end, term in _find_protected_spans(text):
        if not any(
            existing_start < end and existing_end > start
            for existing_start, existing_end, _ in spans
        ):
            spans.append((start, end, term))
    spans.sort(key=lambda item: item[0])
    if not spans:
        return text, []
    terms = []
    idx_map = {}
    parts = []
    cursor = 0
    for start, end, term in spans:
        parts.append(text[cursor:start])
        if term not in idx_map:
            idx_map[term] = len(terms)
            terms.append(term)
        parts.append(f"[NAME{idx_map[term]}]")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), terms


def restore_terms(translated, terms):
    if not terms:
        return translated

    result = translated
    for idx, term in enumerate(terms):
        found = False
        marker_pattern = re.compile(
            rf"\[\s*(?:[^\W\d_]+\s*)+(?:{idx}\s*)+\]?",
            flags=re.IGNORECASE,
        )

        def _restore_marker(_):
            nonlocal found
            if found:
                return ""
            found = True
            return term

        result = marker_pattern.sub(_restore_marker, result)
        for pattern in (rf"\{{\s*(?:{idx}\s*)+\}}", re.escape(f"({idx})")):
            if found:
                result = re.sub(pattern, "", result)
                continue
            m = re.search(pattern, result)
            if m:
                result = result[:m.start()] + term + result[m.end():]
                found = True
        if not found and len(terms) == 1:
            generic_marker = re.compile(
                r"\[\s*N(?:AM|OM)[^\]\d]{0,12}\]",
                flags=re.IGNORECASE,
            )

            def _restore_generic(_):
                nonlocal found
                if found:
                    return ""
                found = True
                return term

            result = generic_marker.sub(_restore_generic, result)
        if not found and term not in result:
            result = result.rstrip() + " " + term
        result = re.sub(
            rf"(?<!\w)({re.escape(term)})(?:\s*[,;:\-]?\s*\1)+(?!\w)",
            term,
            result,
            flags=re.IGNORECASE,
        )
    return re.sub(r"\s{2,}", " ", result).strip()


def normalize_tts_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return text
    if text[-1] not in '.!?\u2026':
        text += '.'
    text = re.sub(r'([.!?,;:])([A-Za-z\u00c0-\u00ff])', r'\1 \2', text)
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'([!?]){2,}', r'\1', text)
    return text


# ---- TESTS ----

print("=" * 60)

# Test 1: CEO
text1 = "The CEO announced a new strategy"
safe, terms = protect_terms(text1)
print(f"Test 1: '{text1}'")
print(f"  Protected: '{safe}' | Terms: {terms}")
assert "CEO" in terms, f"CEO not detected! Got: {terms}"
assert "[NAME0]" in safe, "Placeholder not inserted!"
print("  PASS")

# Test 2: Proper noun mid-sentence
text2 = "I read about Raskolnikov in the book"
safe, terms = protect_terms(text2, frozenset({"Raskolnikov"}))
print(f"\nTest 2: '{text2}'")
print(f"  Protected: '{safe}' | Terms: {terms}")
assert "Raskolnikov" in terms, f"Raskolnikov not detected! Got: {terms}"
print("  PASS")

# Test 3: Multiple terms
text3 = "CEO Raskolnikov joined NATO and used iPhone"
safe, terms = protect_terms(
    text3,
    frozenset({"Raskolnikov"}),
)
print(f"\nTest 3: '{text3}'")
print(f"  Protected: '{safe}' | Terms: {terms}")
assert "CEO" in terms, f"CEO missing! Got: {terms}"
assert "NATO" in terms, f"NATO missing! Got: {terms}"
assert "iPhone" in terms, f"iPhone missing! Got: {terms}"
print("  PASS")

# Test 4: Restore
translated = "Le [NAME0] a annonce rejoindre [NAME2] avec son [NAME3]"
restored = restore_terms(translated, terms)
print(f"\nTest 4 restore: '{translated}'")
print(f"  Restored: '{restored}'")
assert "CEO" in restored, "CEO not restored!"
assert "NATO" in restored, "NATO not restored!"
assert "iPhone" in restored, "iPhone not restored!"
print("  PASS")

# Test 5: Common words NOT protected
text5 = "The quick brown fox said something"
safe, terms = protect_terms(text5)
print(f"\nTest 5: '{text5}'")
print(f"  Protected: '{safe}' | Terms: {terms}")
assert len(terms) == 0, f"False positive! Got: {terms}"
print("  PASS")

# Test 6: TTS normalize
result = normalize_tts_text("hello world")
print(f"\nTest 6: normalize('hello world') = '{result}'")
assert result.endswith('.'), "Missing terminal punctuation!"
result2 = normalize_tts_text("hello world.")
assert result2 == "hello world.", f"Double punctuation! Got: '{result2}'"
print("  PASS")

# Test 7: 2-letter acronyms
text7 = "The AI revolution and TV shows"
safe, terms = protect_terms(text7)
print(f"\nTest 7: '{text7}'")
print(f"  Protected: '{safe}' | Terms: {terms}")
assert "AI" in terms, f"AI not detected! Got: {terms}"
assert "TV" in terms, f"TV not detected! Got: {terms}"
print("  PASS")

# Test 8: CamelCase
text8 = "Download from YouTube or WhatsApp"
safe, terms = protect_terms(text8)
print(f"\nTest 8: '{text8}'")
print(f"  Protected: '{safe}' | Terms: {terms}")
assert "YouTube" in terms, f"YouTube not detected! Got: {terms}"
print("  PASS")

# Test 9: Sentence-start proper noun
text9 = "Raskolnikov walked down the street"
safe, terms = protect_terms(text9, frozenset({"Raskolnikov"}))
print(f"\nTest 9: '{text9}'")
print(f"  Protected: '{safe}' | Terms: {terms}")
assert "Raskolnikov" in terms, f"Raskolnikov not detected at start! Got: {terms}"
print("  PASS")

# Test 10: Fallback restore (mangled placeholder)
text10 = "Le CEO a rejoint (1)"
restored = restore_terms(text10, ["CEO", "NATO"])
print(f"\nTest 10 fallback: '{text10}'")
print(f"  Restored: '{restored}'")
assert "NATO" in restored, f"NATO not restored via fallback! Got: '{restored}'"
print("  PASS")

# Test 11: NLLB may duplicate or damage brackets around the same marker.
duplicated = "The sounding [NAME0 [ NAME0]."
restored = restore_terms(duplicated, ["Akın Altan"])
print(f"\nTest 11 duplicated marker: '{restored}'")
assert restored.count("Akın Altan") == 1, f"Name duplicated: '{restored}'"
assert "NAME0" not in restored, f"Marker leaked: '{restored}'"
print("  PASS")

# Test 12: translated/missing-close marker variants are restored.
variant = "[NAME0 Lev Nikolayevic Tolstoy and [NOME1]."
restored = restore_terms(variant, ["Lev Nikolayevic Tolstoy", "İlyas"])
print(f"\nTest 12 marker variants: '{restored}'")
assert "NAME0" not in restored and "NOME1" not in restored
assert restored.count("Lev Nikolayevic Tolstoy") == 1
assert restored.count("İlyas") == 1
print("  PASS")

# Test 13: NLLB can drop the marker index when a segment has one name.
no_index = "[NAME] And [Name] of [NAMA] [NAM] had many sheep."
restored = restore_terms(no_index, ["İlyas"])
print(f"\nTest 13 no-index marker: '{restored}'")
assert restored.count("İlyas") == 1
assert not re.search(r"\[\s*N(?:AM|OM)", restored, re.IGNORECASE)
print("  PASS")

# Test 14: marker leak detector catches malformed internal artifacts.
def has_protection_artifact(text):
    return bool(re.search(
        r"\[\s*N(?:AM|OM)|\{\s*\d+\s*\}|\[\s*\d+\s*\]",
        text,
        flags=re.IGNORECASE,
    ))

assert has_protection_artifact("[NAME] leaked")
assert has_protection_artifact("{0} leaked")
assert not has_protection_artifact("İlyas kitabı kapattı.")
print("\nTest 14 marker leak gate: PASS")

# Test 15: Ordinary Turkish labels must be translated, not protected.
text11 = "Seslendiren Akın ALTAN"
safe, terms = protect_terms(text11, frozenset({"Akın ALTAN"}))
print(f"\nTest 15 labels: '{text11}'")
print(f"  Protected: '{safe}' | Terms: {terms}")
assert "Seslendiren" not in terms, f"Label was protected! Got: {terms}"
assert "Akın ALTAN" in terms, f"Person name missing! Got: {terms}"
print("  PASS")

# Test 16: Repetition is not evidence that an ordinary word is a name.
from audiobook_pipeline import build_proper_name_glossary
from audiobook_pipeline import (
    TranscriptSegment,
    has_repetition_loop,
    translation_quality_report,
)

repeated_text = [
    "Mutluluk insanın içinde bulunur.",
    "Mutluluk servette değildir.",
    "Mutluluk tekrar bulundu.",
]
glossary = build_proper_name_glossary(repeated_text)
print(f"\nTest 16 repeated ordinary word: {sorted(glossary)}")
assert "Mutluluk" not in glossary, f"Repeated word was treated as a name: {glossary}"
print("  PASS")

# Test 17: A narrator name introduced by a metadata label is protected, while
# the label itself remains translatable.
glossary = build_proper_name_glossary(["Seslendiren Akın ALTAN"])
print(f"\nTest 17 metadata person: {sorted(glossary)}")
assert glossary == frozenset({"Akın ALTAN"}), f"Unexpected metadata glossary: {glossary}"
print("  PASS")

# Test 18: Translation quality gate catches obvious model loops.
looped = "the old man waited the old man waited the old man waited"
print(f"\nTest 18 repetition gate: '{looped}'")
assert has_repetition_loop(looped)
print("  PASS")

# Test 19: Translation quality gate blocks untranslated Turkish metadata.
source = [TranscriptSegment(0, 1, "Seslendiren Akın ALTAN")]
translated = [TranscriptSegment(0, 1, "Seslendiren Akın ALTAN")]
report = translation_quality_report(
    source,
    translated,
    "en",
    frozenset({"Akın ALTAN"}),
)
print(f"\nTest 19 metadata quality gate: {report['status']}")
assert report["status"] == "failed"
assert any(issue["type"] == "source_metadata_not_translated" for issue in report["issues"])
print("  PASS")

# Test 20: Translation quality gate passes a clean metadata translation.
translated = [TranscriptSegment(0, 1, "Narrated by Akın ALTAN.")]
report = translation_quality_report(
    source,
    translated,
    "en",
    frozenset({"Akın ALTAN"}),
)
print(f"\nTest 20 clean quality gate: {report['status']}")
assert report["status"] == "passed"
print("  PASS")

print("\n" + "=" * 60)
print("ALL 20 TESTS PASSED")
