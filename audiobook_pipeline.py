#!/usr/bin/env python3
"""Autonomous local YouTube audiobook translation and dubbing pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import ffmpeg
import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from TTS.api import TTS
from yt_dlp import YoutubeDL


LOGGER = logging.getLogger("audiobook_pipeline")

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
TURKISH_NER_MODEL = "akdeniz27/bert-base-turkish-cased-ner"

# CLI language -> NLLB-200 FLORES code.
NLLB_LANGUAGES = {
    "ar": "arb_Arab",
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "hu": "hun_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "tr": "tur_Latn",
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
}

# Languages supported by the public XTTS-v2 multilingual checkpoint.
XTTS_LANGUAGES = {
    "ar": "ar",
    "cs": "cs",
    "de": "de",
    "en": "en",
    "es": "es",
    "fr": "fr",
    "hi": "hi",
    "hu": "hu",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "nl": "nl",
    "pl": "pl",
    "pt": "pt",
    "ru": "ru",
    "tr": "tr",
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
}


# ---------------------------------------------------------------------------
# Universal term protection for translation
# ---------------------------------------------------------------------------
# Words like CEO, Raskolnikov, iPhone, NATO are universally recognised and
# must survive translation unchanged.  The system detects them before NLLB
# sees the text, replaces them with indexed {N} placeholders, translates,
# and then restores the originals.

# Two-letter acronyms require a whitelist because many two-letter
# combinations are real words in various languages.
_PROTECTED_ACRONYMS_2L = frozenset({
    "AI", "AR", "DJ", "EU", "EQ", "FM", "HD", "HR", "IP", "IQ",
    "IT", "MC", "OK", "PC", "PR", "TV", "UK", "UN", "US", "VR",
    "VP",
})

# Words that legitimately start sentences but are NOT proper nouns.
_COMMON_TITLE_WORDS: frozenset[str] = frozenset({
    # English
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
    # Turkish
    "Bir", "Bu", "Şu", "Ve", "Ya", "Veya", "De", "Da", "İle",
    "İçin", "Ama", "Fakat", "Ancak", "Çünkü", "Hem", "Ne", "Nasıl",
    "Neden", "Nerede", "Kim", "Hangi", "Her", "Hiç", "Çok", "Az",
    "Daha", "En", "Bazı", "Bütün", "Tüm", "Aynı", "Başka", "Diğer",
    "Sonra", "Önce", "Şimdi", "Zaten", "Bile", "Hala", "Yine",
    "Artık", "Belki", "Sadece", "Yalnız", "Ben", "Sen", "Biz",
    "Siz", "Onlar", "Benim", "Senin", "Bizim", "Sizin", "Onun",
    "Seslendiren", "Seslendirme", "Okuyan", "Anlatan", "Anlatıcı",
    "Yazar", "Eser", "Bölüm", "Kısım", "Kitap", "Çeviren", "Çeviri",
    # German (nouns are capitalised; only non-nouns listed)
    "Aber", "Als", "Auch", "Auf", "Aus", "Bei", "Bin", "Bis",
    "Das", "Dem", "Den", "Der", "Des", "Die", "Dies", "Ein",
    "Eine", "Einem", "Einen", "Er", "Es", "Für", "Hat", "Ich",
    "Ihr", "Ihre", "Im", "Ist", "Ja", "Kann", "Kein", "Man",
    "Mit", "Nach", "Nicht", "Noch", "Nur", "Ob", "Oder", "Sie",
    "Sind", "Über", "Und", "Uns", "Vom", "Von", "Vor", "War",
    "Was", "Wenn", "Wer", "Wie", "Wir", "Wird", "Zu", "Zum", "Zur",
    # French
    "Alors", "Au", "Aussi", "Avec", "Car", "Ce", "Cette", "Dans",
    "Des", "Donc", "Du", "Elle", "En", "Est", "Et", "Il", "Ils",
    "Je", "La", "Le", "Les", "Leur", "Lui", "Ma", "Mais", "Mon",
    "Nos", "Notre", "Nous", "On", "Ou", "Par", "Pas", "Plus",
    "Pour", "Quand", "Que", "Qui", "Sa", "Sans", "Ses", "Si",
    "Son", "Sont", "Sur", "Tout", "Tu", "Un", "Une", "Vous",
    # Spanish
    "Al", "Con", "Del", "El", "Ella", "Ellos", "Era", "Esa", "Ese",
    "Esta", "Este", "Fue", "Hay", "Las", "Lo", "Los", "Más", "Mi",
    "Muy", "Ni", "Nos", "Para", "Pero", "Por", "Se", "Sin", "Son",
    "Su", "Sus", "Tan", "Te", "Todo", "Un", "Una", "Usted", "Yo",
})


def _find_protected_spans(
    text: str,
    glossary: frozenset[str] = frozenset(),
) -> list[tuple[int, int, str]]:
    """Return ``(start, end, term)`` for every universal term in *text*.

    Detection layers (evaluated in order, overlaps suppressed):
    1. All-caps acronyms of 3+ letters (CEO, NATO, DNA).
    2. Two-letter acronyms from a curated whitelist (AI, TV, PC).
    3. CamelCase identifiers (iPhone, YouTube, WhatsApp, McDonald).
    4. Proper names supplied by the corpus-level glossary.
    """
    spans: list[tuple[int, int, str]] = []

    def _occupied(start: int, end: int) -> bool:
        return any(s < end and e > start for s, e, _ in spans)

    def _add(start: int, end: int, term: str) -> None:
        if not _occupied(start, end):
            spans.append((start, end, term))

    # 1. Corpus-level proper names, longest first. This runs before acronym
    # detection so a full name such as "Akın ALTAN" is kept as one term.
    for term in sorted(glossary, key=len, reverse=True):
        for match in re.finditer(
            rf"(?<!\w){re.escape(term)}(?!\w)",
            text,
            flags=re.UNICODE,
        ):
            _add(match.start(), match.end(), match.group(0))

    # 2.  All-caps ≥3 letters
    for m in re.finditer(r'\b([A-Z][A-Z0-9]{2,})\b', text):
        _add(m.start(1), m.end(1), m.group(1))

    # 3.  Two-letter acronyms from whitelist
    for m in re.finditer(r'\b([A-Z]{2})\b', text):
        if m.group(1) in _PROTECTED_ACRONYMS_2L:
            _add(m.start(1), m.end(1), m.group(1))

    # 4.  CamelCase  (lowerUpper or UpperLowerUpper)
    for m in re.finditer(r'\b([a-z]+[A-Z][a-zA-Z]*)\b', text):
        _add(m.start(1), m.end(1), m.group(1))
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b', text):
        _add(m.start(1), m.end(1), m.group(1))

    return sorted(spans, key=lambda s: s[0])


def protect_terms(
    text: str,
    glossary: frozenset[str] = frozenset(),
) -> tuple[str, list[str]]:
    """Replace universal terms with stable ``[NAME<N>]`` placeholders.

    Returns the modified text and the ordered list of original terms.
    """
    spans = _find_protected_spans(text, glossary)
    if not spans:
        return text, []

    terms: list[str] = []
    idx_map: dict[str, int] = {}
    parts: list[str] = []
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


def build_proper_name_glossary(
    texts: Sequence[str],
) -> frozenset[str]:
    """Extract only names explicitly introduced by audiobook metadata.

    Ordinary sentence-initial labels such as ``Seslendiren`` and ``Bölüm`` are
    deliberately not protected. Repetition and capitalization are not treated
    as evidence of a name; narrative person names are supplied by the NER model.
    """
    word = (
        r"[A-ZÀ-ÖØ-ÞİŞĞÜÖÇ]"
        r"[a-zà-öø-ÿışğüöçâîû']{2,}"
    )
    glossary: set[str] = set()

    for text in texts:
        for match in re.finditer(
            rf"(?i:\b(?:seslendiren|okuyan|anlatan|anlatıcı|yazar|"
            rf"çeviren)\b)\s*[:\-]?\s*((?:{word}|[A-ZÇĞİÖŞÜ]{{2,}})"
            rf"(?:\s+(?:{word}|[A-ZÇĞİÖŞÜ]{{2,}})){{0,3}})",
            text,
        ):
            glossary.add(match.group(1).strip())
    return frozenset(glossary)


def restore_terms(translated: str, terms: list[str]) -> str:
    """Restore names from NLLB placeholders without corrupting real numbers."""
    if not terms:
        return translated

    result = translated
    for idx, term in enumerate(terms):
        found = False
        marker_pattern = re.compile(
            rf"\[\s*(?:[^\W\d_]+\s*)+(?:{idx}\s*)+\]?",
            flags=re.IGNORECASE,
        )

        def _restore_marker(_: re.Match) -> str:
            nonlocal found
            if found:
                return ""
            found = True
            return term

        result = marker_pattern.sub(_restore_marker, result)

        # Recover old checkpoints that used {0}/{00}. A bare number is never
        # replaced because it may be a year, quantity, or chapter number.
        for pattern in (
            rf"\{{\s*(?:{idx}\s*)+\}}",
            re.escape(f"({idx})"),
        ):
            if found:
                result = re.sub(pattern, "", result)
                continue
            match = re.search(pattern, result)
            if match:
                result = result[: match.start()] + term + result[match.end() :]
                found = True

        if not found and len(terms) == 1:
            generic_marker = re.compile(
                r"\[\s*N(?:AM|OM)[^\]\d]{0,12}\]",
                flags=re.IGNORECASE,
            )

            def _restore_generic(_: re.Match) -> str:
                nonlocal found
                if found:
                    return ""
                found = True
                return term

            result = generic_marker.sub(_restore_generic, result)

        if not found and term not in result:
            LOGGER.warning(
                "Translation dropped protected term %r; appending it.",
                term,
            )
            result = result.rstrip() + " " + term

        # Beam search can copy the marker or the restored name more than once.
        # Collapse only adjacent exact copies; unrelated repeated words remain.
        result = re.sub(
            rf"(?<!\w)({re.escape(term)})"
            rf"(?:\s*[,;:\-]?\s*\1)+(?!\w)",
            term,
            result,
            flags=re.IGNORECASE,
        )

    return re.sub(r"\s{2,}", " ", result).strip()


def has_protection_artifact(text: str) -> bool:
    """Return True when a damaged internal name marker leaked into output."""
    return bool(
        re.search(
            r"\[\s*N(?:AM|OM)|\{\s*\d+\s*\}|\[\s*\d+\s*\]",
            text,
            flags=re.IGNORECASE,
        )
    )


_SOURCE_METADATA_LABELS = {
    "seslendiren",
    "seslendirme",
    "okuyan",
    "anlatan",
    "anlatıcı",
    "yazar",
    "çeviren",
    "çeviri",
}


def has_repetition_loop(text: str) -> bool:
    """Detect obvious model loops before expensive TTS starts."""
    words = re.findall(r"[\wÀ-ÿİıŞşĞğÜüÖöÇç'-]+", text.lower())
    if len(words) < 8:
        return False

    repeated_single = 1
    for previous, current in zip(words, words[1:]):
        repeated_single = repeated_single + 1 if previous == current else 1
        if repeated_single >= 4:
            return True

    for size in range(2, 7):
        for start in range(0, len(words) - size * 3 + 1):
            phrase = words[start : start + size]
            if (
                words[start + size : start + size * 2] == phrase
                and words[start + size * 2 : start + size * 3] == phrase
            ):
                return True
    return False


def translation_quality_report(
    source_segments: Sequence["TranscriptSegment"],
    translated_segments: Sequence["TranscriptSegment"],
    target_language: str,
    glossary: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Return a release-blocking quality report for translated text."""
    issues: list[dict[str, object]] = []
    if len(source_segments) != len(translated_segments):
        issues.append(
            {
                "type": "segment_count_mismatch",
                "source_count": len(source_segments),
                "translated_count": len(translated_segments),
            }
        )

    for index, (source, translated) in enumerate(
        zip(source_segments, translated_segments)
    ):
        source_text = source.text.strip()
        text = translated.text.strip()
        lower_text = text.casefold()
        if not text:
            issues.append({"index": index, "type": "empty_translation"})
        if has_protection_artifact(text):
            issues.append(
                {"index": index, "type": "protection_marker_leaked", "text": text}
            )
        if has_repetition_loop(text):
            issues.append(
                {"index": index, "type": "repetition_loop", "text": text[:240]}
            )
        if target_language != "tr" and any(
            re.search(rf"\b{label}\b", lower_text, re.IGNORECASE)
            for label in _SOURCE_METADATA_LABELS
        ):
            issues.append(
                {
                    "index": index,
                    "type": "source_metadata_not_translated",
                    "text": text[:240],
                }
            )
        for name in glossary:
            if name in source_text and name not in text:
                issues.append(
                    {
                        "index": index,
                        "type": "protected_name_missing",
                        "name": name,
                        "text": text[:240],
                    }
                )

    return {
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": issues[:100],
    }


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class DownloadResult:
    title: str
    audio_path: Path


@dataclass(frozen=True)
class StemResult:
    vocals: Path
    bgm: Path


class PipelineError(RuntimeError):
    """Raised when a pipeline stage cannot produce a valid artifact."""


class Runtime:
    """Hardware and external binary validation for Apple Silicon execution."""

    @staticmethod
    def validate(image_path: Path) -> None:
        if not image_path.is_file():
            raise PipelineError(f"Image does not exist: {image_path}")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise PipelineError(
                "This build targets native Apple Silicon macOS (Darwin/arm64). "
                f"Detected {platform.system()}/{platform.machine()}."
            )
        if not torch.backends.mps.is_available():
            raise PipelineError(
                "PyTorch MPS is unavailable. Install a native arm64 Python and "
                "the pinned macOS PyTorch wheel."
            )
        for binary in ("ffmpeg", "ffprobe"):
            if shutil.which(binary) is None:
                raise PipelineError(
                    f"{binary} is required. Install it with: brew install ffmpeg"
                )
        LOGGER.info("PyTorch MPS device is ready")

    @staticmethod
    def mps_device() -> torch.device:
        if not torch.backends.mps.is_available():
            raise PipelineError("MPS became unavailable during execution.")
        return torch.device("mps")


class MediaDownloader:
    """Downloads the best YouTube audio and derives the output root title."""

    def download(self, url: str, workspace: Path) -> DownloadResult:
        template = str(workspace / "source.%(ext)s")
        options = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "outtmpl": template,
            "quiet": False,
            "no_warnings": False,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)

        if not info:
            raise PipelineError("yt-dlp returned no video metadata.")
        audio_path = workspace / "source.wav"
        if not audio_path.is_file():
            candidates = sorted(workspace.glob("source*.wav"))
            if not candidates:
                raise PipelineError("yt-dlp did not produce the expected WAV audio.")
            audio_path = candidates[0]
        return DownloadResult(
            title=sanitize_title(str(info.get("title") or "Audiobook")),
            audio_path=audio_path,
        )


class StemSeparator:
    """Runs fine-tuned Hybrid Transformer Demucs with a safe device policy."""

    def separate(self, audio_path: Path, workspace: Path) -> StemResult:
        # HTDemucs uses a Conv1d whose output-channel count exceeds Metal's
        # current 65,536-channel limit. Explicit MPS execution raises before
        # PyTorch can apply CPU fallback, so Demucs must run on CPU on macOS.
        # Translation and XTTS remain on MPS.
        device = "cpu" if platform.system() == "Darwin" else "mps"
        output_dir = workspace / "demucs"
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "--two-stems",
            "vocals",
            "--name",
            "htdemucs_ft",
            "--device",
            device,
            "--shifts",
            "2",
            "--overlap",
            "0.5",
            "--float32",
            "--clip-mode",
            "clamp",
            "--out",
            str(output_dir),
            str(audio_path),
        ]
        LOGGER.info("Separating vocals and BGM with Demucs on %s", device)
        try:
            subprocess.run(
                command,
                check=True,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
            )
        except subprocess.CalledProcessError as error:
            raise PipelineError(
                "Demucs stem separation failed. See worker.log for the "
                f"underlying error (device={device}, exit={error.returncode})."
            ) from error

        stem_dir = output_dir / "htdemucs_ft" / audio_path.stem
        vocals = stem_dir / "vocals.wav"
        bgm = stem_dir / "no_vocals.wav"
        if not vocals.is_file() or not bgm.is_file():
            raise PipelineError(f"Demucs stems were not found under {stem_dir}")
        return StemResult(vocals=vocals, bgm=bgm)


class Transcriber:
    """Timestamped transcription through faster-whisper/CTranslate2."""

    def __init__(self, model_name: str) -> None:
        # CTranslate2 has CUDA and CPU backends, but no Apple MPS backend.
        # int8 CPU is the fastest supported local backend on Apple Silicon.
        LOGGER.info(
            "Loading faster-whisper %s on Apple Silicon CPU (CTranslate2 int8)",
            model_name,
        )
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=max(1, (os.cpu_count() or 8) - 2),
        )

    def transcribe(
        self, vocals_path: Path, source_language: str
    ) -> list[TranscriptSegment]:
        segments, info = self.model.transcribe(
            str(vocals_path),
            language="zh" if source_language == "zh-cn" else source_language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 350,
                "speech_pad_ms": 180,
            },
            condition_on_previous_text=True,
        )
        result = [
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in segments
            if segment.text.strip()
        ]
        if not result:
            raise PipelineError("No speech was found in the vocal stem.")
        LOGGER.info(
            "Transcribed %d segments; detected language=%s probability=%.3f",
            len(result),
            info.language,
            info.language_probability,
        )
        return result


class LocalTranslator:
    """Offline NLLB-200 translation accelerated by PyTorch MPS."""

    def __init__(self, source_language: str, model_name: str) -> None:
        self.source_language = source_language
        self.source_code = nllb_code(source_language)
        self.device = Runtime.mps_device()
        LOGGER.info("Loading NLLB translation model on MPS: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            src_lang=self.source_code,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()

    def translate(
        self,
        segments: Sequence[TranscriptSegment],
        target_language: str,
        batch_size: int = 8,
        glossary: frozenset[str] = frozenset(),
    ) -> list[TranscriptSegment]:
        if target_language == self.source_language:
            return list(segments)

        target_code = nllb_code(target_language)
        translated: list[TranscriptSegment] = []
        for batch in batched(segments, batch_size):
            texts = [segment.text for segment in batch]

            # Protect universal terms (CEO, proper names, etc.)
            protected = [protect_terms(t, glossary) for t in texts]
            safe_texts = [p[0] for p in protected]
            term_maps = [p[1] for p in protected]

            inputs = self.tokenizer(
                safe_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                        target_code
                    ),
                    max_new_tokens=512,
                    num_beams=4,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                )
            output_texts = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )

            # Restore protected terms in translated output
            restored_texts = [
                restore_terms(out, terms)
                for out, terms in zip(output_texts, term_maps)
            ]
            for text in restored_texts:
                if has_protection_artifact(text):
                    raise PipelineError(
                        "Çeviri modelinin özel isim koruma etiketi çıktıya "
                        f"sızdı; hatalı TTS engellendi: {text!r}"
                    )

            translated.extend(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=text.strip(),
                )
                for segment, text in zip(batch, restored_texts, strict=True)
            )
        torch.mps.empty_cache()
        return translated


class ProperNameRecognizer:
    """Extracts Turkish named entities with a dedicated MPS NER model."""

    def __init__(self, model_name: str = TURKISH_NER_MODEL) -> None:
        self.device = Runtime.mps_device()
        LOGGER.info("Loading Turkish NER model on MPS: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
        ).to(self.device)
        self.model.eval()

    def extract_people(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
    ) -> frozenset[str]:
        names: set[str] = set()
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=512,
            )
            offsets = encoded.pop("offset_mapping")
            model_inputs = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }
            with torch.inference_mode():
                predictions = self.model(
                    **model_inputs,
                ).logits.argmax(dim=-1).cpu()

            for row, text in enumerate(batch):
                entity_start: int | None = None
                entity_end: int | None = None
                for token_index, prediction in enumerate(predictions[row]):
                    offset_start, offset_end = offsets[row, token_index].tolist()
                    label = self.model.config.id2label[int(prediction)].upper()
                    entity_type = label.split("-", 1)[-1]
                    is_person = entity_type in {"PER", "PERSON"}
                    if is_person and offset_end > offset_start:
                        starts_new_entity = (
                            entity_start is not None
                            and entity_end is not None
                            and label.startswith("B-")
                            and offset_start > entity_end
                        )
                        if entity_start is None or starts_new_entity:
                            if starts_new_entity:
                                names.add(text[entity_start:entity_end].strip())
                            entity_start = offset_start
                        entity_end = offset_end
                    elif entity_start is not None and entity_end is not None:
                        names.add(text[entity_start:entity_end].strip())
                        entity_start = entity_end = None
                if entity_start is not None and entity_end is not None:
                    names.add(text[entity_start:entity_end].strip())

        torch.mps.empty_cache()
        return frozenset(
            name
            for name in names
            if len(name) >= 3 and name not in _COMMON_TITLE_WORDS
        )


class VoiceCloner:
    """Zero-shot multilingual XTTS-v2 synthesis on CPU.

    XTTS-v2's HiFi-GAN vocoder contains Conv1d layers whose output-channel
    count exceeds the MPS backend's 65 536-channel limit.  Unlike entirely
    unimplemented ops, this is a *runtime* check inside an implemented Metal
    kernel, so PYTORCH_ENABLE_MPS_FALLBACK cannot intercept it.  The model
    must therefore run on CPU.  Apple Silicon's unified memory architecture
    keeps CPU inference reasonably fast for this workload.
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        LOGGER.info("Loading XTTS-v2 voice cloning model on CPU (MPS Conv1d channel limit)")
        self.tts = TTS(model_name=XTTS_MODEL, progress_bar=True).to(self.device)

    def synthesize_timeline(
        self,
        segments: Sequence[TranscriptSegment],
        target_language: str,
        reference_wav: Path,
        workspace: Path,
    ) -> Path:
        language = xtts_code(target_language)
        chunks_dir = workspace / f"tts_{target_language}"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        timeline = AudioSegment.silent(duration=0, frame_rate=24_000).set_channels(1)
        cursor_ms = 0
        for index, segment in enumerate(segments):
            text = normalize_tts_text(segment.text)
            if not text:
                continue
            chunk_path = chunks_dir / f"{index:06d}.wav"
            self.tts.tts_to_file(
                text=text,
                speaker_wav=str(reference_wav),
                language=language,
                file_path=str(chunk_path),
                split_sentences=True,
            )
            chunk = (
                AudioSegment.from_file(chunk_path)
                .set_frame_rate(24_000)
                .set_channels(1)
            )

            # Preserve original pauses when possible. If natural speech runs
            # longer, continue sequentially without time-stretching or overlap.
            position_ms = max(round(segment.start * 1000), cursor_ms)
            required_ms = position_ms + len(chunk)
            if len(timeline) < required_ms:
                timeline += AudioSegment.silent(
                    duration=required_ms - len(timeline),
                    frame_rate=24_000,
                ).set_channels(1)
            timeline = timeline.overlay(chunk, position=position_ms)
            cursor_ms = required_ms + 80

        if len(timeline) == 0:
            raise PipelineError(
                f"XTTS produced no audio for language {target_language}."
            )
        output_path = workspace / f"{target_language}_speech.wav"
        timeline.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        torch.mps.empty_cache()
        return output_path


class AudioMixer:
    """Loops BGM to the natural TTS duration and produces a lossless WAV mix."""

    def mix(self, speech_path: Path, bgm_path: Path, output_path: Path) -> float:
        duration = media_duration(speech_path)
        speech = ffmpeg.input(str(speech_path)).audio
        bgm = (
            ffmpeg.input(str(bgm_path), stream_loop=-1)
            .audio.filter("atrim", duration=duration)
            .filter("asetpts", "PTS-STARTPTS")
            .filter("volume", 0.35)
        )
        mixed = ffmpeg.filter(
            [speech, bgm],
            "amix",
            inputs=2,
            duration="first",
            dropout_transition=0,
            normalize=0,
        )
        (
            ffmpeg.output(
                mixed,
                str(output_path),
                format="wav",
                acodec="pcm_f32le",
                ar=48_000,
                ac=2,
                t=duration,
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return media_duration(output_path)


class VideoRenderer:
    """Renders one still image and an audio-timed bottom progress bar."""

    def render(
        self,
        image_path: Path,
        audio_path: Path,
        output_path: Path,
        duration: float,
    ) -> None:
        image = ffmpeg.input(str(image_path), loop=1, framerate=30).video
        image = image.filter(
            "scale",
            1920,
            1080,
            force_original_aspect_ratio="decrease",
        ).filter("pad", 1920, 1080, "(ow-iw)/2", "(oh-ih)/2", color="black")

        # drawbox creates the bar; overlay evaluates t as the current frame
        # timestamp and reveals it from left to right. Using t directly inside
        # drawbox is unsafe because FFmpeg also names box thickness "t".
        bar = (
            ffmpeg.input(
                "color=c=black:s=1920x10:r=30",
                f="lavfi",
            )
            .video.filter(
                "drawbox",
                x=0,
                y=0,
                w="iw",
                h="ih",
                color="red@0.95",
                t="fill",
            )
        )
        video = ffmpeg.overlay(
            image,
            bar,
            x=f"-overlay_w+main_w*min(t/{duration:.6f},1)",
            y="main_h-overlay_h",
            eval="frame",
        )
        audio = ffmpeg.input(str(audio_path)).audio
        (
            ffmpeg.output(
                video,
                audio,
                str(output_path),
                vcodec="libx264",
                preset="medium",
                crf=18,
                pix_fmt="yuv420p",
                r=30,
                acodec="aac",
                audio_bitrate="320k",
                movflags="+faststart",
                shortest=None,
                t=duration,
            )
            .overwrite_output()
            .run()
        )


class AudiobookPipeline:
    """Coordinates all pipeline stages and owns the final output contract."""

    def __init__(
        self,
        output_base: Path,
        whisper_model: str,
        translation_model: str,
    ) -> None:
        self.output_base = output_base
        self.whisper_model = whisper_model
        self.translation_model = translation_model
        self.downloader = MediaDownloader()
        self.separator = StemSeparator()
        self.mixer = AudioMixer()
        self.renderer = VideoRenderer()

    def run(
        self,
        url: str,
        image_path: Path,
        source_language: str,
        target_languages: Sequence[str],
    ) -> Path:
        Runtime.validate(image_path)
        validate_languages(source_language, target_languages)
        self.output_base.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=".audiobook-pipeline-",
            dir=self.output_base,
        ) as temp_dir:
            workspace = Path(temp_dir)
            download = self.downloader.download(url, workspace)
            output_root = unique_output_root(self.output_base, download.title)
            output_root.mkdir(parents=True)

            try:
                stems = self.separator.separate(download.audio_path, workspace)
                source_bgm = output_root / (
                    f"{source_language.upper()}_original_bgm.wav"
                )
                shutil.copy2(stems.bgm, source_bgm)

                transcript = Transcriber(self.whisper_model).transcribe(
                    stems.vocals,
                    source_language,
                )
                write_transcript(workspace / "transcript.json", transcript)

                reference_wav = workspace / "speaker_reference.wav"
                create_speaker_reference(stems.vocals, reference_wav)

                translator = LocalTranslator(
                    source_language,
                    self.translation_model,
                )
                voice_cloner = VoiceCloner()

                for language in target_languages:
                    LOGGER.info("Producing %s dub", language.upper())
                    translated = translator.translate(transcript, language)
                    write_transcript(
                        workspace / f"transcript_{language}.json",
                        translated,
                    )
                    speech = voice_cloner.synthesize_timeline(
                        translated,
                        language,
                        reference_wav,
                        workspace,
                    )
                    mixed = workspace / f"{language}_mixed.wav"
                    duration = self.mixer.mix(speech, stems.bgm, mixed)
                    video_path = output_root / f"{language.upper()}_dubbed.mp4"
                    self.renderer.render(
                        image_path,
                        mixed,
                        video_path,
                        duration,
                    )
            except Exception:
                shutil.rmtree(output_root, ignore_errors=True)
                raise

        LOGGER.info("Completed output: %s", output_root)
        return output_root


def sanitize_title(title: str) -> str:
    title = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", " ", title)
    title = re.sub(r"\s+", " ", title).strip().strip(".")
    return title[:140] or "Audiobook"


def unique_output_root(base: Path, title: str) -> Path:
    candidate = base / title
    if not candidate.exists():
        return candidate
    index = 2
    while (base / f"{title}_{index}").exists():
        index += 1
    return base / f"{title}_{index}"


def normalize_language(language: str) -> str:
    normalized = language.strip().lower().replace("_", "-")
    aliases = {
        "eng": "en",
        "english": "en",
        "spa": "es",
        "spanish": "es",
        "tur": "tr",
        "turkish": "tr",
        "zho": "zh",
        "chinese": "zh",
    }
    return aliases.get(normalized, normalized)


def nllb_code(language: str) -> str:
    try:
        return NLLB_LANGUAGES[language]
    except KeyError as exc:
        raise PipelineError(f"NLLB language is not configured: {language}") from exc


def xtts_code(language: str) -> str:
    try:
        return XTTS_LANGUAGES[language]
    except KeyError as exc:
        raise PipelineError(f"XTTS-v2 language is not supported: {language}") from exc


def validate_languages(source: str, targets: Sequence[str]) -> None:
    if source not in NLLB_LANGUAGES:
        raise PipelineError(
            f"Unsupported source language '{source}'. Supported: "
            f"{', '.join(sorted(NLLB_LANGUAGES))}"
        )
    invalid = [
        language
        for language in targets
        if language not in NLLB_LANGUAGES or language not in XTTS_LANGUAGES
    ]
    if invalid:
        raise PipelineError(
            f"Unsupported target language(s): {', '.join(invalid)}. "
            f"XTTS targets: {', '.join(sorted(XTTS_LANGUAGES))}"
        )


def batched(
    items: Sequence[TranscriptSegment],
    size: int,
) -> Iterable[Sequence[TranscriptSegment]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def normalize_tts_text(text: str) -> str:
    """Clean text for natural-sounding TTS output.

    Ensures terminal punctuation (critical for XTTS-v2 prosody),
    fixes spacing after punctuation, and keeps literary pauses/emphasis
    explicit enough for the voice model to read like prose instead of a list.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return text
    text = text.replace("—", " — ").replace("–", " — ")
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+([.!?,;:…])", r"\1", text)
    text = re.sub(r"([.!?…])\s+([\"'”’])", r"\1\2 ", text)
    # Terminal punctuation lets XTTS produce natural sentence-final
    # intonation instead of trailing off robotically.
    if text[-1] not in '.!?…':
        text += '.'
    # Fix missing space after punctuation.
    text = re.sub(r'([.!?,;:])([A-Za-zÀ-ÿ])', r'\1 \2', text)
    # Collapse excessive punctuation.
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'([!?]){2,}', r'\1', text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def write_transcript(path: Path, segments: Sequence[TranscriptSegment]) -> None:
    path.write_text(
        json.dumps(
            [asdict(segment) for segment in segments],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_speaker_reference(vocals_path: Path, output_path: Path) -> None:
    audio = AudioSegment.from_file(vocals_path).set_channels(1).set_frame_rate(24_000)
    if len(audio) == 0:
        raise PipelineError("The vocal stem is empty.")

    threshold = max(-45.0, audio.dBFS - 14.0) if audio.dBFS != float("-inf") else -45
    regions = detect_nonsilent(
        audio,
        min_silence_len=400,
        silence_thresh=threshold,
        seek_step=10,
    )
    if regions:
        start, end = max(regions, key=lambda region: region[1] - region[0])
        center = (start + end) // 2
        clip_start = max(0, center - 15_000)
        clip_end = min(len(audio), clip_start + 30_000)
        clip_start = max(0, clip_end - 30_000)
    else:
        clip_start, clip_end = 0, min(len(audio), 30_000)

    if clip_end - clip_start < min(3_000, len(audio)):
        raise PipelineError("Could not find a usable XTTS speaker reference.")
    reference = audio[clip_start:clip_end].fade_in(20).fade_out(50)
    reference.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])


def media_duration(path: Path) -> float:
    try:
        probe = ffmpeg.probe(str(path))
        duration = float(probe["format"]["duration"])
    except (ffmpeg.Error, KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"Could not determine media duration: {path}") from exc
    if duration <= 0:
        raise PipelineError(f"Media duration must be positive: {path}")
    return duration


def parse_target_languages(values: Sequence[str]) -> list[str]:
    languages: list[str] = []
    for value in values:
        languages.extend(
            normalize_language(part)
            for part in value.split(",")
            if part.strip()
        )
    return list(dict.fromkeys(languages))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create locally translated, voice-cloned audiobook videos.",
    )
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Single still image used for the entire video",
    )
    parser.add_argument(
        "--source_lang",
        required=True,
        help="Source language code, for example en",
    )
    parser.add_argument(
        "--target_langs",
        required=True,
        nargs="+",
        help="Target codes separated by spaces or commas, for example tr es",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="Directory in which the YouTube-title folder is created",
    )
    parser.add_argument(
        "--whisper_model",
        default="large-v3",
        help="faster-whisper model name (default: large-v3)",
    )
    parser.add_argument(
        "--translation_model",
        default=TRANSLATION_MODEL,
        help=f"Local Hugging Face translation model (default: {TRANSLATION_MODEL})",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args()
    source_language = normalize_language(args.source_lang)
    target_languages = parse_target_languages(args.target_langs)
    if not target_languages:
        raise PipelineError("At least one target language is required.")

    pipeline = AudiobookPipeline(
        output_base=args.output_dir.expanduser().resolve(),
        whisper_model=args.whisper_model,
        translation_model=args.translation_model,
    )
    output = pipeline.run(
        url=args.url,
        image_path=args.image.expanduser().resolve(),
        source_language=source_language,
        target_languages=target_languages,
    )
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PipelineError, subprocess.CalledProcessError, ffmpeg.Error) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error
