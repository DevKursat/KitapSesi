#!/usr/bin/env python3
"""Crash-safe, segment-checkpointed audiobook dubbing worker."""

from __future__ import annotations

import argparse
import json
import logging
import os

# MPS fallback must be configured before any torch import.  HTDemucs and
# other models may contain Conv1d layers whose output-channel count exceeds
# Metal's current 65 536-channel limit.  The fallback transparently routes
# those individual ops to CPU while all other ops remain on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from yt_dlp import YoutubeDL

from audiobook_pipeline import (
    DownloadResult,
    LocalTranslator,
    MediaDownloader,
    PipelineError,
    ProperNameRecognizer,
    Runtime,
    StemSeparator,
    TranscriptSegment,
    Transcriber,
    VoiceCloner,
    build_proper_name_glossary,
    create_speaker_reference,
    media_duration,
    normalize_language,
    normalize_tts_text,
    sanitize_title,
    translation_quality_report,
    validate_languages,
)


LOGGER = logging.getLogger("resumable_pipeline")
SOURCE_CHUNK_SECONDS = 600
VIDEO_CHUNK_SECONDS = 1800
LANGUAGE_ARTIFACT_VERSION = 2
PUBLISH_SAFE_RIGHTS = {"owned", "licensed", "public_domain", "permission"}


class JobCancelled(PipelineError):
    """Raised after a user requests cancellation."""


class JobStore:
    """Atomic manifest/control storage shared by the worker and web UI."""

    def __init__(self, job_dir: Path) -> None:
        self.job_dir = job_dir.resolve()
        self.manifest_path = self.job_dir / "job.json"
        self.control_path = self.job_dir / "control.json"
        self.log_path = self.job_dir / "worker.log"
        self.work_dir = self.job_dir / "work"
        self.output_dir = self.job_dir / "output"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def update(self, **changes: Any) -> dict[str, Any]:
        data = self.read()
        data.update(changes)
        data["updated_at"] = now_iso()
        atomic_json(self.manifest_path, data)
        return data

    def progress(
        self,
        stage: str,
        message: str,
        completed: int,
        total: int,
        overall: float,
    ) -> None:
        self.update(
            status="running",
            stage=stage,
            message=message,
            stage_completed=completed,
            stage_total=max(total, 1),
            progress=round(max(0.0, min(overall, 100.0)), 2),
            error=None,
        )

    def control(self) -> str:
        if not self.control_path.exists():
            return "run"
        try:
            return json.loads(
                self.control_path.read_text(encoding="utf-8")
            ).get("action", "run")
        except (OSError, json.JSONDecodeError):
            return "run"

    def checkpoint(self) -> None:
        action = self.control()
        if action == "cancel":
            raise JobCancelled("İş kullanıcı tarafından iptal edildi.")
        while action == "pause":
            self.update(
                status="paused",
                message="Güvenli checkpoint noktasında duraklatıldı.",
            )
            time.sleep(1)
            action = self.control()
            if action == "cancel":
                raise JobCancelled("İş kullanıcı tarafından iptal edildi.")
        if self.read().get("status") == "paused":
            self.update(status="running", message="Checkpoint'ten devam ediliyor.")


class ResumableAudiobookPipeline:
    """A long-running pipeline whose every expensive unit is restartable."""

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self.downloader = MediaDownloader()
        self.separator = StemSeparator()

    def run(self) -> Path:
        job = self.store.read()
        image = Path(job["image_path"]).resolve()
        source_language = normalize_language(job["source_language"])
        target_languages = [
            normalize_language(language) for language in job["target_languages"]
        ]
        validate_languages(source_language, target_languages)
        Runtime.validate(image)
        if job.get("content_rights") not in PUBLISH_SAFE_RIGHTS:
            raise PipelineError(
                "Yayın hakkı güvenli değil. Kendi içeriğiniz, lisanslı/izinli "
                "içerik veya kamu malı eser seçilmeden üretim başlatılmaz."
            )

        self.store.update(
            status="running",
            started_at=job.get("started_at") or now_iso(),
            worker_pid=os.getpid(),
            error=None,
        )
        self.store.checkpoint()

        download = self._download(job["url"])
        visual_mode = job.get("visual_mode", "still_image")
        subtitle_mode = job.get("subtitle_mode", "translated")
        source_video = (
            self._download_source_video(job["url"])
            if visual_mode == "source_video"
            else None
        )
        chunks = self._split_source(download.audio_path)
        stems = self._separate_chunks(chunks)
        transcript = self._transcribe_chunks(stems, source_language)
        proper_names = self._proper_name_glossary(
            transcript,
            source_language,
        )
        reference = self._speaker_reference(stems)
        bgm = self._assemble_bgm(stems, source_language)

        with model_loading_status(
            self.store,
            stage="translation_model",
            label="NLLB çeviri modeli",
            overall=48,
            cache_pattern=(
                "~/.cache/huggingface/hub/"
                "models--facebook--nllb-200-distilled-600M"
            ),
        ):
            translator = LocalTranslator(
                source_language,
                job.get(
                    "translation_model",
                    "facebook/nllb-200-distilled-600M",
                ),
            )
        output_root = self.store.output_dir / sanitize_title(download.title)
        output_root.mkdir(parents=True, exist_ok=True)
        final_bgm = output_root / f"{source_language.upper()}_original_bgm.wav"
        if not final_bgm.exists():
            shutil.copy2(bgm, final_bgm)

        translations: dict[str, list[TranscriptSegment]] = {}
        for language_index, language in enumerate(target_languages):
            self.store.checkpoint()
            translations[language] = self._translate_language(
                translator,
                transcript,
                proper_names,
                language,
                language_index,
                len(target_languages),
            )

        del translator
        torch.mps.empty_cache()
        voice_license = self.store.read().get("voice_license")
        if voice_license not in {"commercial", "noncommercial"}:
            raise PipelineError(
                "XTTS lisans onayı eksik. Arayüzden ticari lisans veya "
                "ticari olmayan CPML seçimini kaydedip Devam Et'e basın."
            )
        os.environ["COQUI_TOS_AGREED"] = "1"
        with model_loading_status(
            self.store,
            stage="voice_model",
            label="XTTS-v2 ses klonlama modeli",
            overall=60,
            cache_pattern="~/Library/Application Support/tts",
        ):
            voice_cloner = VoiceCloner()

        quality_results: list[dict[str, Any]] = []
        for language_index, language in enumerate(target_languages):
            self.store.checkpoint()
            speech = self._synthesize_language(
                voice_cloner,
                translations[language],
                language,
                reference,
                language_index,
                len(target_languages),
            )
            quality_results.append(self._render_language(
                speech,
                bgm,
                image,
                source_video,
                visual_mode,
                subtitle_mode,
                translations[language],
                output_root,
                language,
                language_index,
                len(target_languages),
            ))

        quality_report = {
            "source_duration_seconds": round(media_duration(download.audio_path), 3),
            "bgm_duration_seconds": round(media_duration(bgm), 3),
            "transcript_segments": len(transcript),
            "transcript_last_end_seconds": round(transcript[-1].end, 3),
            "source_suitability": source_suitability(
                transcript,
                media_duration(download.audio_path),
            ),
            "content_rights": job.get("content_rights"),
            "rights_notes": job.get("rights_notes", ""),
            "rights_accepted_at": job.get("rights_accepted_at"),
            "visual_mode": visual_mode,
            "subtitle_mode": subtitle_mode,
            "languages": quality_results,
            "voice_license": voice_license,
            "status": "validated",
        }
        atomic_json(output_root / "quality_report.json", quality_report)

        self.store.update(
            status="completed",
            stage="completed",
            message="Tüm diller başarıyla tamamlandı.",
            progress=100.0,
            completed_at=now_iso(),
            output_path=str(output_root),
            worker_pid=None,
        )
        return output_root

    def _download(self, url: str) -> DownloadResult:
        source = self.store.work_dir / "source.wav"
        metadata = self.store.work_dir / "download.json"
        if valid_media(source) and metadata.exists():
            data = json.loads(metadata.read_text(encoding="utf-8"))
            return DownloadResult(title=data["title"], audio_path=source)

        self.store.progress("download", "YouTube sesi indiriliyor.", 0, 1, 1)
        result = self.downloader.download(url, self.store.work_dir)
        atomic_json(metadata, {"title": result.title})
        self.store.update(title=result.title)
        self.store.progress("download", "Kaynak ses indirildi.", 1, 1, 5)
        return result

    def _download_source_video(self, url: str) -> Path:
        visual = self.store.work_dir / "source_visual.mp4"
        if valid_media(visual):
            return visual

        self.store.progress(
            "download_video",
            "Orijinal video görüntüsü indiriliyor.",
            0,
            1,
            5,
        )
        scratch = self.store.work_dir / "source_visual_download"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(exist_ok=True)
        template = str(scratch / "source_visual.%(ext)s")
        options = {
            "format": (
                "bestvideo[height<=1080][ext=mp4]/"
                "bestvideo[height<=1080]/best[height<=1080]/best"
            ),
            "noplaylist": True,
            "outtmpl": template,
            "quiet": False,
            "no_warnings": False,
        }
        with YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)
        candidates = sorted(
            path for path in scratch.glob("source_visual.*") if path.is_file()
        )
        if not candidates:
            raise PipelineError("YouTube video görüntüsü indirilemedi.")

        partial = visual.with_suffix(".partial.mp4")
        run_ffmpeg(
            "-i",
            str(candidates[0]),
            "-an",
            "-vf",
            "scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(partial),
        )
        partial.replace(visual)
        shutil.rmtree(scratch, ignore_errors=True)
        self.store.progress(
            "download_video",
            "Orijinal video görüntüsü hazır.",
            1,
            1,
            6,
        )
        return visual

    def _split_source(self, source: Path) -> list[Path]:
        duration = media_duration(source)
        count = max(1, int((duration + SOURCE_CHUNK_SECONDS - 0.001) // SOURCE_CHUNK_SECONDS))
        chunks_dir = self.store.work_dir / "source_chunks"
        chunks_dir.mkdir(exist_ok=True)
        chunks: list[Path] = []
        for index in range(count):
            self.store.checkpoint()
            chunk = chunks_dir / f"{index:05d}.wav"
            chunks.append(chunk)
            if valid_media(chunk):
                continue
            start = index * SOURCE_CHUNK_SECONDS
            length = min(SOURCE_CHUNK_SECONDS, duration - start)
            self.store.progress(
                "split",
                f"Kaynak bölünüyor: {index + 1}/{count}",
                index,
                count,
                5 + 3 * index / count,
            )
            partial = chunk.with_suffix(".partial.wav")
            run_ffmpeg(
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{length:.3f}",
                "-i",
                str(source),
                "-vn",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "pcm_f32le",
                str(partial),
            )
            partial.replace(chunk)
        self.store.progress("split", "Kaynak parçaları hazır.", count, count, 8)
        return chunks

    def _separate_chunks(self, chunks: Sequence[Path]) -> list[tuple[Path, Path]]:
        stem_root = self.store.work_dir / "stems"
        stem_root.mkdir(exist_ok=True)
        stems: list[tuple[Path, Path]] = []
        total = len(chunks)
        for index, chunk in enumerate(chunks):
            self.store.checkpoint()
            final_dir = stem_root / chunk.stem
            vocals = final_dir / "vocals.wav"
            bgm = final_dir / "bgm.wav"
            stems.append((vocals, bgm))
            if valid_media(vocals) and valid_media(bgm):
                continue
            self.store.progress(
                "separation",
                (
                    f"Ses ve fon ayrılıyor: {index + 1}/{total}. "
                    "HTDemucs 4 model × 2 analiz yapar; günlükte aynı "
                    "sürenin tekrarı normaldir."
                ),
                index,
                total,
                8 + 22 * index / total,
            )
            scratch = self.store.work_dir / "demucs_scratch" / chunk.stem
            shutil.rmtree(scratch, ignore_errors=True)
            result = self.separator.separate(chunk, scratch)
            final_dir.mkdir(parents=True, exist_ok=True)
            atomic_copy(result.vocals, vocals)
            atomic_copy(result.bgm, bgm)
            shutil.rmtree(scratch, ignore_errors=True)
        self.store.progress(
            "separation",
            "Tüm kaynak parçaların stem ayrımı tamamlandı.",
            total,
            total,
            30,
        )
        return stems

    def _transcribe_chunks(
        self,
        stems: Sequence[tuple[Path, Path]],
        source_language: str,
    ) -> list[TranscriptSegment]:
        transcript_dir = self.store.work_dir / "transcripts"
        transcript_dir.mkdir(exist_ok=True)
        transcriber: Transcriber | None = None
        all_segments: list[TranscriptSegment] = []
        total = len(stems)
        for index, (vocals, _) in enumerate(stems):
            self.store.checkpoint()
            transcript_path = transcript_dir / f"{index:05d}.json"
            if transcript_path.exists():
                local_segments = read_segments(transcript_path)
            else:
                if transcriber is None:
                    transcriber = Transcriber(
                        self.store.read().get("whisper_model", "large-v3")
                    )
                self.store.progress(
                    "transcription",
                    f"Konuşma yazıya çevriliyor: {index + 1}/{total}",
                    index,
                    total,
                    30 + 14 * index / total,
                )
                local_segments = transcriber.transcribe(vocals, source_language)
                atomic_json(
                    transcript_path,
                    [asdict(segment) for segment in local_segments],
                )
            offset = index * SOURCE_CHUNK_SECONDS
            all_segments.extend(
                TranscriptSegment(
                    start=segment.start + offset,
                    end=segment.end + offset,
                    text=segment.text,
                )
                for segment in local_segments
            )
        combined = self.store.work_dir / "transcript.json"
        atomic_json(combined, [asdict(segment) for segment in all_segments])
        self.store.progress(
            "transcription",
            f"{len(all_segments)} konuşma segmenti hazır.",
            total,
            total,
            44,
        )
        return all_segments

    def _speaker_reference(
        self,
        stems: Sequence[tuple[Path, Path]],
    ) -> Path:
        reference = self.store.work_dir / "speaker_reference.wav"
        if valid_media(reference):
            return reference
        self.store.progress(
            "voice_reference",
            "Anlatıcının ses referansı hazırlanıyor.",
            0,
            1,
            45,
        )
        candidates = sorted(
            (vocals for vocals, _ in stems),
            key=media_duration,
            reverse=True,
        )
        create_speaker_reference(candidates[0], reference)
        self.store.progress(
            "voice_reference",
            "Anlatıcı sesi hazır.",
            1,
            1,
            46,
        )
        return reference

    def _proper_name_glossary(
        self,
        transcript: Sequence[TranscriptSegment],
        source_language: str,
    ) -> frozenset[str]:
        glossary_path = (
            self.store.work_dir
            / f"proper_names_v{LANGUAGE_ARTIFACT_VERSION}.json"
        )
        if glossary_path.exists():
            return frozenset(
                json.loads(glossary_path.read_text(encoding="utf-8"))
            )

        texts = [segment.text for segment in transcript]
        names = set(build_proper_name_glossary(texts))
        if source_language == "tr":
            with model_loading_status(
                self.store,
                stage="name_detection",
                label="Türkçe özel isim tanıma modeli",
                overall=45,
                cache_pattern=(
                    "~/.cache/huggingface/hub/"
                    "models--akdeniz27--bert-base-turkish-cased-ner"
                ),
            ):
                recognizer = ProperNameRecognizer()
            self.store.progress(
                "name_detection",
                "Kişi adları eser boyunca taranıyor.",
                0,
                len(texts),
                46,
            )
            names.update(recognizer.extract_people(texts))
            del recognizer
            torch.mps.empty_cache()

        atomic_json(glossary_path, sorted(names))
        self.store.progress(
            "name_detection",
            f"{len(names)} korunacak özel isim bulundu.",
            len(texts),
            max(len(texts), 1),
            47,
        )
        return frozenset(names)

    def _assemble_bgm(
        self,
        stems: Sequence[tuple[Path, Path]],
        source_language: str,
    ) -> Path:
        bgm = self.store.work_dir / f"{source_language}_original_bgm.wav"
        if valid_media(bgm):
            return bgm
        self.store.progress(
            "bgm",
            "Fon müziği parçaları birleştiriliyor.",
            0,
            1,
            47,
        )
        concatenate_wavs([path for _, path in stems], bgm)
        self.store.progress("bgm", "Fon müziği hazır.", 1, 1, 48)
        return bgm

    def _translate_language(
        self,
        translator: LocalTranslator,
        transcript: Sequence[TranscriptSegment],
        glossary: frozenset[str],
        language: str,
        language_index: int,
        language_count: int,
    ) -> list[TranscriptSegment]:
        translated_path = (
            self.store.work_dir
            / f"translation_v{LANGUAGE_ARTIFACT_VERSION}_{language}.json"
        )
        existing = read_segments(translated_path) if translated_path.exists() else []
        completed = len(existing)
        if completed > len(transcript):
            translated_path.unlink()
            existing = []
            completed = 0

        batch_size = 8
        for start in range(completed, len(transcript), batch_size):
            self.store.checkpoint()
            batch = transcript[start : start + batch_size]
            translated_batch = translator.translate(
                batch,
                language,
                glossary=glossary,
            )
            existing.extend(translated_batch)
            atomic_json(
                translated_path,
                [asdict(segment) for segment in existing],
            )
            phase = (start + len(batch)) / max(len(transcript), 1)
            self.store.progress(
                "translation",
                (
                    f"{language.upper()} çevriliyor: "
                    f"{min(start + len(batch), len(transcript))}/{len(transcript)}"
                ),
                min(start + len(batch), len(transcript)),
                len(transcript),
                translation_progress(
                    language_index,
                    language_count,
                    phase,
                ),
            )
        report = translation_quality_report(
            transcript,
            existing,
            language,
            glossary,
        )
        report_path = (
            self.store.work_dir
            / f"translation_quality_v{LANGUAGE_ARTIFACT_VERSION}_{language}.json"
        )
        atomic_json(report_path, report)
        if report.get("status") != "passed":
            issue_count = report.get("issue_count", 0)
            raise PipelineError(
                f"{language.upper()} çeviri kalite kapısı başarısız "
                f"({issue_count} sorun). TTS başlamadan durduruldu; "
                f"rapor: {report_path}"
            )
        return existing

    def _synthesize_language(
        self,
        cloner: VoiceCloner,
        segments: Sequence[TranscriptSegment],
        language: str,
        reference: Path,
        language_index: int,
        language_count: int,
    ) -> Path:
        version = f"v{LANGUAGE_ARTIFACT_VERSION}"
        tts_dir = self.store.work_dir / "tts" / version / language
        unit_dir = self.store.work_dir / "timeline" / version / language
        tts_dir.mkdir(parents=True, exist_ok=True)
        unit_dir.mkdir(parents=True, exist_ok=True)

        cursor = 0.0
        units: list[Path] = []
        subtitle_entries: list[dict[str, Any]] = []
        total = len(segments)
        for index, segment in enumerate(segments):
            self.store.checkpoint()
            speech = tts_dir / f"{index:07d}.wav"
            if not valid_media(speech):
                self.store.progress(
                    "voice_cloning",
                    f"{language.upper()} seslendiriliyor: {index + 1}/{total}",
                    index,
                    total,
                    synthesis_progress(
                        language_index,
                        language_count,
                        0.7 * index / max(total, 1),
                    ),
                )
                partial_dir = tts_dir / ".partial"
                partial_dir.mkdir(exist_ok=True)
                partial = partial_dir / speech.name
                cloner.tts.tts_to_file(
                    text=normalize_tts_text(segment.text),
                    speaker_wav=str(reference),
                    language=language if language != "zh" else "zh-cn",
                    file_path=str(partial),
                    split_sentences=True,
                )
                if not valid_media(partial):
                    raise PipelineError(f"XTTS geçerli ses üretemedi: {index}")
                partial.replace(speech)

            speech_duration = media_duration(speech)
            position = max(segment.start, cursor)
            gap = max(0.0, position - cursor)
            subtitle_entries.append(
                {
                    "index": index,
                    "start": round(position, 3),
                    "end": round(position + speech_duration, 3),
                    "text": segment.text,
                }
            )
            unit = unit_dir / f"{index:07d}.wav"
            expected = gap + speech_duration + 0.08
            if not valid_media(unit) or abs(media_duration(unit) - expected) > 0.08:
                create_timeline_unit(speech, unit, gap, 0.08)
            cursor += media_duration(unit)
            units.append(unit)

        speech_timeline = (
            self.store.work_dir
            / f"{language}_speech_v{LANGUAGE_ARTIFACT_VERSION}.wav"
        )
        if not valid_media(speech_timeline):
            concatenate_wavs(units, speech_timeline)
        atomic_json(
            self.store.work_dir
            / f"subtitle_timeline_v{LANGUAGE_ARTIFACT_VERSION}_{language}.json",
            subtitle_entries,
        )
        self.store.progress(
            "voice_cloning",
            f"{language.upper()} doğal tempoda seslendirildi.",
            total,
            total,
            synthesis_progress(
                language_index,
                language_count,
                0.7,
            ),
        )
        torch.mps.empty_cache()
        return speech_timeline

    def _render_language(
        self,
        speech: Path,
        bgm: Path,
        image: Path,
        source_video: Path | None,
        visual_mode: str,
        subtitle_mode: str,
        translated_segments: Sequence[TranscriptSegment],
        output_root: Path,
        language: str,
        language_index: int,
        language_count: int,
    ) -> dict[str, Any]:
        audio_flac = output_root / f"{language.upper()}_dubbed_audio.flac"
        youtube_audio = output_root / f"{language.upper()}_youtube_audio.m4a"
        video = output_root / f"{language.upper()}_dubbed.mp4"
        captions = output_root / f"{language.upper()}_captions.srt"
        raw_video = output_root / f".{language.upper()}_dubbed_raw.mp4"
        version_dir = self.store.work_dir / "language_versions"
        version_dir.mkdir(exist_ok=True)
        version_marker = version_dir / f"{language}.json"
        version_data = (
            json.loads(version_marker.read_text(encoding="utf-8"))
            if version_marker.exists()
            else {}
        )
        if (
            version_data.get("version") != LANGUAGE_ARTIFACT_VERSION
            or version_data.get("visual_mode", "still_image") != visual_mode
            or version_data.get("subtitle_mode", "translated") != subtitle_mode
        ):
            for stale in (audio_flac, youtube_audio, video, raw_video, captions):
                stale.unlink(missing_ok=True)

        if subtitle_mode == "translated":
            self._subtitle_entries(language, translated_segments, captions)

        self.store.progress(
            "mixing",
            f"{language.upper()} ses ve fon müziği birleştiriliyor.",
            0,
            3,
            synthesis_progress(language_index, language_count, 0.7),
        )
        if not valid_media(audio_flac):
            mix_audio(speech, bgm, audio_flac)
        self.store.progress(
            "mixing",
            f"{language.upper()} YouTube ses kanalı hazırlanıyor.",
            1,
            3,
            synthesis_progress(language_index, language_count, 0.8),
        )
        if not valid_media(youtube_audio):
            run_ffmpeg(
                "-i",
                str(audio_flac),
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "320k",
                "-movflags",
                "+faststart",
                str(youtube_audio.with_suffix(".partial.m4a")),
            )
            youtube_audio.with_suffix(".partial.m4a").replace(youtube_audio)

        self.store.progress(
            "rendering",
            (
                f"{language.upper()} orijinal video üzerine dublaj render ediliyor."
                if visual_mode == "source_video"
                else f"{language.upper()} sabit görselli video oluşturuluyor."
            ),
            2,
            3,
            synthesis_progress(language_index, language_count, 0.86),
        )
        if not valid_media(video):
            render_target = raw_video if subtitle_mode == "translated" else video
            self._render_video_chunks(
                image,
                source_video,
                visual_mode,
                audio_flac,
                render_target,
                language,
                language_index,
                language_count,
            )
            if subtitle_mode == "translated":
                mux_soft_subtitles(render_target, captions, video, language)
        self.store.progress(
            "rendering",
            f"{language.upper()} çıktıları tamamlandı.",
            3,
            3,
            synthesis_progress(language_index, language_count, 1),
        )
        result = validate_language_outputs(
            language,
            speech,
            audio_flac,
            youtube_audio,
            video,
        )
        atomic_json(
            version_marker,
            {
                "version": LANGUAGE_ARTIFACT_VERSION,
                "visual_mode": visual_mode,
                "subtitle_mode": subtitle_mode,
                "completed_at": now_iso(),
            },
        )
        return result

    def _subtitle_entries(
        self,
        language: str,
        translated_segments: Sequence[TranscriptSegment],
        output_ass: Path,
    ) -> list[dict[str, Any]]:
        timeline_path = (
            self.store.work_dir
            / f"subtitle_timeline_v{LANGUAGE_ARTIFACT_VERSION}_{language}.json"
        )
        if not timeline_path.exists():
            raise PipelineError(f"{language.upper()} altyazı timeline bulunamadı.")
        entries = json.loads(timeline_path.read_text(encoding="utf-8"))
        texts = [segment.text for segment in translated_segments]
        for entry in entries:
            index = int(entry["index"])
            if 0 <= index < len(texts):
                entry["text"] = texts[index]
        write_srt_subtitles(output_ass, entries)
        return entries

    def _render_video_chunks(
        self,
        image: Path,
        source_video: Path | None,
        visual_mode: str,
        audio: Path,
        destination: Path,
        language: str,
        language_index: int,
        language_count: int,
    ) -> None:
        duration = media_duration(audio)
        count = max(
            1,
            int((duration + VIDEO_CHUNK_SECONDS - 0.001) // VIDEO_CHUNK_SECONDS),
        )
        chunks_dir = (
            self.store.work_dir
            / "video_chunks"
            / f"v{LANGUAGE_ARTIFACT_VERSION}"
            / language
        )
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[Path] = []
        for index in range(count):
            self.store.checkpoint()
            chunk = chunks_dir / f"{index:05d}.mp4"
            chunks.append(chunk)
            if valid_media(chunk):
                continue
            start = index * VIDEO_CHUNK_SECONDS
            length = min(VIDEO_CHUNK_SECONDS, duration - start)
            self.store.progress(
                "rendering",
                (
                    f"{language.upper()} video render: "
                    f"{index + 1}/{count}"
                ),
                index,
                count,
                synthesis_progress(
                    language_index,
                    language_count,
                    0.86 + 0.13 * index / count,
                ),
            )
            if visual_mode == "source_video":
                if source_video is None:
                    raise PipelineError("Orijinal video görseli hazır değil.")
                render_source_video_chunk(
                    source_video,
                    audio,
                    chunk,
                    start,
                    length,
                    duration,
                )
            else:
                render_video_chunk(
                    image,
                    audio,
                    chunk,
                    start,
                    length,
                    duration,
                )
        concatenate_mp4s(chunks, destination)


def translation_progress(
    language_index: int,
    language_count: int,
    phase: float,
) -> float:
    width = 12 / max(language_count, 1)
    return 48 + width * (
        language_index + max(0.0, min(phase, 1.0))
    )


def synthesis_progress(
    language_index: int,
    language_count: int,
    phase: float,
) -> float:
    width = 40 / max(language_count, 1)
    return 60 + width * (
        language_index + max(0.0, min(phase, 1.0))
    )


def read_segments(path: Path) -> list[TranscriptSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSegment(**item) for item in data]


def valid_media(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 128:
        return False
    try:
        return media_duration(path) > 0.02
    except PipelineError:
        return False


def validate_language_outputs(
    language: str,
    speech: Path,
    lossless_audio: Path,
    youtube_audio: Path,
    video: Path,
) -> dict[str, Any]:
    paths = (speech, lossless_audio, youtube_audio, video)
    invalid = [str(path) for path in paths if not valid_media(path)]
    if invalid:
        raise PipelineError(
            "Çıktı doğrulaması başarısız; geçersiz dosyalar: "
            + ", ".join(invalid)
        )
    durations = {
        "natural_speech_seconds": media_duration(speech),
        "lossless_audio_seconds": media_duration(lossless_audio),
        "youtube_audio_seconds": media_duration(youtube_audio),
        "video_seconds": media_duration(video),
    }
    reference = durations["lossless_audio_seconds"]
    mismatches = {
        name: duration
        for name, duration in durations.items()
        if name != "natural_speech_seconds"
        and abs(duration - reference) > 0.35
    }
    if mismatches:
        raise PipelineError(
            f"{language.upper()} süre doğrulaması başarısız: {durations}"
        )
    return {
        "language": language,
        **{name: round(value, 3) for name, value in durations.items()},
        "duration_validation": "passed",
    }


def source_suitability(
    transcript: Sequence[TranscriptSegment],
    source_duration: float,
) -> str:
    if not transcript or source_duration <= 0:
        return "failed"
    spoken_duration = sum(
        max(0.0, segment.end - segment.start)
        for segment in transcript
    )
    return (
        "passed"
        if spoken_duration / source_duration >= 0.05
        else "failed"
    )


@contextmanager
def model_loading_status(
    store: JobStore,
    stage: str,
    label: str,
    overall: float,
    cache_pattern: str,
):
    stop = threading.Event()
    started = time.monotonic()
    cache_root = Path(cache_pattern).expanduser()

    def heartbeat() -> None:
        while not stop.wait(2):
            elapsed = int(time.monotonic() - started)
            downloaded = directory_size(cache_root)
            size_text = (
                f", yerel cache {downloaded / 1024**2:.0f} MB"
                if downloaded
                else ""
            )
            store.progress(
                stage,
                (
                    f"{label} ilk kullanım için indiriliyor/yükleniyor "
                    f"({elapsed} sn{size_text})."
                ),
                elapsed,
                max(elapsed + 1, 1),
                overall,
            )

    store.progress(
        stage,
        f"{label} hazırlanıyor. İlk çalıştırma uzun sürebilir.",
        0,
        1,
        overall,
    )
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=3)


def directory_size(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    try:
        for path in root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    except OSError:
        return total
    return total


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def run_ffmpeg(*arguments: str) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments]
    subprocess.run(command, check=True)


def concat_file(paths: Iterable[Path], destination: Path) -> Path:
    manifest = destination.with_suffix(destination.suffix + ".concat.txt")
    lines = [
        "file '" + str(path.resolve()).replace("'", "'\\''") + "'"
        for path in paths
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def concatenate_wavs(paths: Sequence[Path], destination: Path) -> None:
    if not paths:
        raise PipelineError("Birleştirilecek ses parçası bulunamadı.")
    manifest = concat_file(paths, destination)
    partial = destination.with_suffix(".partial.wav")
    run_ffmpeg(
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-c:a",
        "pcm_f32le",
        "-rf64",
        "auto",
        str(partial),
    )
    partial.replace(destination)


def create_timeline_unit(
    speech: Path,
    destination: Path,
    leading_silence: float,
    trailing_silence: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.wav")
    if leading_silence > 0.005:
        run_ffmpeg(
            "-f",
            "lavfi",
            "-t",
            f"{leading_silence:.6f}",
            "-i",
            "anullsrc=r=24000:cl=mono",
            "-i",
            str(speech),
            "-f",
            "lavfi",
            "-t",
            f"{trailing_silence:.6f}",
            "-i",
            "anullsrc=r=24000:cl=mono",
            "-filter_complex",
            "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
            "-map",
            "[out]",
            "-c:a",
            "pcm_f32le",
            str(partial),
        )
    else:
        run_ffmpeg(
            "-i",
            str(speech),
            "-f",
            "lavfi",
            "-t",
            f"{trailing_silence:.6f}",
            "-i",
            "anullsrc=r=24000:cl=mono",
            "-filter_complex",
            "[0:a][1:a]concat=n=2:v=0:a=1[out]",
            "-map",
            "[out]",
            "-c:a",
            "pcm_f32le",
            str(partial),
        )
    partial.replace(destination)


def mix_audio(speech: Path, bgm: Path, destination: Path) -> None:
    duration = media_duration(speech)
    partial = destination.with_suffix(".partial.flac")
    run_ffmpeg(
        "-i",
        str(speech),
        "-stream_loop",
        "-1",
        "-i",
        str(bgm),
        "-filter_complex",
        (
            f"[1:a]atrim=duration={duration:.6f},asetpts=PTS-STARTPTS,"
            "volume=0.35[bgm];"
            "[0:a][bgm]amix=inputs=2:duration=first:"
            "dropout_transition=0:normalize=0[out]"
        ),
        "-map",
        "[out]",
        "-t",
        f"{duration:.6f}",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-c:a",
        "flac",
        "-compression_level",
        "8",
        str(partial),
    )
    partial.replace(destination)


def ffmpeg_filter_path(path: Path) -> str:
    """Quote a filesystem path for FFmpeg filter arguments."""
    value = str(path.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    return "'" + value + "'"


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    centiseconds = int(round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def srt_time(seconds: float) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def ass_text(text: str) -> str:
    clean = " ".join(text.replace("\n", " ").split())
    clean = clean.replace("{", "(").replace("}", ")")
    if len(clean) <= 68:
        return clean
    words = clean.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > 46 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
        if len(lines) == 1 and len(current) > 46:
            break
    if current:
        lines.append(current)
    return "\\N".join(lines[:2])


def srt_text(text: str) -> str:
    return ass_text(text).replace("\\N", "\n")


def write_srt_subtitles(
    destination: Path,
    entries: Sequence[dict[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for number, entry in enumerate(entries, start=1):
        start = float(entry["start"])
        end = float(entry["end"])
        text = srt_text(str(entry.get("text") or ""))
        if end - start < 0.15 or not text:
            continue
        blocks.append(
            f"{number}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n"
        )
    destination.write_text("\n".join(blocks), encoding="utf-8")


def mux_soft_subtitles(
    video: Path,
    subtitles: Path,
    destination: Path,
    language: str,
) -> None:
    partial = destination.with_suffix(".partial.mp4")
    run_ffmpeg(
        "-i",
        str(video),
        "-i",
        str(subtitles),
        "-map",
        "0",
        "-map",
        "1:0",
        "-c",
        "copy",
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        f"language={language}",
        "-movflags",
        "+faststart",
        str(partial),
    )
    partial.replace(destination)


def write_ass_subtitles(
    destination: Path,
    entries: Sequence[dict[str, Any]],
    offset: float = 0.0,
    duration: float | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    window_end = None if duration is None else offset + duration
    events: list[str] = []
    for entry in entries:
        start = float(entry["start"])
        end = float(entry["end"])
        if window_end is not None and (end <= offset or start >= window_end):
            continue
        start = max(start - offset, 0.0)
        end = end - offset
        if duration is not None:
            end = min(end, duration)
        if end - start < 0.15:
            continue
        text = ass_text(str(entry.get("text") or ""))
        if not text:
            continue
        events.append(
            "Dialogue: 0,"
            f"{ass_time(start)},{ass_time(end)},CreatorDub,,0,0,0,,{text}"
        )

    destination.write_text(
        "\n".join(
            [
                "[Script Info]",
                "ScriptType: v4.00+",
                "PlayResX: 1920",
                "PlayResY: 1080",
                "ScaledBorderAndShadow: yes",
                "",
                "[V4+ Styles]",
                (
                    "Format: Name,Fontname,Fontsize,PrimaryColour,"
                    "SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
                    "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
                    "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,"
                    "MarginV,Encoding"
                ),
                (
                    "Style: CreatorDub,Arial,54,&H00FFFFFF,&H000000FF,"
                    "&H00101010,&H9A000000,-1,0,0,0,100,100,0,0,"
                    "4,2,1,2,130,130,74,1"
                ),
                "",
                "[Events]",
                "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
                *events,
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_video_chunk(
    image: Path,
    audio: Path,
    destination: Path,
    start: float,
    length: float,
    total_duration: float,
    subtitles: Path | None = None,
) -> None:
    partial = destination.with_suffix(".partial.mp4")
    filter_graph = (
        "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black[base];"
        "[2:v]drawbox=x=0:y=0:w=iw:h=ih:color=red@0.95:t=fill[bar];"
        "[base][bar]overlay=x='-overlay_w+main_w*"
        f"min((t+{start:.6f})/{total_duration:.6f},1)':"
        "y='main_h-overlay_h':eval=frame[progress]"
    )
    if subtitles is not None:
        filter_graph += (
            f";[progress]subtitles=filename={ffmpeg_filter_path(subtitles)}[outv]"
        )
    else:
        filter_graph += ";[progress]copy[outv]"
    run_ffmpeg(
        "-loop",
        "1",
        "-framerate",
        "10",
        "-i",
        str(image),
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{length:.6f}",
        "-i",
        str(audio),
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=1920x10:r=10",
        "-filter_complex",
        filter_graph,
        "-map",
        "[outv]",
        "-map",
        "1:a:0",
        "-t",
        f"{length:.6f}",
        "-shortest",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-movflags",
        "+faststart",
        str(partial),
    )
    partial.replace(destination)


def render_source_video_chunk(
    source_video: Path,
    audio: Path,
    destination: Path,
    start: float,
    length: float,
    total_duration: float,
    subtitles: Path | None = None,
) -> None:
    partial = destination.with_suffix(".partial.mp4")
    filter_graph = (
        "[0:v]fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1[base];"
        "[2:v]drawbox=x=0:y=0:w=iw:h=ih:color=red@0.95:t=fill[bar];"
        "[base][bar]overlay=x='-overlay_w+main_w*"
        f"min((t+{start:.6f})/{total_duration:.6f},1)':"
        "y='main_h-overlay_h':eval=frame[progress]"
    )
    if subtitles is not None:
        filter_graph += (
            f";[progress]subtitles=filename={ffmpeg_filter_path(subtitles)}[outv]"
        )
    else:
        filter_graph += ";[progress]copy[outv]"
    run_ffmpeg(
        "-stream_loop",
        "-1",
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{length:.6f}",
        "-i",
        str(source_video),
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{length:.6f}",
        "-i",
        str(audio),
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=1920x10:r=30",
        "-filter_complex",
        filter_graph,
        "-map",
        "[outv]",
        "-map",
        "1:a:0",
        "-t",
        f"{length:.6f}",
        "-shortest",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-movflags",
        "+faststart",
        str(partial),
    )
    partial.replace(destination)


def concatenate_mp4s(paths: Sequence[Path], destination: Path) -> None:
    if not paths:
        raise PipelineError("Birleştirilecek video parçası bulunamadı.")
    manifest = concat_file(paths, destination)
    partial = destination.with_suffix(".partial.mp4")
    run_ffmpeg(
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(partial),
    )
    partial.replace(destination)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args()
    store = JobStore(args.job_dir)
    try:
        output = ResumableAudiobookPipeline(store).run()
        LOGGER.info("Completed: %s", output)
        return 0
    except JobCancelled as error:
        store.update(
            status="cancelled",
            message=str(error),
            worker_pid=None,
        )
        LOGGER.warning("%s", error)
        return 2
    except Exception as error:
        LOGGER.exception("Pipeline failed")
        store.update(
            status="failed",
            message="Hata oluştu. Devam et ile son checkpoint'ten sürdürülebilir.",
            error=f"{type(error).__name__}: {error}",
            worker_pid=None,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
