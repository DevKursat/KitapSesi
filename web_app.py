#!/usr/bin/env python3
"""Local web control panel for resumable audiobook dubbing jobs."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from yt_dlp import YoutubeDL


ROOT = Path(__file__).resolve().parent
JOBS_ROOT = ROOT / "jobs"
WEB_ROOT = ROOT / "web"
JOBS_ROOT.mkdir(exist_ok=True)

SUPPORTED_LANGUAGES = {
    "ar": "Arapça",
    "cs": "Çekçe",
    "de": "Almanca",
    "en": "İngilizce",
    "es": "İspanyolca",
    "fr": "Fransızca",
    "hi": "Hintçe",
    "hu": "Macarca",
    "it": "İtalyanca",
    "ja": "Japonca",
    "ko": "Korece",
    "nl": "Felemenkçe",
    "pl": "Lehçe",
    "pt": "Portekizce",
    "ru": "Rusça",
    "tr": "Türkçe",
    "zh": "Çince",
}

PUBLISH_SAFE_RIGHTS = {"owned", "licensed", "public_domain", "permission"}
MODEL_TIERS = {"local", "premium_cloud", "enterprise_private"}

@asynccontextmanager
async def lifespan(_: FastAPI):
    scheduler = asyncio.create_task(queue_scheduler())
    try:
        yield
    finally:
        scheduler.cancel()
        try:
            await scheduler
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Kitap Sesi", version="1.0.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/studio")
def studio() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/landing")
def landing() -> FileResponse:
    return FileResponse(WEB_ROOT / "landing.html")


@app.get("/landing.css")
def landing_css() -> FileResponse:
    return FileResponse(WEB_ROOT / "landing.css")


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "languages": SUPPORTED_LANGUAGES,
        "preflight": production_preflight(),
    }


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    jobs = []
    for manifest in JOBS_ROOT.glob("*/job.json"):
        try:
            jobs.append(enrich_job(read_json(manifest), manifest.parent))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(jobs, key=lambda job: job["created_at"], reverse=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job_dir = safe_job_dir(job_id)
    return enrich_job(read_json(job_dir / "job.json"), job_dir)


@app.post("/api/jobs")
async def create_job(
    url: str = Form(...),
    source_language: str = Form(...),
    target_languages: str = Form(...),
    voice_license: str = Form(...),
    content_rights: str = Form(...),
    rights_notes: str = Form(""),
    model_tier: str = Form("local"),
    visual_mode: str = Form("still_image"),
    subtitle_mode: str = Form("translated"),
    image: UploadFile | None = File(None),
) -> dict[str, Any]:
    if model_tier not in MODEL_TIERS:
        raise HTTPException(400, "Geçersiz model kalitesi seçimi.")
    if model_tier == "local":
        preflight = production_preflight()
        critical_failures = [
            check["label"]
            for check in preflight["checks"]
            if check["critical"] and not check["passed"]
        ]
        if critical_failures:
            raise HTTPException(
                503,
                "Sistem ön kontrolü başarısız: " + ", ".join(critical_failures),
            )
    targets = list(
        dict.fromkeys(
            part.strip().lower()
            for part in target_languages.split(",")
            if part.strip()
        )
    )
    if source_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, "Geçersiz kaynak dili.")
    if not targets or any(lang not in SUPPORTED_LANGUAGES for lang in targets):
        raise HTTPException(400, "En az bir geçerli hedef dil seçin.")
    has_uploaded_image = (
        image is not None
        and bool(image.filename)
        and bool(image.content_type)
        and image.content_type.startswith("image/")
    )
    if image is not None and image.filename and not has_uploaded_image:
        raise HTTPException(400, "Kapak dosyası bir görsel olmalıdır.")
    if voice_license not in {"commercial", "noncommercial"}:
        raise HTTPException(400, "Geçerli bir XTTS lisans seçimi yapın.")
    if visual_mode not in {"still_image", "source_video"}:
        raise HTTPException(400, "Geçersiz görsel modu.")
    if subtitle_mode not in {"translated", "off"}:
        raise HTTPException(400, "Geçersiz altyazı modu.")
    if content_rights not in PUBLISH_SAFE_RIGHTS:
        raise HTTPException(
            400,
            (
                "Yayın için hak durumu güvenli değil. Sadece kendi içeriğiniz, "
                "lisanslı/izinli içerik veya kamu malı eserler üretime alınır."
            ),
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True)
    if has_uploaded_image and image is not None:
        suffix = Path(image.filename or "cover.jpg").suffix.lower() or ".jpg"
        image_path = input_dir / f"cover{suffix}"
        with image_path.open("wb") as destination:
            shutil.copyfileobj(image.file, destination)
        cover_source = "uploaded"
    elif model_tier == "local":
        image_path = download_youtube_thumbnail(url.strip(), input_dir)
        cover_source = "youtube_thumbnail"
    else:
        image_path = input_dir / "cover_not_required.txt"
        image_path.write_text(
            "Premium/Enterprise talebi için kapak dosyası gerekmez.\n",
            encoding="utf-8",
        )
        cover_source = "not_required_for_quote"

    now = now_iso()
    manifest = {
        "id": job_id,
        "url": url.strip(),
        "image_path": str(image_path.resolve()),
        "cover_source": cover_source,
        "visual_mode": visual_mode,
        "subtitle_mode": subtitle_mode,
        "source_language": source_language,
        "target_languages": targets,
        "status": "queued" if model_tier == "local" else "quote_requested",
        "stage": "queued" if model_tier == "local" else "sales",
        "message": (
            "İş kuyruğa alındı."
            if model_tier == "local"
            else "Premium/Enterprise model entegrasyonu için teklif talebi kaydedildi."
        ),
        "progress": 0,
        "stage_completed": 0,
        "stage_total": 1,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "worker_pid": None,
        "whisper_model": "large-v3",
        "translation_model": "facebook/nllb-200-distilled-600M",
        "voice_license": voice_license,
        "voice_license_accepted_at": now,
        "model_tier": model_tier,
        "content_rights": content_rights,
        "rights_notes": rights_notes.strip()[:1000],
        "rights_accepted_at": now,
    }
    atomic_json(job_dir / "job.json", manifest)
    atomic_json(job_dir / "control.json", {"action": "run"})
    if model_tier == "local":
        start_next_job()
    return enrich_job(read_json(job_dir / "job.json"), job_dir)


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str) -> dict[str, Any]:
    job_dir = safe_job_dir(job_id)
    job = read_json(job_dir / "job.json")
    if job.get("status") == "completed":
        raise HTTPException(409, "Tamamlanan iş duraklatılamaz.")
    atomic_json(job_dir / "control.json", {"action": "pause"})
    pid = job.get("worker_pid")
    if process_alive(pid):
        update_manifest(
            job_dir,
            status="paused",
            message="İşlem anında duraklatıldı. Devam Et ile aynı yerden sürer.",
        )
        signal_process_group(pid, signal.SIGSTOP)
    else:
        update_manifest(
            job_dir,
            status="paused",
            worker_pid=None,
            message="Kuyruktaki iş duraklatıldı.",
        )
    return get_job(job_id)


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict[str, Any]:
    job_dir = safe_job_dir(job_id)
    job = read_json(job_dir / "job.json")
    if job.get("status") == "completed":
        raise HTTPException(409, "Tamamlanan iş yeniden başlatılamaz.")
    atomic_json(job_dir / "control.json", {"action": "run"})
    pid = job.get("worker_pid")
    if process_alive(pid):
        signal_process_group(pid, signal.SIGCONT)
        update_manifest(
            job_dir,
            status="running",
            message="İşlem durduğu noktadan devam ediyor.",
        )
    else:
        update_manifest(
            job_dir,
            status="queued",
            worker_pid=None,
            message="Devam etmek için üretim kuyruğuna alındı.",
        )
        start_next_job()
    return get_job(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    job_dir = safe_job_dir(job_id)
    job = read_json(job_dir / "job.json")
    if job.get("status") == "completed":
        raise HTTPException(409, "Tamamlanan iş iptal edilemez.")
    atomic_json(job_dir / "control.json", {"action": "cancel"})
    pid = job.get("worker_pid")
    if process_alive(pid):
        terminate_process_group(pid)
    update_manifest(
        job_dir,
        status="cancelled",
        worker_pid=None,
        message=(
            "İşlem iptal edildi. Tamamlanan checkpoint dosyaları korundu; "
            "Devam Et ile yeniden sürdürülebilir."
        ),
        cancelled_at=now_iso(),
    )
    start_next_job()
    return get_job(job_id)


@app.post("/api/jobs/{job_id}/license")
def set_job_license(
    job_id: str,
    voice_license: str = Form(...),
) -> dict[str, Any]:
    if voice_license not in {"commercial", "noncommercial"}:
        raise HTTPException(400, "Geçerli bir lisans seçimi yapın.")
    job_dir = safe_job_dir(job_id)
    update_manifest(
        job_dir,
        voice_license=voice_license,
        voice_license_accepted_at=now_iso(),
        message=(
            "XTTS ticari lisansı kaydedildi."
            if voice_license == "commercial"
            else "XTTS ticari olmayan CPML kabulü kaydedildi."
        ),
    )
    return get_job(job_id)


@app.post("/api/jobs/{job_id}/languages")
def add_job_languages(
    job_id: str,
    target_languages: str = Form(...),
) -> dict[str, Any]:
    job_dir = safe_job_dir(job_id)
    job = read_json(job_dir / "job.json")
    if process_alive(job.get("worker_pid")):
        raise HTTPException(
            409,
            "Yeni dil eklemek için çalışan işin tamamlanmasını veya "
            "durdurulmasını bekleyin.",
        )
    requested = [
        part.strip().lower()
        for part in target_languages.split(",")
        if part.strip()
    ]
    invalid = [
        language
        for language in requested
        if language not in SUPPORTED_LANGUAGES
    ]
    if invalid:
        raise HTTPException(
            400,
            "Desteklenmeyen dil: " + ", ".join(invalid),
        )
    existing = list(job.get("target_languages") or [])
    added = [
        language
        for language in requested
        if language not in existing
    ]
    if not added:
        raise HTTPException(409, "Seçilen diller projede zaten mevcut.")
    update_manifest(
        job_dir,
        target_languages=existing + added,
        status="queued",
        stage="queued",
        progress=48,
        worker_pid=None,
        completed_at=None,
        error=None,
        message=(
            "Yeni diller kuyruğa alındı. Kaynak, stem ve transkript "
            "checkpoint'leri yeniden kullanılacak."
        ),
    )
    atomic_json(job_dir / "control.json", {"action": "run"})
    start_next_job()
    return get_job(job_id)


@app.post("/api/jobs/{job_id}/rebuild-language")
def rebuild_job_language(
    job_id: str,
    target_language: str = Form(...),
) -> dict[str, Any]:
    job_dir = safe_job_dir(job_id)
    job = read_json(job_dir / "job.json")
    if process_alive(job.get("worker_pid")):
        raise HTTPException(
            409,
            "Bir dili yeniden üretmek için çalışan işi önce durdurun.",
        )
    language = target_language.strip().lower()
    if language not in (job.get("target_languages") or []):
        raise HTTPException(400, "Bu dil mevcut projenin hedeflerinde yok.")

    work = job_dir / "work"
    versioned_files = [
        *work.glob("proper_names_v*.json"),
        *work.glob(f"translation_v*_{language}.json"),
        *work.glob(f"translation_quality_v*_{language}.json"),
        *work.glob(f"subtitle_timeline_v*_{language}.json"),
        *work.glob(f"{language}_speech_v*.wav"),
        work / "language_versions" / f"{language}.json",
    ]
    for path in versioned_files:
        path.unlink(missing_ok=True)
    for parent in ("tts", "timeline", "video_chunks"):
        for version_dir in (work / parent).glob("v*"):
            shutil.rmtree(version_dir / language, ignore_errors=True)

    output_path = job.get("output_path")
    if output_path:
        output = Path(output_path)
        for name in (
            f"{language.upper()}_dubbed_audio.flac",
            f"{language.upper()}_youtube_audio.m4a",
            f"{language.upper()}_dubbed.mp4",
            f"{language.upper()}_captions.ass",
            "quality_report.json",
        ):
            (output / name).unlink(missing_ok=True)

    update_manifest(
        job_dir,
        status="queued",
        stage="queued",
        progress=45,
        worker_pid=None,
        completed_at=None,
        error=None,
        message=(
            f"{language.upper()} yeniden üretim kuyruğunda. Kaynak ses, "
            "stemler ve transkript checkpoint'leri korunuyor."
        ),
    )
    atomic_json(job_dir / "control.json", {"action": "run"})
    start_next_job()
    return get_job(job_id)


@app.delete("/api/jobs/{job_id}")
@app.post("/api/jobs/{job_id}/delete")
def delete_job(job_id: str) -> dict[str, str]:
    job_dir = safe_job_dir(job_id)
    job = read_json(job_dir / "job.json")
    pid = job.get("worker_pid")
    if process_alive(pid):
        terminate_process_group(pid)
    try:
        shutil.rmtree(job_dir)
    except OSError as error:
        raise HTTPException(
            500,
            f"İş klasörü tamamen silinemedi: {error}",
        ) from error
    start_next_job()
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/files/{relative_path:path}")
def download_file(job_id: str, relative_path: str) -> FileResponse:
    job_dir = safe_job_dir(job_id)
    output = (job_dir / "output").resolve()
    requested = (output / relative_path).resolve()
    if output not in requested.parents or not requested.is_file():
        raise HTTPException(404, "Dosya bulunamadı.")
    return FileResponse(requested, filename=requested.name)


def start_worker(job_dir: Path) -> None:
    job = read_json(job_dir / "job.json")
    if process_alive(job.get("worker_pid")):
        return
    log = (job_dir / "worker.log").open("ab", buffering=0)
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "resumable_pipeline.py"),
            "--job-dir",
            str(job_dir),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={
            **os.environ,
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    update_manifest(
        job_dir,
        status="running",
        worker_pid=process.pid,
        message="İşçi başlatıldı; checkpoint sistemi aktif.",
    )


def start_next_job() -> None:
    jobs: list[tuple[dict[str, Any], Path]] = []
    for manifest in JOBS_ROOT.glob("*/job.json"):
        try:
            jobs.append((read_json(manifest), manifest.parent))
        except (OSError, json.JSONDecodeError):
            continue
    active = [
        job
        for job, _ in jobs
        if job.get("status") in {"running", "paused"}
        and process_alive(job.get("worker_pid"))
    ]
    if active:
        return
    queued = sorted(
        (
            (job, job_dir)
            for job, job_dir in jobs
            if job.get("status") == "queued"
        ),
        key=lambda item: item[0]["created_at"],
    )
    if queued:
        start_worker(queued[0][1])


def download_youtube_thumbnail(url: str, input_dir: Path) -> Path:
    """Fetch the YouTube thumbnail when the user does not upload a cover."""
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as error:
        raise HTTPException(
            400,
            "Kapak yüklenmedi ve YouTube thumbnail alınamadı.",
        ) from error

    thumbnail = (info or {}).get("thumbnail")
    if not thumbnail:
        raise HTTPException(
            400,
            "Kapak yüklenmedi ve bu videoda kullanılabilir thumbnail bulunamadı.",
        )

    parsed = urllib.parse.urlparse(str(thumbnail))
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    target = input_dir / f"cover{suffix}"
    try:
        with urllib.request.urlopen(str(thumbnail), timeout=30) as response:
            target.write_bytes(response.read())
    except Exception as error:
        raise HTTPException(
            400,
            "YouTube thumbnail indirilemedi; lütfen kapak görseli yükleyin.",
        ) from error
    if not target.is_file() or target.stat().st_size < 512:
        raise HTTPException(
            400,
            "YouTube thumbnail geçersiz; lütfen kapak görseli yükleyin.",
        )
    return target


async def queue_scheduler() -> None:
    while True:
        start_next_job()
        await asyncio.sleep(2)


def enrich_job(job: dict[str, Any], job_dir: Path) -> dict[str, Any]:
    enriched = dict(job)
    pid = enriched.get("worker_pid")
    if enriched.get("status") == "running" and not process_alive(pid):
        enriched["status"] = "interrupted"
        enriched["message"] = "İşçi durmuş. Devam Et ile checkpoint'ten sürdürün."
    output = job_dir / "output"
    enriched["files"] = [
        {
            "name": str(path.relative_to(output)),
            "size": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    log = job_dir / "worker.log"
    enriched["log_tail"] = tail_text(log, 30) if log.exists() else ""
    if (
        enriched.get("status") == "running"
        and enriched.get("stage") == "separation"
        and log.exists()
    ):
        demucs = demucs_live_progress(log)
        if demucs:
            chunk_index = int(enriched.get("stage_completed") or 0)
            chunk_total = max(int(enriched.get("stage_total") or 1), 1)
            chunk_fraction = (
                demucs["pass_index"] + demucs["fraction"]
            ) / demucs["pass_total"]
            overall_fraction = (
                chunk_index + min(chunk_fraction, 0.999)
            ) / chunk_total
            enriched["progress"] = round(8 + 22 * overall_fraction, 2)
            enriched["message"] = (
                f"Ses ve fon ayrılıyor: parça {chunk_index + 1}/"
                f"{chunk_total}, Demucs analizi "
                f"{demucs['pass_index'] + 1}/{demucs['pass_total']} "
                f"(%{round(demucs['fraction'] * 100)})"
            )
    reports = list((job_dir / "output").glob("*/quality_report.json"))
    enriched["release_ready"] = (
        enriched.get("status") == "completed"
        and enriched.get("voice_license") == "commercial"
        and enriched.get("content_rights") in PUBLISH_SAFE_RIGHTS
        and len(reports) == 1
        and quality_report_passed(reports[0])
    )
    return enriched


def safe_job_dir(job_id: str) -> Path:
    if not job_id.isalnum():
        raise HTTPException(400, "Geçersiz iş kimliği.")
    job_dir = (JOBS_ROOT / job_id).resolve()
    if JOBS_ROOT.resolve() not in job_dir.parents or not (job_dir / "job.json").exists():
        raise HTTPException(404, "İş bulunamadı.")
    return job_dir


def process_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def signal_process_group(pid: Any, requested_signal: signal.Signals) -> None:
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        os.killpg(pid, requested_signal)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise HTTPException(
            500,
            "İşlem kontrol sinyali gönderilemedi.",
        ) from error


def terminate_process_group(pid: Any) -> None:
    if not isinstance(pid, int) or pid <= 0:
        return
    # A stopped process cannot handle SIGTERM until it is continued.
    signal_process_group(pid, signal.SIGCONT)
    signal_process_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + 2.0
    while process_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if process_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # A terminated child can remain as a zombie until its parent
            # reaps it. It is already stopped and cannot do more work.
            pass


def update_manifest(job_dir: Path, **changes: Any) -> None:
    manifest = job_dir / "job.json"
    data = read_json(manifest)
    data.update(changes)
    data["updated_at"] = now_iso()
    atomic_json(manifest, data)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def tail_text(path: Path, lines: int) -> str:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 24_000))
        text = stream.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def demucs_live_progress(path: Path) -> dict[str, float] | None:
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None
    values = [
        int(value)
        for value in re.findall(r"(\d{1,3})%\|", text)
    ]
    if not values:
        return None
    pass_index = 0
    previous = values[0]
    for value in values[1:]:
        if previous >= 95 and value <= 5:
            pass_index += 1
        previous = value
    pass_total = 8
    return {
        "pass_index": float(min(pass_index, pass_total - 1)),
        "pass_total": float(pass_total),
        "fraction": min(max(values[-1] / 100, 0.0), 1.0),
    }


def production_preflight() -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    checks = [
        {
            "id": "platform",
            "label": "Apple Silicon macOS",
            "passed": platform.system() == "Darwin"
            and platform.machine() == "arm64",
            "critical": True,
        },
        {
            "id": "mps",
            "label": "PyTorch MPS",
            "passed": torch.backends.mps.is_available(),
            "critical": True,
        },
        {
            "id": "ffmpeg",
            "label": "FFmpeg ve FFprobe",
            "passed": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
            "critical": True,
        },
        {
            "id": "disk",
            "label": "En az 20 GB boş disk",
            "passed": disk.free >= 20 * 1024**3,
            "critical": True,
            "detail": f"{disk.free / 1024**3:.1f} GB boş",
        },
        {
            "id": "release_test",
            "label": "Uçtan uca doğrulanmış çıktı",
            "passed": has_validated_output(),
            "critical": False,
        },
    ]
    return {
        "production_ready": all(check["passed"] for check in checks),
        "can_run": all(
            check["passed"] for check in checks if check["critical"]
        ),
        "checks": checks,
    }


def has_validated_output() -> bool:
    for report in JOBS_ROOT.glob("*/output/*/quality_report.json"):
        if quality_report_passed(report):
            return True
    return False


def quality_report_passed(path: Path) -> bool:
    try:
        report = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        report.get("status") == "validated"
        and report.get("content_rights") in PUBLISH_SAFE_RIGHTS
        and report.get("source_suitability") == "passed"
        and bool(report.get("languages"))
        and all(
            language.get("duration_validation") == "passed"
            for language in report["languages"]
        )
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
