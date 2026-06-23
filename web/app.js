const form = document.querySelector("#job-form");
const jobsList = document.querySelector("#jobs-list");
const template = document.querySelector("#job-template");
const submitButton = document.querySelector("#submit-button");
let languages = {};
let preflight = {};
const openLogs = new Set(loadOpenLogs());

function loadOpenLogs() {
  try {
    const value = JSON.parse(
      localStorage.getItem("kitap-sesi-open-logs") || "[]"
    );
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function saveOpenLogs() {
  localStorage.setItem(
    "kitap-sesi-open-logs",
    JSON.stringify([...openLogs])
  );
}

function rememberVisibleLogState() {
  for (const card of jobsList.querySelectorAll(".job-card[data-id]")) {
    const details = card.querySelector("details");
    if (!details) continue;
    if (details.open) openLogs.add(card.dataset.id);
    else openLogs.delete(card.dataset.id);
  }
  saveOpenLogs();
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "İstek başarısız.");
  }
  return response.json();
}

function statusLabel(status) {
  return ({
    queued: "Kuyrukta", running: "Çalışıyor", paused: "Duraklatıldı",
    failed: "Hata", interrupted: "Kesintiye uğradı", completed: "Tamamlandı",
    cancelled: "İptal edildi", quote_requested: "Teklif talebi",
  })[status] || status;
}

function humanSize(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit++; }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function renderJobs(jobs) {
  // The queue refreshes every three seconds. Capture the live DOM state before
  // replacing cards so an expanded technical log never collapses itself.
  rememberVisibleLogState();
  document.querySelector("#job-count").textContent = `${jobs.length} iş`;
  jobsList.innerHTML = "";
  if (!jobs.length) {
    jobsList.innerHTML = '<div class="empty">Henüz bir eser eklenmedi.</div>';
    return;
  }
  for (const job of jobs) {
    const node = template.content.cloneNode(true);
    const card = node.querySelector(".job-card");
    card.dataset.id = job.id;
    node.querySelector(".status").textContent = statusLabel(job.status);
    node.querySelector(".title").textContent = job.title || "YouTube eseri";
    node.querySelector(".languages").textContent =
      `${languages[job.source_language] || job.source_language} → ` +
      job.target_languages.map(code => languages[code] || code).join(", ");
    const release = node.querySelector(".release-state");
    release.textContent = job.status === "quote_requested"
      ? "PREMIUM/ENTERPRISE TEKLİF TALEBİ KAYDEDİLDİ"
      : job.release_ready
      ? "✓ YAYINA HAZIR — OTOMATİK KONTROLLER GEÇTİ"
      : "YAYIN ONAYI YOK";
    release.classList.toggle("ready", Boolean(job.release_ready));
    node.querySelector(".percent").textContent = `${Math.round(job.progress || 0)}%`;
    node.querySelector(".progress-fill").style.width = `${job.progress || 0}%`;
    node.querySelector(".stage").textContent = job.stage || "";
    node.querySelector(".message").textContent = job.message || "";
    node.querySelector(".error").textContent = job.error || "";
    const addLanguagePanel = node.querySelector(".add-language-panel");
    const addLanguageSelect = node.querySelector("[data-add-language]");
    const availableLanguages = Object.entries(languages).filter(
      ([code]) => !job.target_languages.includes(code)
    );
    addLanguagePanel.hidden =
      job.status === "running" || job.status === "quote_requested" ||
      availableLanguages.length === 0;
    addLanguageSelect.innerHTML = availableLanguages
      .map(([code, label]) => `<option value="${code}">${label}</option>`)
      .join("");
    node.querySelector("[data-add-language-save]").addEventListener(
      "click",
      () => addLanguage(job.id, addLanguageSelect.value)
    );
    const rebuildPanel = node.querySelector(".rebuild-language-panel");
    const rebuildSelect = node.querySelector("[data-rebuild-language]");
    rebuildPanel.hidden =
      job.status === "running" || job.status === "quote_requested" ||
      job.target_languages.length === 0;
    rebuildSelect.innerHTML = job.target_languages
      .map(code => `<option value="${code}">${languages[code] || code}</option>`)
      .join("");
    node.querySelector("[data-rebuild-language-save]").addEventListener(
      "click",
      () => rebuildLanguage(job.id, rebuildSelect.value)
    );
    const licensePanel = node.querySelector(".license-panel");
    const licenseSelect = node.querySelector("[data-license]");
    licenseSelect.value = job.voice_license || "";
    licensePanel.hidden = Boolean(job.voice_license) || job.status === "quote_requested";
    node.querySelector("[data-license-save]").addEventListener(
      "click",
      () => saveLicense(job.id, licenseSelect.value)
    );
    node.querySelector(".log").textContent = job.log_tail || "Günlük henüz oluşmadı.";
    const details = node.querySelector("details");
    if (openLogs.has(job.id)) details.setAttribute("open", "");
    details.addEventListener("toggle", () => {
      if (details.open) openLogs.add(job.id);
      else openLogs.delete(job.id);
      saveOpenLogs();
    });

    const pause = node.querySelector('[data-action="pause"]');
    const resume = node.querySelector('[data-action="resume"]');
    const cancel = node.querySelector('[data-action="cancel"]');
    pause.hidden = !["running", "queued"].includes(job.status);
    resume.hidden = !["paused", "failed", "interrupted", "cancelled"].includes(job.status);
    cancel.hidden = ["completed", "cancelled", "quote_requested"].includes(job.status);

    for (const button of node.querySelectorAll("[data-action]")) {
      button.addEventListener("click", () => controlJob(job.id, button.dataset.action));
    }
    node.querySelector("[data-delete]").addEventListener(
      "click",
      () => deleteJob(job)
    );
    const outputs = node.querySelector(".outputs");
    for (const file of job.files || []) {
      const link = document.createElement("a");
      link.href = `/api/jobs/${job.id}/files/${file.name.split("/").map(encodeURIComponent).join("/")}`;
      link.textContent = `${file.name.split("/").pop()} · ${humanSize(file.size)}`;
      outputs.append(link);
    }
    jobsList.append(node);
  }
}

async function deleteJob(job) {
  const title = job.title || "Bu iş";
  const confirmed = window.confirm(
    `${title} tamamen silinsin mi?\n\n` +
    "İş kaydı, indirilen kaynak, checkpoint'ler ve tüm çıktılar kalıcı olarak silinecek."
  );
  if (!confirmed) return;
  try {
    await api(`/api/jobs/${job.id}/delete`, { method: "POST" });
    openLogs.delete(job.id);
    saveOpenLogs();
    await refreshJobs();
  } catch (error) {
    alert(error.message);
  }
}

async function saveLicense(id, voiceLicense) {
  if (!voiceLicense) {
    alert("Önce XTTS lisans durumunu seçin.");
    return;
  }
  const commercial = voiceLicense === "commercial";
  const message = commercial
    ? "Coqui XTTS için satın alınmış ticari lisansınız olduğunu onaylıyor musunuz?"
    : "Ticari olmayan CPML koşullarını kabul ediyor musunuz? Bu seçim ticari yayın onayı vermez.";
  if (!window.confirm(message)) return;
  const data = new FormData();
  data.set("voice_license", voiceLicense);
  try {
    await api(`/api/jobs/${id}/license`, { method: "POST", body: data });
    await refreshJobs();
  } catch (error) {
    alert(error.message);
  }
}

async function addLanguage(id, language) {
  if (!language) return;
  const data = new FormData();
  data.set("target_languages", language);
  try {
    await api(`/api/jobs/${id}/languages`, {
      method: "POST",
      body: data,
    });
    await refreshJobs();
  } catch (error) {
    alert(error.message);
  }
}

async function rebuildLanguage(id, language) {
  if (!language) return;
  if (!confirm(`${languages[language] || language} çıktısı yeniden üretilecek. Devam edilsin mi?`)) {
    return;
  }
  const data = new FormData();
  data.set("target_language", language);
  try {
    await api(`/api/jobs/${id}/rebuild-language`, {
      method: "POST",
      body: data,
    });
    await refreshJobs();
  } catch (error) {
    alert(error.message);
  }
}

async function controlJob(id, action) {
  try {
    await api(`/api/jobs/${id}/${action}`, { method: "POST" });
    await refreshJobs();
  } catch (error) {
    alert(error.message);
  }
}

async function refreshJobs() {
  try { renderJobs(await api("/api/jobs")); } catch (error) { console.error(error); }
}

async function init() {
  const config = await api("/api/config");
  languages = config.languages;
  preflight = config.preflight;
  renderPreflight(preflight);
  const source = document.querySelector("#source-language");
  const targets = document.querySelector("#target-languages");
  for (const [code, label] of Object.entries(languages)) {
    source.add(new Option(label, code, code === "en", code === "en"));
    const chip = document.createElement("label");
    chip.className = "chip";
    chip.innerHTML = `<input type="checkbox" value="${code}"><span>${label}</span>`;
    targets.append(chip);
  }
  await refreshJobs();
  setInterval(refreshJobs, 3000);
}

function renderPreflight(state) {
  const badge = document.querySelector("#system-badge");
  badge.classList.toggle("not-ready", state.can_run && !state.production_ready);
  badge.classList.toggle("blocked", !state.can_run);
  badge.querySelector("b").textContent = state.production_ready
    ? "Production doğrulandı"
    : state.can_run
      ? "Smoke test gerekli"
      : "Sistem hazır değil";
  const container = document.querySelector("#preflight");
  container.innerHTML = "";
  for (const check of state.checks) {
    const item = document.createElement("span");
    item.className = `check ${check.passed ? "passed" : "failed"}`;
    item.textContent =
      `${check.passed ? "✓" : "×"} ${check.label}` +
      (check.detail ? ` · ${check.detail}` : "");
    container.append(item);
  }
  submitButton.disabled = !state.can_run;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const selected = [...document.querySelectorAll("#target-languages input:checked")]
    .map(input => input.value);
  if (!selected.length) { alert("En az bir dublaj dili seçin."); return; }
  const data = new FormData(form);
  data.set("target_languages", selected.join(","));
  submitButton.disabled = true;
  submitButton.textContent = "Başlatılıyor…";
  try {
    await api("/api/jobs", { method: "POST", body: data });
    form.reset();
    await refreshJobs();
  } catch (error) {
    alert(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Üretimi başlat";
  }
});

init();
