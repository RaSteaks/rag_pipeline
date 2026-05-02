function statusClass(v) {
  if (v === "online") return "ok";
  if (v === "disabled" || v === "unknown") return "muted";
  return "off";
}

function fmtNum(v) {
  return Number.isFinite(v) ? v.toLocaleString() : "—";
}

function fmtDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";

  const totalMinutes = Math.max(0, Math.floor(seconds / 60));
  const days = Math.floor(totalMinutes / 1440);
  const hours = Math.floor((totalMinutes % 1440) / 60);
  const minutes = totalMinutes % 60;

  return `${days} Day ${hours} hour ${minutes} min`;
}

function toast(msg, type = "info", duration = 3000) {
  const c = document.getElementById("toastContainer");
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

function card(title, value, cls = "") {
  const hasIndicator = ["ok", "off", "muted"].includes(cls);
  return `
    <div class="card ${cls}">
      ${hasIndicator ? `<div class="card-indicator ${cls}"></div>` : ""}
      <div class="card-label">${title}</div>
      <div class="card-value ${cls}">${value}</div>
    </div>`;
}

async function callApi(path, method = "GET", body = null) {
  const init = { method, headers: { "Content-Type": "application/json" } };
  if (body) init.body = JSON.stringify(body);
  const r = await fetch(path, init);
  return r.json();
}

function updateSourceTable(sources) {
  const tbody = document.querySelector("#sourcesTable tbody");
  tbody.innerHTML = "";
  for (const s of sources) {
    const tr = document.createElement("tr");
    const types = (s.file_types || [])
      .map(t => `<code style="background:var(--surface-2);padding:1px 6px;border-radius:4px;font-size:11px;font-family:monospace">${t}</code>`)
      .join(" ");
    const yes = `<span style="color:var(--green);font-weight:600">✓</span>`;
    const no  = `<span style="color:var(--text-dim)">—</span>`;
    tr.innerHTML = `
      <td style="font-weight:500;color:var(--text)">${s.name}</td>
      <td style="font-family:monospace;font-size:12px">${s.path}</td>
      <td>${s.enabled ? yes : no}</td>
      <td>${s.recursive ? yes : no}</td>
      <td>${s.weight}</td>
      <td>${types}</td>`;
    tbody.appendChild(tr);
  }

  const select = document.getElementById("sourceName");
  const enabled = sources.filter((s) => s.enabled);
  select.innerHTML = enabled.map((s) => `<option value="${s.name}">${s.name}</option>`).join("");
}

function updateCards(status) {
  document.getElementById("cards").innerHTML =
    card("API 状态",     status.api.status,                        statusClass(status.api.status)) +
    card("运行时长",     fmtDuration(status.api.uptime_seconds)) +
    card("Embedding",    status.embedding.status,                  statusClass(status.embedding.status)) +
    card("Reranker",     status.reranker.status,                   statusClass(status.reranker.status)) +
    card("Chroma Chunks",fmtNum(status.indexes.chroma_chunks)) +
    card("已索引文件",   fmtNum(status.indexes.indexed_files)) +
    card("BM25",  status.indexes.bm25_ready ? "ready" : "未就绪", status.indexes.bm25_ready ? "ok" : "off") +
    card("文件监听",     status.watcher.running ? "running" : "stopped", status.watcher.running ? "ok" : "off") +
    card("关闭请求",     status.shutdown?.requested ? "pending" : "idle", status.shutdown?.requested ? "muted" : "");

  const dot = document.getElementById("globalStatus");
  const txt = document.getElementById("globalStatusText");
  if (status.api.status === "online") {
    dot.className = "status-dot online";
    txt.textContent = "服务正常";
  } else {
    dot.className = "status-dot error";
    txt.textContent = "服务异常";
  }
}

function updateResultBox(data) {
  document.getElementById("resultBox").textContent = JSON.stringify(data, null, 2);
}

function updateConfigSourcePathTable(sources) {
  const tbody = document.querySelector("#sourcePathTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const s of sources || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-weight:500;color:var(--text)">${s.name || ""}</td>
      <td><input class="source-path-input" data-source-name="${s.name || ""}" value="${(s.path || "").replace(/"/g, "&quot;")}" /></td>
    `;
    tbody.appendChild(tr);
  }
}

function collectSourcePathUpdates() {
  return Array.from(document.querySelectorAll(".source-path-input")).map((el) => ({
    name: el.getAttribute("data-source-name"),
    path: el.value.trim(),
  }));
}

const LOG_COLORS = {
  DEBUG:    "#475569",
  INFO:     "#60a5fa",
  WARNING:  "#fbbf24",
  ERROR:    "#f87171",
  CRITICAL: "#fb7185",
};

function formatLogLine(entry) {
  const t = new Date(entry.ts * 1000).toLocaleTimeString();
  const lvl = entry.level.toUpperCase();
  const color = LOG_COLORS[lvl] || "#94a3b8";
  return `<span style="color:#3d4f6a">[${entry.id}] ${t} </span><span style="color:${color};font-weight:600">${lvl.padEnd(8, " ")}</span><span style="color:#334155"> ${entry.logger} | </span>${entry.message}`;
}

function updateLogBox(items) {
  const box = document.getElementById("logBox");
  box.innerHTML = items.map(formatLogLine).join("\n");
  if (document.getElementById("autoScrollLogs")?.checked) {
    box.scrollTop = box.scrollHeight;
  }
}

async function refreshStatus() {
  try {
    const status = await callApi("/status");
    updateCards(status);
    updateSourceTable(status.config.sources || []);
    if (status.last_sync?.result) updateResultBox(status.last_sync.result);
  } catch (e) {
    updateResultBox({ error: String(e) });
    document.getElementById("globalStatus").className = "status-dot error";
    document.getElementById("globalStatusText").textContent = "连接失败";
  }
}

async function loadConfigEditor() {
  try {
    const data = await callApi("/config/editor");
    if (data.parse_error) {
      toast("配置解析异常，请检查 YAML 原文", "error", 4500);
    }
    const cfg = data.config || {};
    document.getElementById("embeddingModelInput").value = cfg.embedding?.model || "";
    document.getElementById("rerankerModelInput").value = cfg.reranker?.model || "";
    updateConfigSourcePathTable(cfg.knowledge_sources || []);
    document.getElementById("configRawEditor").value = data.raw_yaml || "";

    const hint = document.getElementById("configFileHint");
    if (hint) {
      hint.textContent = data.exists
        ? `配置文件: ${data.path}`
        : `配置文件不存在，将在保存时创建: ${data.path}`;
    }
  } catch (e) {
    toast("读取配置失败: " + String(e), "error", 4200);
  }
}

async function saveQuickConfig() {
  const embedding_model = document.getElementById("embeddingModelInput").value.trim();
  const reranker_model = document.getElementById("rerankerModelInput").value.trim();
  const knowledge_sources = collectSourcePathUpdates();

  const payload = {
    embedding_model: embedding_model || null,
    reranker_model: reranker_model || null,
    knowledge_sources,
  };

  toast("正在保存快速配置…", "info");
  try {
    const res = await callApi("/config/save-quick", "POST", payload);
    if (res.status !== "ok") {
      toast("保存失败: " + (res.message || "未知错误"), "error", 4500);
      updateResultBox(res);
      return;
    }
    updateResultBox(res);
    toast("配置已保存并重载", "success");
    if (res.missing_sources?.length) {
      toast("未匹配知识源: " + res.missing_sources.join(", "), "info", 4200);
    }
    await refreshStatus();
    await refreshLogs();
    await loadConfigEditor();
  } catch (e) {
    toast("保存失败: " + String(e), "error", 4500);
  }
}

async function saveRawConfig() {
  const yaml_text = document.getElementById("configRawEditor").value;
  if (!yaml_text.trim()) {
    toast("配置原文为空，已取消", "error", 3000);
    return;
  }
  toast("正在保存配置原文…", "info");
  try {
    const res = await callApi("/config/save-text", "POST", { yaml_text, create_if_missing: true });
    updateResultBox(res);
    if (res.status !== "ok") {
      toast("保存失败: " + (res.message || "未知错误"), "error", 4500);
      return;
    }
    toast("配置原文保存成功，服务已重载", "success");
    await refreshStatus();
    await refreshLogs();
    await loadConfigEditor();
  } catch (e) {
    toast("保存失败: " + String(e), "error", 4500);
  }
}

async function refreshLogs(force = false) {
  try {
    const level = document.getElementById("logLevel")?.value || "INFO";
    const logs = await callApi(`/logs?limit=400&min_level=${encodeURIComponent(level)}`);
    updateLogBox(logs.items || []);
  } catch (e) {
    if (force) {
      updateLogBox([{ id: 0, ts: Date.now() / 1000, level: "ERROR", logger: "dashboard", message: String(e) }]);
    }
  }
}

async function runSync(rebuild) {
  toast(rebuild ? "全量重建启动中…" : "增量同步启动中…", "info");
  updateResultBox({ status: "running", rebuild });
  try {
    const data = await callApi("/sync", "POST", { rebuild });
    updateResultBox(data);
    toast(data.status === "ok" ? "同步完成" : "同步已完成（请检查结果）", data.status === "ok" ? "success" : "info");
    await refreshStatus();
    await refreshLogs();
  } catch (e) {
    toast("同步失败: " + String(e), "error");
  }
}

async function rebuildSource() {
  const sourceName = document.getElementById("sourceName").value;
  if (!sourceName) return;
  toast(`正在重建 ${sourceName}…`, "info");
  updateResultBox({ status: "running", source_name: sourceName, rebuild: true });
  try {
    const data = await callApi("/sync", "POST", { source_name: sourceName, rebuild: true });
    updateResultBox(data);
    toast(`${sourceName} 重建完成`, "success");
    await refreshStatus();
    await refreshLogs();
  } catch (e) {
    toast("重建失败: " + String(e), "error");
  }
}

async function reloadConfig() {
  toast("正在重载配置…", "info");
  updateResultBox({ status: "running", action: "reload-config" });
  try {
    const data = await callApi("/reload-config", "POST");
    updateResultBox(data);
    toast("配置已重载", "success");
    await refreshStatus();
    await refreshLogs();
  } catch (e) {
    toast("配置重载失败: " + String(e), "error");
  }
}

function openShutdownDialog() {
  const modal = document.getElementById("shutdownModal");
  modal.hidden = false;
  document.getElementById("shutdownForce").checked = false;
  document.getElementById("shutdownWait").checked = true;
  document.getElementById("shutdownTimeout").value = "300";
  document.getElementById("confirmShutdownBtn").disabled = false;
  document.getElementById("shutdownStateNote").textContent = "正在读取当前索引状态…";
  refreshShutdownState();
}

function closeShutdownDialog() {
  document.getElementById("shutdownModal").hidden = true;
}

function formatShutdownState(status) {
  const sync = status.sync || {};
  const image = status.image_indexing || {};
  const rows = [
    `sync: ${sync.running ? sync.mode || "running" : "idle"}`,
    `image_jobs: pending=${image.pending_jobs || 0}, active=${image.active_jobs || 0}`,
  ];
  if (status.shutdown?.requested) rows.push("shutdown: requested");
  return rows.join("\n");
}

async function refreshShutdownState() {
  try {
    const status = await callApi("/status");
    document.getElementById("shutdownStateNote").textContent = formatShutdownState(status);
  } catch (e) {
    document.getElementById("shutdownStateNote").textContent = "status: unavailable";
  }
}

async function shutdownService() {
  const btn = document.getElementById("confirmShutdownBtn");
  const wait_for_indexing = document.getElementById("shutdownWait").checked;
  const force = document.getElementById("shutdownForce").checked;
  const timeout_seconds = Math.max(0, Math.min(3600, parseInt(document.getElementById("shutdownTimeout").value, 10) || 0));

  btn.disabled = true;
  toast("正在提交关闭请求…", "info", 2500);
  updateResultBox({ status: "running", action: "shutdown", wait_for_indexing, timeout_seconds, force });

  try {
    const res = await callApi("/shutdown", "POST", {
      wait_for_indexing,
      timeout_seconds,
      force,
      reason: "dashboard",
    });
    updateResultBox(res);
    document.getElementById("shutdownStateNote").textContent = JSON.stringify(res.indexing || {}, null, 2);

    if (res.status === "shutting_down") {
      toast("服务正在关闭", "success", 4000);
      setTimeout(refreshStatus, 1200);
      return;
    }

    toast(res.message || "关闭请求未执行", "error", 4500);
    btn.disabled = false;
  } catch (e) {
    toast("关闭请求失败: " + String(e), "error", 4500);
    updateResultBox({ error: String(e), action: "shutdown" });
    btn.disabled = false;
  }
}

function renderResults(data) {
  const box = document.getElementById("searchResults");
  const meta = document.getElementById("searchMeta");

  if (!data.results || data.results.length === 0) {
    box.innerHTML = `
      <div class="search-empty">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        未找到相关结果
      </div>`;
    meta.textContent = "";
    return;
  }

  meta.textContent = `共 ${data.count} 条结果`;

  box.innerHTML = data.results.map((r, i) => {
    const reranked = r.rerank_score != null;
    const scoreVal = reranked ? r.rerank_score : (r.score ?? 0);
    const scoreStr = typeof scoreVal === "number" ? scoreVal.toFixed(4) : "—";
    const srcType = reranked ? "reranked" : (r.source || "vector");
    const srcLabel = reranked ? "reranked" : srcType;
    const filePath = r.meta?.source || r.source_file || "";
    const sourceName = r.source_name || r.meta?.source_name || "";

    const debugHtml = data.query && r.meta ? `
      <div class="result-debug">${JSON.stringify(r.meta, null, 2)}</div>` : "";

    return `
      <div class="result-item">
        <div class="result-header">
          <div class="result-rank">${i + 1}</div>
          <span class="result-chip ${srcLabel}">${srcLabel}</span>
          <span class="result-chip score">score: ${scoreStr}</span>
          ${sourceName ? `<span class="result-chip score">${sourceName}</span>` : ""}
          ${filePath ? `<div class="result-path" title="${filePath}">${filePath}</div>` : ""}
        </div>
        <div class="result-text">${r.text || ""}</div>
        ${debugHtml}
      </div>`;
  }).join("");
}

async function runSearch() {
  const query = document.getElementById("searchQuery").value.trim();
  if (!query) return;

  const top_k = parseInt(document.getElementById("searchTopK").value, 10);
  const debug = document.getElementById("searchDebug").checked;
  const btn = document.getElementById("searchBtn");
  const box = document.getElementById("searchResults");
  const meta = document.getElementById("searchMeta");

  btn.disabled = true;
  meta.textContent = "搜索中…";
  box.innerHTML = `<div class="search-empty"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>正在检索…</div>`;

  try {
    const data = await callApi("/search", "POST", { query, top_k, debug });
    if (data.error) {
      toast("搜索失败: " + data.error, "error");
      box.innerHTML = `<div class="search-empty">搜索出错: ${data.error}</div>`;
      meta.textContent = "";
    } else {
      renderResults(data);
    }
  } catch (e) {
    toast("搜索请求失败: " + String(e), "error");
    box.innerHTML = `<div class="search-empty">请求失败: ${String(e)}</div>`;
    meta.textContent = "";
  } finally {
    btn.disabled = false;
  }
}

// Init
refreshStatus();
refreshLogs();
loadConfigEditor();
setInterval(refreshStatus, 8000);
setInterval(refreshLogs, 5000);

document.getElementById("logLevel")?.addEventListener("change", () => refreshLogs(true));
document.getElementById("shutdownModal")?.addEventListener("click", (event) => {
  if (event.target.id === "shutdownModal") closeShutdownDialog();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !document.getElementById("shutdownModal")?.hidden) {
    closeShutdownDialog();
  }
});

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    item.classList.add("active");
  });
});
