"""Merge profile JSONs and generate self-contained HTML report with Chart.js."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_profile_report(
    agent_json_path: Path | None = None,
    main_profiler: Any | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Merge agent + main profiler data and write an HTML report.

    Returns the path to the generated HTML file.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "profiler_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}

    if agent_json_path and agent_json_path.exists():
        data = json.loads(agent_json_path.read_text())

    if main_profiler is not None:
        main_data = main_profiler._build_output()
        data = _merge_profiles(data, main_data)

    if not data:
        data = {"summary": {}, "turns": [], "resource_samples": [], "errors": [],
                "mode_transitions": [], "coaching_events": [], "ipc_messages": [],
                "tool_calls": [], "compaction_cycles": []}

    _ensure_summary(data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"profile_{timestamp}.json"
    json_path.write_text(json.dumps(data, indent=2))

    html_path = output_dir / f"profile_report_{timestamp}.html"
    html = _build_html(data)
    html_path.write_text(html)

    return html_path


def _merge_profiles(agent: dict, main: dict) -> dict:
    """Merge main.py profiler data into agent data."""
    merged = dict(agent)

    for key in ("events", "mode_transitions", "errors", "coaching_events",
                "ipc_messages", "tool_calls", "compaction_cycles"):
        agent_list = agent.get(key, [])
        main_list = main.get(key, [])
        merged[key] = agent_list + main_list

    agent_samples = agent.get("resource_samples", [])
    main_samples = main.get("resource_samples", [])
    if main_samples and not agent_samples:
        merged["resource_samples"] = main_samples
    elif agent_samples:
        merged["resource_samples"] = agent_samples

    if not merged.get("session_start") and main.get("session_start"):
        merged["session_start"] = main["session_start"]
    if not merged.get("session_end") and main.get("session_end"):
        merged["session_end"] = main["session_end"]
    if not merged.get("duration_s") and main.get("duration_s"):
        merged["duration_s"] = main["duration_s"]
    if not merged.get("system") and main.get("system"):
        merged["system"] = main["system"]

    if not merged.get("turns"):
        merged["turns"] = main.get("turns", [])

    return merged


def _ensure_summary(data: dict) -> None:
    """Recompute summary from raw data if missing."""
    turns = data.get("turns", [])
    errors = data.get("errors", [])
    mode_transitions = data.get("mode_transitions", [])
    coaching = data.get("coaching_events", [])

    ttfts = [t["llm_ttft_s"] for t in turns if t.get("llm_ttft_s")]
    ttfas = [t["ttfa_s"] for t in turns if t.get("ttfa_s")]
    e2es = [t["e2e_s"] for t in turns if t.get("e2e_s")]

    data["summary"] = {
        "total_turns": len(turns),
        "total_errors": len(errors),
        "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
        "avg_ttfa_s": round(sum(ttfas) / len(ttfas), 3) if ttfas else None,
        "avg_e2e_s": round(sum(e2es) / len(e2es), 3) if e2es else None,
        "modes_visited": list(dict.fromkeys(
            e.get("to", "") for e in mode_transitions if e.get("to")
        )),
        "total_reps": sum(1 for e in coaching if e.get("event_type") == "rep_complete"),
        "total_faults": sum(1 for e in coaching if e.get("event_type") == "fault"),
        "duration_s": data.get("duration_s"),
    }


def _build_html(data: dict) -> str:
    """Build self-contained HTML report."""
    return _HTML_TEMPLATE.replace(
        "{{PROFILE_JSON}}", json.dumps(data, indent=2)
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nowva Session Profile Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f1117; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; padding: 24px; max-width: 1400px; margin: 0 auto; }
  h1 { color: #fff; margin-bottom: 8px; font-size: 26px; }
  h2 { color: #a0a8c0; margin: 36px 0 16px; font-size: 18px; border-bottom: 1px solid #2a2d3a; padding-bottom: 8px; }
  .meta { color: #707890; font-size: 13px; margin-bottom: 24px; }
  .summary { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
  .badge { padding: 10px 20px; border-radius: 10px; font-weight: 600; font-size: 14px; text-align: center; min-width: 120px; }
  .badge small { display: block; font-size: 11px; font-weight: 400; opacity: 0.7; margin-top: 2px; }
  .badge-blue { background: #1a2a3a; color: #60a5fa; }
  .badge-green { background: #1a3a2a; color: #4ade80; }
  .badge-orange { background: #3a2a1a; color: #fb923c; }
  .badge-red { background: #3a1a1a; color: #f87171; }
  .badge-purple { background: #2a1a3a; color: #a78bfa; }
  .badge-cyan { background: #1a3a3a; color: #22d3ee; }
  .chart-container { background: #1a1d2a; border-radius: 12px; padding: 20px; margin-bottom: 24px; }
  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
  .chart-row .chart-container { margin-bottom: 0; }
  canvas { max-height: 500px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
  th { background: #1a1d2a; padding: 10px 12px; text-align: left; font-size: 13px; color: #a0a8c0; }
  td { padding: 10px 12px; border-bottom: 1px solid #1a1d2a; font-size: 13px; }
  tr:hover { background: #1a1d2a; }
  .hidden { display: none; }
  .error-row { background: #2a1a1a !important; }
  .turn-table { font-size: 12px; }
  .turn-table td, .turn-table th { padding: 6px 10px; }
</style>
</head>
<body>
<h1>Nowva Live Session Profile</h1>
<div class="meta" id="meta-info"></div>
<div class="summary" id="summary-badges"></div>

<h2>Turn Waterfall — STT → LLM → TTS</h2>
<div class="chart-container"><canvas id="waterfallChart"></canvas></div>

<div class="chart-row">
  <div>
    <h2>TTFT + TTFA Distribution</h2>
    <div class="chart-container"><canvas id="histogramChart"></canvas></div>
  </div>
  <div>
    <h2>Turn Latency Trend</h2>
    <div class="chart-container"><canvas id="trendChart"></canvas></div>
  </div>
</div>

<h2>Resource Usage Over Time</h2>
<div class="chart-container"><canvas id="resourceChart"></canvas></div>

<div class="chart-row">
  <div>
    <h2>Mode Timeline</h2>
    <div class="chart-container"><canvas id="modeChart"></canvas></div>
  </div>
  <div>
    <h2>Mode Time Breakdown</h2>
    <div class="chart-container"><canvas id="modePieChart"></canvas></div>
  </div>
</div>

<div id="coaching-section" class="hidden">
  <h2>Coaching Events Timeline</h2>
  <div class="chart-container"><canvas id="coachingChart"></canvas></div>
</div>

<div id="ipc-section" class="hidden">
  <h2>IPC Messages</h2>
  <div class="chart-container"><canvas id="ipcChart"></canvas></div>
</div>

<div id="compaction-section" class="hidden">
  <h2>Compaction Stats</h2>
  <div class="chart-container"><canvas id="compactionChart"></canvas></div>
</div>

<h2>Turn Details</h2>
<table class="turn-table" id="turn-table">
  <thead><tr><th>#</th><th>Elapsed</th><th>Transcript</th><th>TTFT</th><th>TTFA</th><th>E2E</th><th>LLM Dur</th><th>TTS Dur</th><th>Tokens</th><th>Affect wait</th><th>Affect infer</th></tr></thead>
  <tbody></tbody>
</table>

<div id="error-section" class="hidden">
  <h2>Errors</h2>
  <table id="error-table">
    <thead><tr><th>Elapsed</th><th>Source</th><th>Error</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<script>
const D = {{PROFILE_JSON}};
const turns = D.turns || [];
const samples = D.resource_samples || [];
const modeTransitions = D.mode_transitions || [];
const errors = D.errors || [];
const coaching = D.coaching_events || [];
const ipcMsgs = D.ipc_messages || [];
const compaction = D.compaction_cycles || [];
const summary = D.summary || {};

// ---- Meta ----
document.getElementById('meta-info').innerHTML = [
  D.session_start ? `Started: <strong>${D.session_start}</strong>` : '',
  D.duration_s ? `Duration: <strong>${Math.round(D.duration_s)}s</strong>` : '',
  D.system?.cpu ? `CPU: <strong>${D.system.cpu}</strong>` : '',
  D.system?.git_commit ? `Commit: <strong>${D.system.git_commit}</strong>` : '',
].filter(Boolean).join(' &middot; ');

// ---- Summary Badges ----
const badges = document.getElementById('summary-badges');
function badge(val, label, cls) {
  if (val == null) return '';
  return `<span class="badge ${cls}">${val}<small>${label}</small></span>`;
}
badges.innerHTML = [
  badge(summary.total_turns, 'Turns', 'badge-blue'),
  badge(summary.avg_ttft_s != null ? summary.avg_ttft_s.toFixed(2) + 's' : null, 'Avg TTFT', 'badge-orange'),
  badge(summary.avg_ttfa_s != null ? summary.avg_ttfa_s.toFixed(2) + 's' : null, 'Avg TTFA', 'badge-red'),
  badge(summary.avg_e2e_s != null ? summary.avg_e2e_s.toFixed(2) + 's' : null, 'Avg E2E', 'badge-purple'),
  badge(summary.total_errors || 0, 'Errors', summary.total_errors > 0 ? 'badge-red' : 'badge-green'),
  badge(summary.total_reps || null, 'Reps', 'badge-cyan'),
  badge(summary.total_faults || null, 'Faults', 'badge-orange'),
  badge(Math.round(D.duration_s || 0) + 's', 'Duration', 'badge-blue'),
].filter(Boolean).join('');

// ---- 1. Turn Waterfall ----
if (turns.length > 0) {
  const labels = turns.map(t => `Turn ${t.turn_number}`);
  // Estimate STT duration as gap between speech end and LLM start
  const sttDur = turns.map(t => {
    if (t.ttfa_s && t.llm_ttft_s && t.tts_duration_s) {
      const stt = t.ttfa_s - t.llm_ttft_s - (t.tts_duration_s || 0);
      return Math.max(stt, 0);
    }
    return 0;
  });
  new Chart(document.getElementById('waterfallChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'STT (est.)', data: sttDur, backgroundColor: '#3b82f6' },
        { label: 'LLM TTFT', data: turns.map(t => t.llm_ttft_s || 0), backgroundColor: '#f59e0b' },
        { label: 'LLM Gen', data: turns.map(t => Math.max((t.llm_duration_s || 0) - (t.llm_ttft_s || 0), 0)), backgroundColor: '#10b981' },
        { label: 'TTS', data: turns.map(t => t.tts_duration_s || 0), backgroundColor: '#8b5cf6' },
      ]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: {
        x: { stacked: true, title: { display: true, text: 'Seconds', color: '#a0a8c0' }, ticks: { color: '#707890' }, grid: { color: '#1a1d2a' } },
        y: { stacked: true, ticks: { color: '#a0a8c0', font: { size: 11 } }, grid: { display: false } }
      },
      plugins: { legend: { labels: { color: '#a0a8c0' } } }
    }
  });
} else {
  document.getElementById('waterfallChart').parentElement.innerHTML = '<p style="color:#707890;text-align:center;padding:40px">No turn data collected</p>';
}

// ---- 2. TTFT + TTFA Histogram ----
function makeHistogram(values, binWidth) {
  if (!values.length) return { bins: [], counts: [] };
  const maxVal = Math.max(...values);
  const nBins = Math.ceil(maxVal / binWidth) + 1;
  const counts = new Array(nBins).fill(0);
  values.forEach(v => { counts[Math.min(Math.floor(v / binWidth), nBins - 1)]++; });
  const bins = Array.from({length: nBins}, (_, i) => (i * binWidth).toFixed(1));
  return { bins, counts };
}
const ttfts = turns.map(t => t.llm_ttft_s).filter(v => v != null);
const ttfas = turns.map(t => t.ttfa_s).filter(v => v != null);
if (ttfts.length > 0 || ttfas.length > 0) {
  const binW = 0.1;
  const hTTFT = makeHistogram(ttfts, binW);
  const hTTFA = makeHistogram(ttfas, binW);
  const maxBins = Math.max(hTTFT.bins.length, hTTFA.bins.length);
  const labels = Array.from({length: maxBins}, (_, i) => (i * binW).toFixed(1) + 's');
  const pad = (arr, len) => { while(arr.length < len) arr.push(0); return arr; };
  new Chart(document.getElementById('histogramChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'TTFT', data: pad(hTTFT.counts, maxBins), backgroundColor: 'rgba(245,158,11,0.6)', borderColor: '#f59e0b', borderWidth: 1 },
        { label: 'TTFA', data: pad(hTTFA.counts, maxBins), backgroundColor: 'rgba(248,113,113,0.6)', borderColor: '#f87171', borderWidth: 1 },
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: 'Latency (s)', color: '#a0a8c0' }, ticks: { color: '#707890' }, grid: { color: '#1a1d2a' } },
        y: { title: { display: true, text: 'Count', color: '#a0a8c0' }, ticks: { color: '#707890' }, grid: { color: '#1a1d2a' } }
      },
      plugins: { legend: { labels: { color: '#a0a8c0' } } }
    }
  });
} else {
  document.getElementById('histogramChart').parentElement.innerHTML = '<p style="color:#707890;text-align:center;padding:40px">No TTFT/TTFA data</p>';
}

// ---- 3. Turn Latency Trend ----
if (turns.length > 1) {
  const turnNums = turns.map(t => t.turn_number);
  new Chart(document.getElementById('trendChart'), {
    type: 'line',
    data: {
      labels: turnNums.map(n => `Turn ${n}`),
      datasets: [
        { label: 'TTFT (s)', data: turns.map(t => t.llm_ttft_s), borderColor: '#f59e0b', tension: 0.3, pointRadius: 4, spanGaps: true },
        { label: 'TTFA (s)', data: turns.map(t => t.ttfa_s), borderColor: '#f87171', tension: 0.3, pointRadius: 4, spanGaps: true },
        { label: 'E2E (s)', data: turns.map(t => t.e2e_s), borderColor: '#8b5cf6', tension: 0.3, pointRadius: 4, spanGaps: true },
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: '#707890' }, grid: { color: '#1a1d2a' } },
        y: { title: { display: true, text: 'Seconds', color: '#a0a8c0' }, ticks: { color: '#707890' }, grid: { color: '#1a1d2a' } }
      },
      plugins: {
        legend: { labels: { color: '#a0a8c0' } },
        annotation: undefined
      }
    }
  });
} else {
  document.getElementById('trendChart').parentElement.innerHTML = '<p style="color:#707890;text-align:center;padding:40px">Need 2+ turns for trend</p>';
}

// ---- 4. Resource Usage Over Time ----
if (samples.length > 1) {
  const elapsed = samples.map(s => s.elapsed_s);
  new Chart(document.getElementById('resourceChart'), {
    type: 'line',
    data: {
      labels: elapsed.map(e => Math.round(e) + 's'),
      datasets: [
        { label: 'CPU %', data: samples.map(s => s.cpu_percent), borderColor: '#f59e0b', yAxisID: 'y', tension: 0.3, pointRadius: 0 },
        { label: 'Memory (MB)', data: samples.map(s => s.memory_mb), borderColor: '#8b5cf6', yAxisID: 'y1', tension: 0.3, pointRadius: 0 },
        ...(samples.some(s => s.gpu_vram_mb != null) ? [{
          label: 'GPU VRAM (MB)', data: samples.map(s => s.gpu_vram_mb), borderColor: '#10b981', yAxisID: 'y1', tension: 0.3, pointRadius: 0
        }] : []),
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { ticks: { color: '#707890', maxTicksLimit: 20 }, grid: { color: '#1a1d2a' } },
        y: { type: 'linear', position: 'left', title: { display: true, text: 'CPU %', color: '#f59e0b' }, ticks: { color: '#f59e0b' }, grid: { color: '#1a1d2a' } },
        y1: { type: 'linear', position: 'right', title: { display: true, text: 'Memory (MB)', color: '#8b5cf6' }, ticks: { color: '#8b5cf6' }, grid: { drawOnChartArea: false } },
      },
      plugins: { legend: { labels: { color: '#a0a8c0' } } }
    }
  });
} else {
  document.getElementById('resourceChart').parentElement.innerHTML = '<p style="color:#707890;text-align:center;padding:40px">No resource data</p>';
}

// ---- 5. Mode Timeline (Gantt) ----
const modeColors = { main_menu: '#3b82f6', workout: '#10b981', onboarding: '#f59e0b', program_creation: '#8b5cf6', schedule: '#ec4899', calibration: '#06b6d4' };
if (modeTransitions.length > 0) {
  const totalDur = D.duration_s || 1;
  const segments = [];
  for (let i = 0; i < modeTransitions.length; i++) {
    const start = modeTransitions[i].elapsed_s;
    const end = i + 1 < modeTransitions.length ? modeTransitions[i + 1].elapsed_s : totalDur;
    const mode = modeTransitions[i].to || 'unknown';
    segments.push({ mode, start, end, duration: end - start });
  }
  new Chart(document.getElementById('modeChart'), {
    type: 'bar',
    data: {
      labels: ['Session'],
      datasets: segments.map(s => ({
        label: s.mode,
        data: [s.duration],
        backgroundColor: modeColors[s.mode] || '#707890',
      }))
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: {
        x: { stacked: true, title: { display: true, text: 'Seconds', color: '#a0a8c0' }, ticks: { color: '#707890' }, grid: { color: '#1a1d2a' } },
        y: { stacked: true, display: false }
      },
      plugins: { legend: { labels: { color: '#a0a8c0' } } }
    }
  });
  // Mode pie
  const modeTime = {};
  segments.forEach(s => { modeTime[s.mode] = (modeTime[s.mode] || 0) + s.duration; });
  new Chart(document.getElementById('modePieChart'), {
    type: 'doughnut',
    data: {
      labels: Object.keys(modeTime),
      datasets: [{ data: Object.values(modeTime).map(v => Math.round(v)), backgroundColor: Object.keys(modeTime).map(m => modeColors[m] || '#707890') }]
    },
    options: { responsive: true, plugins: { legend: { position: 'right', labels: { color: '#a0a8c0' } } } }
  });
} else {
  document.getElementById('modeChart').parentElement.innerHTML = '<p style="color:#707890;text-align:center;padding:40px">No mode transitions</p>';
  document.getElementById('modePieChart').parentElement.innerHTML = '<p style="color:#707890;text-align:center;padding:40px">No mode data</p>';
}

// ---- 6. Coaching Events Timeline ----
const coachingCues = coaching.filter(e => e.event_type === 'cue_dispatched' || e.event_type === 'fault' || e.event_type === 'rep_complete');
if (coachingCues.length > 0) {
  document.getElementById('coaching-section').classList.remove('hidden');
  const catMap = { fault: 0, rep_complete: 1, cue_dispatched: 2 };
  const catLabels = ['Fault', 'Rep Complete', 'Cue Dispatched'];
  const catColors = ['#f87171', '#4ade80', '#60a5fa'];
  new Chart(document.getElementById('coachingChart'), {
    type: 'scatter',
    data: {
      datasets: catLabels.map((label, ci) => ({
        label,
        data: coachingCues.filter(e => (catMap[e.event_type] ?? 2) === ci).map(e => ({ x: e.elapsed_s, y: ci })),
        backgroundColor: catColors[ci],
        pointRadius: 6,
      }))
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: 'Elapsed (s)', color: '#a0a8c0' }, ticks: { color: '#707890' }, grid: { color: '#1a1d2a' } },
        y: { ticks: { callback: (v) => catLabels[v] || '', color: '#a0a8c0', stepSize: 1 }, grid: { color: '#1a1d2a' }, min: -0.5, max: 2.5 }
      },
      plugins: { legend: { labels: { color: '#a0a8c0' } }, tooltip: {
        callbacks: { label: (ctx) => { const e = coachingCues.filter(e2 => e2.elapsed_s === ctx.parsed.x)[0]; return e ? `${e.event_type}: ${e.cue_key || ''} @ ${e.elapsed_s}s` : ''; } }
      }}
    }
  });
}

// ---- 7. IPC Messages ----
if (ipcMsgs.length > 0) {
  document.getElementById('ipc-section').classList.remove('hidden');
  const typeCounts = {};
  ipcMsgs.forEach(m => { typeCounts[m.event_type] = (typeCounts[m.event_type] || 0) + 1; });
  new Chart(document.getElementById('ipcChart'), {
    type: 'bar',
    data: {
      labels: Object.keys(typeCounts),
      datasets: [{ label: 'Count', data: Object.values(typeCounts), backgroundColor: '#3b82f6' }]
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: '#a0a8c0' }, grid: { color: '#1a1d2a' } },
        y: { title: { display: true, text: 'Count', color: '#a0a8c0' }, ticks: { color: '#707890' }, grid: { color: '#1a1d2a' } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// ---- 8. Compaction Stats ----
if (compaction.length > 0) {
  document.getElementById('compaction-section').classList.remove('hidden');
  new Chart(document.getElementById('compactionChart'), {
    type: 'line',
    data: {
      labels: compaction.map(c => Math.round(c.elapsed_s) + 's'),
      datasets: [
        { label: 'HOT chars', data: compaction.map(c => c.hot_chars), borderColor: '#f87171', tension: 0.3, pointRadius: 3 },
        { label: 'WARM chars', data: compaction.map(c => c.warm_chars), borderColor: '#fbbf24', tension: 0.3, pointRadius: 3 },
        { label: 'COLD chars', data: compaction.map(c => c.cold_chars), borderColor: '#60a5fa', tension: 0.3, pointRadius: 3 },
        { label: 'Input Tokens', data: compaction.map(c => c.input_tokens), borderColor: '#a78bfa', borderDash: [5,5], tension: 0.3, pointRadius: 2, yAxisID: 'y1' },
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: '#707890' }, grid: { color: '#1a1d2a' } },
        y: { type: 'linear', position: 'left', title: { display: true, text: 'Characters', color: '#a0a8c0' }, ticks: { color: '#707890' }, grid: { color: '#1a1d2a' } },
        y1: { type: 'linear', position: 'right', title: { display: true, text: 'Tokens', color: '#a78bfa' }, ticks: { color: '#a78bfa' }, grid: { drawOnChartArea: false } },
      },
      plugins: { legend: { labels: { color: '#a0a8c0' } } }
    }
  });
}

// ---- 9. Turn Details Table ----
const tbody = document.querySelector('#turn-table tbody');
turns.forEach(t => {
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${t.turn_number}</td>
    <td>${t.start_s != null ? t.start_s.toFixed(1) + 's' : '-'}</td>
    <td title="${(t.transcript || '').replace(/"/g, '&quot;')}">${(t.transcript || '-').substring(0, 60)}${(t.transcript || '').length > 60 ? '...' : ''}</td>
    <td>${t.llm_ttft_s != null ? t.llm_ttft_s.toFixed(3) + 's' : '-'}</td>
    <td style="color:${t.ttfa_s > 2.5 ? '#f87171' : t.ttfa_s > 1.5 ? '#fbbf24' : '#4ade80'}">${t.ttfa_s != null ? t.ttfa_s.toFixed(3) + 's' : '-'}</td>
    <td>${t.e2e_s != null ? t.e2e_s.toFixed(3) + 's' : '-'}</td>
    <td>${t.llm_duration_s != null ? t.llm_duration_s.toFixed(2) + 's' : '-'}</td>
    <td>${t.tts_duration_s != null ? t.tts_duration_s.toFixed(2) + 's' : '-'}</td>
    <td>${t.llm_input_tokens != null ? t.llm_input_tokens + '→' + (t.llm_output_tokens || '?') : '-'}</td>
    <td title="fresh = state came from this turn">${t.affect_wait_ms != null ? t.affect_wait_ms.toFixed(1) + 'ms' + (t.affect_fresh ? ' ✓' : ' ·') : '-'}</td>
    <td>${t.affect_infer_ms != null ? t.affect_infer_ms.toFixed(0) + 'ms' : '-'}</td>
  `;
  tbody.appendChild(row);
});

// ---- 10. Error Log ----
if (errors.length > 0) {
  document.getElementById('error-section').classList.remove('hidden');
  const etbody = document.querySelector('#error-table tbody');
  errors.forEach(e => {
    const row = document.createElement('tr');
    row.className = 'error-row';
    row.innerHTML = `
      <td>${e.elapsed_s != null ? e.elapsed_s.toFixed(1) + 's' : '-'}</td>
      <td>${e.source || '-'}</td>
      <td>${(e.error || '-').substring(0, 200)}</td>
    `;
    etbody.appendChild(row);
  });
}
</script>
</body>
</html>"""
