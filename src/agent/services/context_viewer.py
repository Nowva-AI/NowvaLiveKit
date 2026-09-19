"""
Context Viewer — Real-time debug dashboard for the voice agent's LLM context.

Runs a lightweight aiohttp server on port 8899 that shows:
- System prompt / instructions
- Full chat_ctx items (role, content, timestamps)
- Compaction tiers (HOT / WARM / COLD)
- Session stats

The HTML page auto-refreshes via polling every 1.5s.
Open http://localhost:8899 in a browser, or curl it from a second terminal.
"""

import html
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)

_PORT = 8899


class ContextViewer:
    """Serves a live HTML view of the agent's current LLM context."""

    def __init__(self, session, compaction_service=None, state=None):
        """
        Args:
            session: The AgentSession instance (uses session.current_agent
                     so it automatically tracks agent handoffs).
            compaction_service: Optional CompactionService for tier display.
            state: Optional AgentState for mode display.
        """
        self._session = session
        self._compaction = compaction_service
        self._state = state
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._app = web.Application()
        self._app.router.add_get("/", self._handle_html)
        self._app.router.add_get("/api/context", self._handle_json)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", _PORT)
        await site.start()
        logger.info(f"[CONTEXT VIEWER] Running at http://localhost:{_PORT}")

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            logger.info("[CONTEXT VIEWER] Stopped")

    @property
    def _agent(self):
        """Always returns the live current agent (tracks handoffs automatically)."""
        return self._session.current_agent

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict:
        """Build a JSON-serializable snapshot of the current context."""
        now = datetime.now(timezone.utc).isoformat()
        agent = self._agent
        mode = "unknown"
        if self._state:
            try:
                mode = self._state.get_mode()
            except Exception:
                pass

        # Agent instructions (system prompt)
        instructions = ""
        try:
            instructions = (agent.instructions or "") if agent else ""
        except Exception:
            pass

        # Chat context items
        items = []
        try:
            ctx = agent.chat_ctx if agent else None
            if ctx is None:
                raise ValueError("No active agent")
            for item in ctx.items:
                role = getattr(item, "role", "unknown")
                text = getattr(item, "text_content", None)
                if callable(text):
                    text = text()
                content_parts = getattr(item, "content", [])
                # Summarize non-text content
                if not text and content_parts:
                    parts_summary = []
                    for p in content_parts:
                        if isinstance(p, str):
                            parts_summary.append(p)
                        else:
                            parts_summary.append(f"[{type(p).__name__}]")
                    text = " ".join(parts_summary)
                created = ""
                try:
                    created = getattr(item, "created_at", "")
                    if created and hasattr(created, "isoformat"):
                        created = created.isoformat()
                    else:
                        created = str(created) if created else ""
                except Exception:
                    pass
                items.append({
                    "role": str(role),
                    "content": text or "[empty]",
                    "created_at": created,
                    "type": getattr(item, "type", "message"),
                    "id": getattr(item, "id", ""),
                })
        except Exception as e:
            items.append({"role": "error", "content": f"Failed to read chat_ctx: {e}", "created_at": "", "type": "error", "id": ""})

        # Compaction tiers
        compaction = {"hot": "", "warm": "", "cold": "", "summary": "", "stats": {}}
        if self._compaction:
            try:
                tiers = self._compaction.get_tier_snapshot()
                compaction["hot"] = tiers["hot"]
                compaction["warm"] = tiers["warm"] or ""
                compaction["cold"] = tiers["cold"] or ""
                compaction["summary"] = self._compaction.get_summary()
                compaction["stats"] = self._compaction.get_stats()
            except Exception:
                compaction["error"] = "Failed to retrieve compaction data"

        uptime = time.monotonic() - self._start_time

        # Speech affect / effort perception (optional)
        affect: dict = {}
        try:
            affect_service = getattr(self._session.userdata, "affect_service", None)
            if affect_service is not None:
                affect = affect_service.snapshot_dict()
        except Exception:
            affect = {"error": "Failed to read affect state"}

        return {
            "timestamp": now,
            "mode": mode,
            "agent_class": type(agent).__name__ if agent else "None",
            "uptime_seconds": round(uptime, 1),
            "instructions_length": len(instructions),
            "instructions": instructions,
            "chat_items": items,
            "chat_item_count": len(items),
            "compaction": compaction,
            "affect": affect,
        }

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_json(self, request: web.Request) -> web.Response:
        snapshot = self._snapshot()
        return web.json_response(snapshot)

    async def _handle_html(self, request: web.Request) -> web.Response:
        snapshot = self._snapshot()
        html_content = self._render_html(snapshot)
        return web.Response(text=html_content, content_type="text/html")

    # ------------------------------------------------------------------
    # HTML renderer
    # ------------------------------------------------------------------

    def _render_html(self, snap: dict) -> str:
        esc = html.escape

        # Athlete state badge (speech affect / effort perception)
        affect = snap.get("affect") or {}
        affect_state = affect.get("state") or {}
        if not affect:
            affect_badge = "off"
        elif not affect.get("enabled"):
            affect_badge = "disabled"
        else:
            affect_badge = (
                f"{affect_state.get('effort', '?')} / {affect_state.get('affect', '?')}"
                f"{'' if affect_state.get('confident') else ' (unsure)'}"
            )

        # Chat items
        chat_rows = []
        for i, item in enumerate(snap["chat_items"]):
            role = item["role"]
            role_class = role if role in ("user", "assistant", "system", "tool", "developer") else "other"
            content = esc(item["content"][:3000])
            # Preserve newlines
            content = content.replace("\n", "<br>")
            created = esc(item.get("created_at", ""))
            chat_rows.append(
                f'<div class="msg {role_class}">'
                f'<div class="msg-header"><span class="role-badge {role_class}">{esc(role)}</span>'
                f'<span class="msg-meta">#{i} &middot; {created}</span></div>'
                f'<div class="msg-body">{content}</div>'
                f'</div>'
            )
        chat_html = "\n".join(chat_rows) if chat_rows else '<div class="empty">No chat items yet</div>'

        # Compaction
        comp = snap["compaction"]
        comp_stats = comp.get("stats", {})

        # Instructions (collapsible)
        instr = esc(snap["instructions"][:5000]).replace("\n", "<br>")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Nova Context Viewer</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --dim: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --purple: #bc8cff; --orange: #f0883e;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace; font-size: 13px;
         background: var(--bg); color: var(--text); padding: 16px; line-height: 1.5; }}
  h1 {{ font-size: 16px; color: var(--accent); margin-bottom: 4px; }}
  .meta-bar {{ color: var(--dim); font-size: 12px; margin-bottom: 16px;
              display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
  .meta-bar .badge {{ background: var(--surface); border: 1px solid var(--border);
                     padding: 2px 8px; border-radius: 4px; }}
  .meta-bar .live {{ color: var(--green); }}

  .section {{ background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; margin-bottom: 16px; overflow: hidden; }}
  .section-header {{ padding: 10px 14px; border-bottom: 1px solid var(--border);
                    font-weight: 600; font-size: 13px; display: flex;
                    justify-content: space-between; align-items: center; cursor: pointer; }}
  .section-header:hover {{ background: rgba(88,166,255,0.05); }}
  .section-body {{ padding: 12px 14px; max-height: 600px; overflow-y: auto; }}

  .msg {{ margin-bottom: 10px; padding: 8px 10px; border-radius: 6px;
          border-left: 3px solid var(--border); }}
  .msg.system, .msg.developer {{ border-left-color: var(--purple); background: rgba(188,140,255,0.05); }}
  .msg.user {{ border-left-color: var(--accent); background: rgba(88,166,255,0.05); }}
  .msg.assistant {{ border-left-color: var(--green); background: rgba(63,185,80,0.05); }}
  .msg.tool {{ border-left-color: var(--orange); background: rgba(240,136,62,0.05); }}
  .msg-header {{ display: flex; justify-content: space-between; margin-bottom: 4px; }}
  .role-badge {{ font-size: 11px; font-weight: 700; text-transform: uppercase;
                padding: 1px 6px; border-radius: 3px; }}
  .role-badge.system, .role-badge.developer {{ background: rgba(188,140,255,0.2); color: var(--purple); }}
  .role-badge.user {{ background: rgba(88,166,255,0.2); color: var(--accent); }}
  .role-badge.assistant {{ background: rgba(63,185,80,0.2); color: var(--green); }}
  .role-badge.tool {{ background: rgba(240,136,62,0.2); color: var(--orange); }}
  .msg-meta {{ font-size: 11px; color: var(--dim); }}
  .msg-body {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; }}

  .tier {{ margin-bottom: 8px; }}
  .tier-label {{ font-weight: 700; font-size: 12px; margin-bottom: 2px; }}
  .tier-label.hot {{ color: var(--red); }}
  .tier-label.warm {{ color: var(--yellow); }}
  .tier-label.cold {{ color: var(--accent); }}
  .tier-content {{ background: var(--bg); padding: 8px; border-radius: 4px;
                  font-size: 12px; white-space: pre-wrap; word-break: break-word;
                  min-height: 24px; color: var(--dim); }}

  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
                gap: 8px; }}
  .stat {{ background: var(--bg); padding: 6px 10px; border-radius: 4px; }}
  .stat-label {{ font-size: 11px; color: var(--dim); }}
  .stat-value {{ font-size: 14px; font-weight: 600; }}

  .empty {{ color: var(--dim); font-style: italic; padding: 8px; }}
  details summary {{ cursor: pointer; color: var(--accent); font-size: 12px; }}
  .collapsed {{ display: none; }}

  #status {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            background: var(--green); margin-right: 6px; animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
</style>
</head>
<body>
<h1><span id="status"></span>Nova Context Viewer</h1>
<div class="meta-bar">
  <span class="badge">Agent: <b>{esc(snap["agent_class"])}</b></span>
  <span class="badge">Mode: <b>{esc(snap["mode"])}</b></span>
  <span class="badge">Chat items: <b>{snap["chat_item_count"]}</b></span>
  <span class="badge">Uptime: <b>{snap["uptime_seconds"]}s</b></span>
  <span class="badge">Athlete: <b>{esc(affect_badge)}</b></span>
  <span class="badge live">Refreshing every 1.5s</span>
</div>

<!-- COMPACTION TIERS -->
<div class="section">
  <div class="section-header">
    Compaction Tiers
    <span style="font-weight:normal;font-size:12px;color:var(--dim)">
      Cycles: {comp_stats.get("cycle_count", 0)} &middot;
      Events: {comp_stats.get("total_events_processed", 0)} &middot;
      Cost: ${comp_stats.get("total_cost_usd", 0):.4f}
    </span>
  </div>
  <div class="section-body">
    <div class="tier"><div class="tier-label hot">HOT (last ~60s)</div>
      <div class="tier-content">{esc(comp.get("hot", "") or "(empty)")}</div></div>
    <div class="tier"><div class="tier-label warm">WARM (1-10 min)</div>
      <div class="tier-content">{esc(comp.get("warm", "") or "(empty)")}</div></div>
    <div class="tier"><div class="tier-label cold">COLD (10+ min)</div>
      <div class="tier-content">{esc(comp.get("cold", "") or "(empty)")}</div></div>
  </div>
</div>

<!-- CHAT CONTEXT -->
<div class="section">
  <div class="section-header">
    Chat Context ({snap["chat_item_count"]} items)
    <span style="font-weight:normal;font-size:12px;color:var(--dim)">
      What the LLM sees right now
    </span>
  </div>
  <div class="section-body" id="chat-body">
    {chat_html}
  </div>
</div>

<!-- SYSTEM PROMPT -->
<div class="section">
  <div class="section-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
    System Prompt ({snap["instructions_length"]} chars)
    <span style="font-weight:normal;font-size:12px;color:var(--dim)">click to toggle</span>
  </div>
  <div class="section-body collapsed">
    <div class="msg system"><div class="msg-body">{instr or '<span class="empty">No instructions</span>'}</div></div>
  </div>
</div>

<script>
  // Auto-refresh: reload the whole page every 1.5s
  setTimeout(() => location.reload(), 1500);

  // Auto-scroll chat to bottom
  const chatBody = document.getElementById('chat-body');
  if (chatBody) chatBody.scrollTop = chatBody.scrollHeight;
</script>
</body>
</html>"""
