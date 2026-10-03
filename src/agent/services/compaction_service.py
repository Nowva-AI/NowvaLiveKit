"""
Compaction Service — Pipeline-Style Rolling Context Summarization

Events flow through a 4-stage pipeline: HOT → WARM → COLD → MEMORY.
Each event is compressed exactly once, at the moment it transitions to the
next tier. No event lives in two tiers simultaneously.

HOT:    Raw event buffer (last 60s, no LLM)
WARM:   Compressed facts (LLM compresses aged-out HOT events on transition)
COLD:   Further compressed summaries (LLM compresses WARM on overflow)
MEMORY: Rolling session memory (LLM folds COLD into it on overflow); stays in the
        summary the agents read, and is also logged to memory.md on disk.

Lasting coaching preferences the user states (humor, talk amount, cue style) are
written to the athlete facts store, which every agent's instructions carry.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.services.athlete_facts import set_preference

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (GPT-5.4-mini)
_INPUT_COST_PER_M = 0.75
_OUTPUT_COST_PER_M = 4.50

# Pipeline thresholds
HOT_WINDOW_SECONDS = 60.0
WARM_TOKEN_LIMIT = 700

PREFERENCE_TOPICS = ("humor", "talk_amount", "cue_style")
_PREFERENCE_LINE_RE = re.compile(
    r"^\s*PREFERENCE:\s*(?P<topic>[a-z_ ]+?)\s*=\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_PREFERENCE_RULE = (
    "- If the user states a lasting preference about how they want to be coached — humor, "
    "how much you talk, or how cues are given — also write it on its own line as "
    "PREFERENCE: <topic> = <short value>, topic one of humor, talk_amount, cue_style. "
    "Only preferences the user stated themselves.\n"
)

_HOT_TO_WARM_PROMPT = """\
You are a context compaction agent for a real-time fitness coaching AI.
Compress the raw conversation events below into key facts.

Rules:
- Preserve exact numbers: weights, reps, sets, RPE, depth angles, distances
- NEVER drop safety-relevant information (injuries, pain, equipment issues)
- NEVER drop user preferences or requests that haven't been addressed yet
- Use timestamp ranges like [HH:MM:SS-HH:MM:SS] to group related events
- Output plain text only, no section markers or headers
- Be concise: merge related events, drop filler/silence/ambient noise
""" + _PREFERENCE_RULE

_WARM_TO_COLD_PROMPT = """\
You are a context compaction agent for a real-time fitness coaching AI.
Compress these conversation facts into one-line summaries with timestamp ranges.

Rules:
- Each line should be a self-contained summary with a timestamp range
- Preserve exact numbers: weights, reps, sets, RPE, depth angles
- NEVER drop safety-relevant information (injuries, pain, equipment issues)
- Output plain text only, no section markers or headers
"""

_COLD_TO_MEMORY_PROMPT = """\
You are a context compaction agent for a real-time fitness coaching AI.
Distill these session summaries into essential session memory.

Rules:
- Keep only what matters for future context: exercises done, performance numbers, injuries, user preferences, coaching decisions
- Preserve exact numbers: weights, reps, sets, RPE, depth angles
- NEVER drop safety-relevant information
- Output plain text only, no section markers or headers
""" + _PREFERENCE_RULE


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


class CompactionService:
    """Pipeline-style rolling context compaction via GPT-5.4-mini.

    Events accumulate in a raw buffer (HOT). Every cycle, events older
    than 60s are compressed by the LLM and appended to WARM. When WARM
    overflows its token limit, it is compressed into COLD. When COLD
    overflows, it is folded into the rolling session MEMORY (also logged
    to memory.md on disk).
    """

    def __init__(
        self,
        session,                          # AgentSession
        state,                            # AgentState
        user_id: str = "guest",
        model: str | None = None,
        interval_seconds: float | None = None,
        cold_flush_threshold: int | None = None,
        openai_client=None,
    ):
        self._session = session
        self._state = state
        self._user_id = user_id

        # Config from env vars with defaults
        self._model = model or os.getenv("COMPACTION_MODEL", "gpt-5.4-mini")
        self._interval_seconds = interval_seconds or float(os.getenv("COMPACTION_INTERVAL", "30"))
        self._cold_flush_threshold = cold_flush_threshold or int(os.getenv("COMPACTION_COLD_FLUSH_THRESHOLD", "300"))
        self._audit = os.getenv("COMPACTION_AUDIT", "false") == "true"

        # OpenAI client — lazy-initialised if not injected
        self._client = openai_client

        # State (protected by _lock)
        self._event_buffer: list[dict[str, Any]] = []
        self._warm: str = ""
        self._cold: str = ""
        self._memory: str = ""

        # Lifecycle
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

        # Timing
        self._last_compaction: float = 0.0
        self._session_start_time: datetime = datetime.now(timezone.utc)

        # Counters
        self._cycle_count: int = 0
        self._cold_flush_count: int = 0
        self._total_events_processed: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

        # Session folder
        self._session_dir: Path | None = None
        self._memory_path: Path | None = None

        # Event handlers (stored for unregistration)
        self._on_conversation = None
        self._on_tools = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Register event listeners on the session and start the background compaction loop."""
        if self._running:
            return

        self._running = True
        self._last_compaction = time.monotonic()
        self._session_start_time = datetime.now(timezone.utc)

        # Lazy-init OpenAI client
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except Exception as e:
                logger.error(f"[COMPACTION:ERROR] Failed to initialize OpenAI client: {e}")
                self._running = False
                return

        # Create session folder
        self._init_session_folder()

        # Register event listeners
        self._on_conversation = lambda ev: self._handle_conversation_item(ev)
        self._on_tools = lambda ev: self._handle_tools_executed(ev)
        self._session.on("conversation_item_added", self._on_conversation)
        self._session.on("function_tools_executed", self._on_tools)

        # Start background loop
        self._task = asyncio.create_task(self._compaction_loop())
        logger.info(
            f"[COMPACTION:SESSION] Started (model={self._model}, "
            f"interval={self._interval_seconds}s, warm_limit={WARM_TOKEN_LIMIT}, "
            f"cold_limit={self._cold_flush_threshold})"
        )

    async def stop(self) -> None:
        """Unregister event listeners, cancel background task, write final metadata."""
        if not self._running:
            return

        self._running = False

        # Cancel background task
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Unregister listeners
        if self._on_conversation:
            try:
                self._session.off("conversation_item_added", self._on_conversation)
            except Exception:
                pass
        if self._on_tools:
            try:
                self._session.off("function_tools_executed", self._on_tools)
            except Exception:
                pass

        # Flush whatever never aged out of the pipeline, then write final metadata
        await self._flush_remaining_context()
        self._write_final_metadata()

        logger.info(
            f"[COMPACTION:SESSION] Stopped — {self._cycle_count} cycles, "
            f"{self._cold_flush_count} flushes, {self._total_events_processed} events, "
            f"est. cost=${self.get_stats()['total_cost_usd']:.4f}"
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_conversation_item(self, ev) -> None:
        """Buffer a conversation turn (user or assistant speech).

        Coaching prompts sent as generate_reply(user_input=...) land in the history as
        user messages without an STT confidence; they are not the user speaking.
        """
        try:
            item = ev.item
            role = getattr(item, "role", "unknown")
            if role == "user" and getattr(item, "transcript_confidence", None) is None:
                return
            text = getattr(item, "text_content", None)
            if callable(text):
                text = text()
            if not text:
                return

            now = datetime.now(timezone.utc)
            entry = {
                "timestamp": time.monotonic(),
                "iso_timestamp": now.isoformat(),
                "role": role,
                "content": text[:500],
                "event_type": f"{role}_speech",
            }
            self._event_buffer.append(entry)
            self._total_events_processed += 1
        except Exception as e:
            logger.error(f"[COMPACTION:ERROR] Failed to buffer conversation item: {e}")

    def _handle_tools_executed(self, ev) -> None:
        """Buffer tool call + result pairs."""
        try:
            now = datetime.now(timezone.utc)
            for call, output in ev.zipped():
                call_content = f"{call.name}({call.arguments})"
                self._event_buffer.append({
                    "timestamp": time.monotonic(),
                    "iso_timestamp": now.isoformat(),
                    "role": "tool",
                    "content": call_content,
                    "event_type": "tool_call",
                })
                self._total_events_processed += 1

                result_str = str(output.output)[:500] if output else ""
                if result_str:
                    self._event_buffer.append({
                        "timestamp": time.monotonic(),
                        "iso_timestamp": now.isoformat(),
                        "role": "tool",
                        "content": f"{call.name} → {result_str}",
                        "event_type": "tool_result",
                    })
                    self._total_events_processed += 1
        except Exception as e:
            logger.error(f"[COMPACTION:ERROR] Failed to buffer tool event: {e}")

    # ------------------------------------------------------------------
    # Background compaction loop
    # ------------------------------------------------------------------

    async def _compaction_loop(self) -> None:
        """Run compaction on a timer. Never raises — logs all errors."""
        try:
            while self._running:
                await asyncio.sleep(self._interval_seconds)
                if not self._running:
                    break
                try:
                    await self._run_compaction_cycle()
                except Exception:
                    logger.exception("[COMPACTION:ERROR] Compaction cycle failed — skipping")
        except asyncio.CancelledError:
            pass

    async def _run_compaction_cycle(self) -> None:
        """Execute one pipeline cycle: age out HOT, cascade overflows."""
        now_mono = time.monotonic()
        cutoff = now_mono - HOT_WINDOW_SECONDS

        # Partition buffer: events older than 60s age out
        async with self._lock:
            still_hot = [e for e in self._event_buffer if e["timestamp"] >= cutoff]
            aged_out = [e for e in self._event_buffer if e["timestamp"] < cutoff]
            self._event_buffer = still_hot

        if not aged_out:
            return

        # Buffer overflow guard
        if len(aged_out) > 100:
            logger.warning(
                f"[COMPACTION:OVERFLOW] {len(aged_out)} events aged out — "
                f"truncating to last 50"
            )
            aged_out = aged_out[-50:]

        self._cycle_count += 1
        self._last_compaction = now_mono
        tokens_in_before = self._total_input_tokens
        tokens_out_before = self._total_output_tokens

        logger.info(
            f"[COMPACTION:CYCLE] Cycle {self._cycle_count} — "
            f"{len(aged_out)} events aging out of HOT, "
            f"{len(still_hot)} remaining in buffer"
        )

        # Step 1: HOT → WARM
        transcript = self._format_events(aged_out)
        compressed = await self._compress(
            _HOT_TO_WARM_PROMPT, transcript, "hot_to_warm",
        )
        if compressed is None:
            # LLM call failed — put events back
            async with self._lock:
                self._event_buffer = aged_out + self._event_buffer
            return
        self._store_preferences(compressed)

        async with self._lock:
            if self._warm:
                self._warm += "\n" + compressed
            else:
                self._warm = compressed

        # Step 2: WARM → COLD (if warm overflows)
        if _estimate_tokens(self._warm) > WARM_TOKEN_LIMIT:
            compressed_warm = await self._compress(
                _WARM_TO_COLD_PROMPT, self._warm, "warm_to_cold",
            )
            if compressed_warm is not None:
                async with self._lock:
                    if self._cold:
                        self._cold += "\n" + compressed_warm
                    else:
                        self._cold = compressed_warm
                    self._warm = ""

        # Step 3: COLD → memory.md (if cold overflows)
        if _estimate_tokens(self._cold) > self._cold_flush_threshold:
            await self._flush_cold_to_memory()

        logger.info(
            f"[COMPACTION:CYCLE] Cycle {self._cycle_count} complete — "
            f"warm={len(self._warm)}c cold={len(self._cold)}c"
        )

        from profiler.collector import SessionProfiler
        SessionProfiler.get_instance().record(
            "compaction", "cycle",
            cycle_number=self._cycle_count,
            events_processed=len(aged_out),
            input_tokens=self._total_input_tokens - tokens_in_before,
            output_tokens=self._total_output_tokens - tokens_out_before,
            hot_chars=sum(len(e["content"]) for e in still_hot),
            warm_chars=len(self._warm),
            cold_chars=len(self._cold),
        )

        # Periodic stats
        if self._cycle_count % 5 == 0:
            stats = self.get_stats()
            logger.info(
                f"[COMPACTION:STATS] After {self._cycle_count} cycles: "
                f"total_events={stats['total_events_processed']}, "
                f"warm={len(self._warm)}c/cold={len(self._cold)}c, "
                f"est_cost=${stats['total_cost_usd']:.4f}"
            )

    # ------------------------------------------------------------------
    # LLM compression
    # ------------------------------------------------------------------

    async def _compress(
        self,
        system_prompt: str,
        content: str,
        transition: str,
    ) -> str | None:
        """Call the LLM to compress content. Returns compressed text or None on failure."""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.2,
                max_completion_tokens=600,
            )

            result = response.choices[0].message.content or ""
            usage = response.usage

            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            self._write_audit_file(
                cycle=self._cycle_count,
                transition=transition,
                input_content=content,
                output_content=result,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            logger.info(
                f"[COMPACTION:{transition.upper()}] "
                f"{len(content)}c → {len(result)}c "
                f"({input_tokens}in/{output_tokens}out tokens)"
            )
            return result.strip()

        except Exception as e:
            logger.error(f"[COMPACTION:ERROR] {transition} API call failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Cold flush to memory.md
    # ------------------------------------------------------------------

    async def _flush_cold_to_memory(self) -> None:
        """Fold cold context into the rolling session memory, and log it to memory.md."""
        if not self._cold:
            return

        source = f"{self._memory}\n{self._cold}" if self._memory else self._cold
        compressed = await self._compress(
            _COLD_TO_MEMORY_PROMPT, source, "cold_to_memory",
        )
        if compressed is None:
            return
        self._store_preferences(compressed)

        async with self._lock:
            self._memory = compressed
            self._cold = ""
        self._cold_flush_count += 1
        logger.info(f"[COMPACTION:FLUSH] Cold folded into session memory (flush #{self._cold_flush_count})")

        if not self._memory_path:
            return
        flush_timestamp = datetime.now(timezone.utc).isoformat()
        try:
            with open(self._memory_path, "a") as f:
                f.write(f"\n## Context flushed at {flush_timestamp}\n\n")
                f.write(compressed)
                f.write("\n\n---\n\n")
        except Exception as e:
            logger.error(f"[COMPACTION:ERROR] Failed to log memory to memory.md: {e}")

    def _store_preferences(self, compressed: str) -> None:
        for match in _PREFERENCE_LINE_RE.finditer(compressed):
            topic = match["topic"].strip().lower().replace(" ", "_")
            if topic not in PREFERENCE_TOPICS:
                continue
            try:
                set_preference(self._state, topic, match["value"])
                logger.info(f"[COMPACTION:PREFERENCE] Stored {topic} preference")
            except Exception as e:
                logger.error(f"[COMPACTION:ERROR] Failed to store {topic} preference: {e}")

    async def _flush_remaining_context(self) -> None:
        """Write any un-flushed HOT/WARM/COLD context to memory.md on stop.

        Compressed via the LLM when possible; raw on failure or timeout so
        session memory is never silently lost.
        """
        if not self._memory_path:
            return

        parts = []
        if self._cold:
            parts.append(self._cold)
        if self._warm:
            parts.append(self._warm)
        if self._event_buffer:
            parts.append(self._format_events(self._event_buffer))
        if not parts:
            return
        remaining = "\n".join(parts)

        compressed = None
        if self._client is not None:
            try:
                compressed = await asyncio.wait_for(
                    self._compress(_COLD_TO_MEMORY_PROMPT, remaining, "final_flush"),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[COMPACTION:FLUSH] Final compression timed out — writing raw context"
                )
        if compressed is not None:
            self._store_preferences(compressed)

        flush_timestamp = datetime.now(timezone.utc).isoformat()
        try:
            with open(self._memory_path, "a") as f:
                f.write(f"\n## Session end flush at {flush_timestamp}\n\n")
                f.write(compressed if compressed is not None else remaining)
                f.write("\n")
            self._event_buffer = []
            self._warm = ""
            self._cold = ""
            self._cold_flush_count += 1
            logger.info("[COMPACTION:FLUSH] Final context flushed to memory.md on stop")
        except Exception as e:
            logger.error(f"[COMPACTION:ERROR] Final flush failed: {e}")

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _write_audit_file(
        self,
        cycle: int,
        transition: str,
        input_content: str,
        output_content: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if not self._audit or not self._session_dir:
            return
        audit = {
            "cycle": cycle,
            "transition": transition,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": input_content,
            "output": output_content,
            "tokens": {"input": input_tokens, "output": output_tokens},
        }
        path = self._session_dir / f"audit_cycle_{cycle}_{transition}.json"
        try:
            path.write_text(json.dumps(audit, indent=2))
        except Exception as e:
            logger.error(f"[COMPACTION:AUDIT] Failed to write {path}: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_events(events: list[dict[str, Any]]) -> str:
        lines = []
        for e in events:
            iso = e["iso_timestamp"]
            ts = iso.split("T")[1][:8] if "T" in iso else "??:??:??"
            lines.append(f"[{ts}] ({e['role']}) {e['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Return the current summary (cold + warm + hot).

        HOT tier is formatted directly from the raw event buffer.
        Called by prune/truncate methods — must be non-blocking.
        """
        parts = []
        if self._memory:
            parts.append(f"[SESSION MEMORY]\n{self._memory}")
        if self._cold:
            parts.append(f"[SESSION CONTEXT]\n{self._cold}")
        if self._warm:
            parts.append(f"[EARLIER CONTEXT]\n{self._warm}")
        if self._event_buffer:
            hot_text = self._format_events(self._event_buffer)
            parts.append(f"[RECENT CONTEXT]\n{hot_text}")
        return "\n\n".join(parts)

    def get_event_buffer_size(self) -> int:
        """Return number of events in the pending buffer."""
        return len(self._event_buffer)

    def get_tier_snapshot(self) -> dict[str, str]:
        """Return formatted HOT/WARM/COLD tier contents for debug display."""
        return {
            "hot": self._format_events(self._event_buffer),
            "warm": self._warm,
            "cold": self._cold,
        }

    def get_stats(self) -> dict[str, Any]:
        """Return compaction stats for monitoring and debug logging."""
        input_cost = (self._total_input_tokens / 1_000_000) * _INPUT_COST_PER_M
        output_cost = (self._total_output_tokens / 1_000_000) * _OUTPUT_COST_PER_M
        age = time.monotonic() - self._last_compaction if self._last_compaction else 0

        return {
            "cycle_count": self._cycle_count,
            "cold_flush_count": self._cold_flush_count,
            "total_events_processed": self._total_events_processed,
            "buffer_size": len(self._event_buffer),
            "warm_length": len(self._warm),
            "cold_length": len(self._cold),
            "last_compaction_age_seconds": round(age, 1),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": round(input_cost + output_cost, 6),
        }

    # ------------------------------------------------------------------
    # Session folder management
    # ------------------------------------------------------------------

    def _init_session_folder(self) -> None:
        """Create the per-session log folder with memory.md and session_meta.json."""
        run_dir = os.environ.get("NOWVA_SESSION_OUTPUT_DIR")
        if run_dir:
            self._session_dir = Path(run_dir) / "compaction"
        else:
            base_dir = os.getenv(
                "COMPACTION_SESSION_LOG_DIR",
                str(Path(__file__).parent.parent / "session_logs"),
            )
            session_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            self._session_dir = Path(base_dir) / self._user_id / session_timestamp

        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(
                f"[COMPACTION:ERROR] Failed to create session folder "
                f"{self._session_dir}: {e} — continuing without persistence"
            )
            self._session_dir = None
            self._memory_path = None
            return

        # Initialize memory.md
        self._memory_path = self._session_dir / "memory.md"
        try:
            now = datetime.now(timezone.utc)
            self._memory_path.write_text(
                f"# Nova Session Memory\n\n"
                f"- **user_id**: {self._user_id}\n"
                f"- **started**: {now.isoformat()}\n"
                f"- **mode**: {self._state.get_mode()}\n\n"
                f"---\n\n"
            )
        except Exception as e:
            logger.error(f"[COMPACTION:ERROR] Failed to write memory.md: {e}")
            self._memory_path = None

        # Write session_meta.json (atomic)
        try:
            meta = {
                "user_id": self._user_id,
                "session_start": datetime.now(timezone.utc).isoformat(),
                "initial_mode": self._state.get_mode(),
                "model": self._model,
                "interval_seconds": self._interval_seconds,
                "warm_token_limit": WARM_TOKEN_LIMIT,
                "cold_flush_threshold": self._cold_flush_threshold,
            }
            meta_path = self._session_dir / "session_meta.json"
            tmp_path = meta_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(meta, indent=2))
            os.replace(str(tmp_path), str(meta_path))
        except Exception as e:
            logger.error(f"[COMPACTION:ERROR] Failed to write session_meta.json: {e}")

        logger.info(f"[COMPACTION:SESSION] Created session folder: {self._session_dir}")

    def _write_final_metadata(self) -> None:
        """Update session_meta.json with final stats on stop()."""
        if not self._session_dir:
            return

        meta_path = self._session_dir / "session_meta.json"
        if not meta_path.exists():
            return

        try:
            meta = json.loads(meta_path.read_text())
            meta.update({
                "session_end": datetime.now(timezone.utc).isoformat(),
                "total_compaction_cycles": self._cycle_count,
                "total_cold_flushes": self._cold_flush_count,
                "total_events_processed": self._total_events_processed,
                "estimated_cost_usd": self.get_stats()["total_cost_usd"],
                "final_mode": self._state.get_mode(),
            })
            tmp_path = meta_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(meta, indent=2))
            os.replace(str(tmp_path), str(meta_path))
            logger.info(f"[COMPACTION:SESSION] Final metadata saved to {meta_path}")
        except Exception as e:
            logger.error(f"[COMPACTION:ERROR] Failed to write final metadata: {e}")
