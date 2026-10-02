"""
Session Logger
Tracks all LLM calls, function calls, and conversation during a session
Outputs to CSV format
"""

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading
from .pricing_config import calculate_cost

# Tool parameters that identify the athlete or describe their health never reach the CSV.
PRIVATE_PARAMETER_KEYS = frozenset({"name", "email", "injury_history"})
REDACTED = "[redacted]"


def _redact(parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {key: REDACTED if key in PRIVATE_PARAMETER_KEYS else value for key, value in parameters.items()}


@dataclass
class LogEvent:
    """Represents a single logged event"""
    timestamp: str
    event_type: str         # "llm_call", "function_call", "conversation", "system"
    component: str          # "realtime_api", "summarization", "program_gen", etc.
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    details_json: str       # JSON string
    transcript_user: str
    transcript_agent: str


class SessionLogger:
    """Singleton session logger for tracking LLM usage and costs"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.session_id: Optional[str] = None
        self.events: List[LogEvent] = []
        self.log_file_path: Optional[Path] = None
        self._lock = threading.Lock()

        # Totals for summary
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.conversation_count = 0
        self.function_call_count = 0
        self.llm_call_count = 0

    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        return cls()

    def start_session(self, log_dir: str | None = None):
        """Initialize new session, or join an existing one via NOWVA_SESSION_LOG env var."""
        import os
        shared_log = os.environ.get("NOWVA_SESSION_LOG")
        if shared_log:
            self.log_file_path = Path(shared_log)
            self.session_id = self.log_file_path.stem.replace("session_", "")
            return

        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        output_dir = Path(log_dir) if log_dir else Path("session_logs")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file_path = output_dir / f"session_{self.session_id}.csv"

        print(f"[SESSION LOGGER] Session started: {self.session_id}")
        print(f"[SESSION LOGGER] Logging to: {self.log_file_path}")

    def log_llm_call(
        self,
        component: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float = None,
        cached_tokens: int = 0,
        is_audio_input: bool = False,
        is_audio_output: bool = False,
        details: Dict[str, Any] = None
    ):
        """Log an LLM API call"""
        # Auto-start session if not started (fallback safety)
        if self.log_file_path is None:
            self.start_session()

        if cost is None:
            cost = calculate_cost(
                input_tokens, output_tokens, model,
                cached_input_tokens=cached_tokens,
                is_audio_input=is_audio_input,
                is_audio_output=is_audio_output
            )

        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            event_type="llm_call",
            component=component,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            details_json=json.dumps(details or {}),
            transcript_user="",
            transcript_agent=""
        )

        self._add_event(event)
        self.llm_call_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

    def log_function_call(
        self,
        function_name: str,
        parameters: Dict[str, Any],
        result: Any,
        estimated_tokens: Tuple[int, int] = (0, 0)
    ):
        """Log a function tool call"""
        input_tokens, output_tokens = estimated_tokens

        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            event_type="function_call",
            component="voice_agent_tools",
            model="function_calling",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=0.0,  # Function calls don't have direct cost (included in conversation)
            details_json=json.dumps({
                "function": function_name,
                "parameters": _redact(parameters),
                "result": str(result)[:500]  # Truncate long results
            }),
            transcript_user="",
            transcript_agent=""
        )

        self._add_event(event)
        self.function_call_count += 1

    def log_conversation(
        self,
        user_text: str,
        agent_text: str,
        estimated_tokens: Tuple[int, int] = None
    ):
        """Log a conversation exchange"""
        if estimated_tokens is None:
            from .token_estimator import estimate_text_tokens
            input_tokens = estimate_text_tokens(user_text)
            output_tokens = estimate_text_tokens(agent_text)
        else:
            input_tokens, output_tokens = estimated_tokens

        cost = calculate_cost(
            input_tokens, output_tokens,
            "gpt-4o-mini",
        )

        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            event_type="conversation",
            component="cascade_pipeline",
            model="gpt-4o-mini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            details_json="{}",
            transcript_user=user_text,
            transcript_agent=agent_text
        )

        self._add_event(event)
        self.conversation_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

    def log_system_event(self, event_name: str, details: Dict[str, Any] = None):
        """Log system event (app start, mode change, etc.)"""
        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            event_type="system",
            component="app",
            model="",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            details_json=json.dumps({"event": event_name, **(details or {})}),
            transcript_user="",
            transcript_agent=""
        )

        self._add_event(event)

    def _add_event(self, event: LogEvent):
        """Thread-safe event addition — flushes immediately so subprocess
        kills don't lose buffered data."""
        with self._lock:
            self.events.append(event)
            self._flush_to_csv()

    def _flush_to_csv(self):
        """Write buffered events to CSV (append mode)"""
        if not self.log_file_path or not self.events:
            return

        # Check if file exists (for header)
        file_exists = self.log_file_path.exists()

        with open(self.log_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'event_type', 'component', 'model',
                'input_tokens', 'output_tokens', 'total_tokens', 'cost_usd',
                'details_json', 'transcript_user', 'transcript_agent'
            ])

            if not file_exists:
                writer.writeheader()

            for event in self.events:
                writer.writerow(asdict(event))

        # Clear buffer after writing
        self.events.clear()

    def end_session(self) -> str:
        """Flush final events and generate summary report"""
        self.log_system_event("session_end")

        with self._lock:
            self._flush_to_csv()

        # Generate summary
        summary = self._generate_summary()

        # Write summary to separate text file
        if self.log_file_path:
            summary_path = self.log_file_path.with_suffix('.txt')
            with open(summary_path, 'w') as f:
                f.write(summary)

        return summary

    def _generate_summary(self) -> str:
        """Generate human-readable summary from the CSV (includes subprocess data)."""
        totals = self._read_csv_totals()
        summary_lines = [
            "="*60,
            f"SESSION SUMMARY - {self.session_id}",
            "="*60,
            "",
            "TOTALS:",
            f"  Total LLM Calls: {totals['llm_calls']}",
            f"  Total Function Calls: {totals['function_calls']}",
            f"  Total Conversation Exchanges: {totals['conversations']}",
            "",
            f"  Total Input Tokens: {totals['input_tokens']:,}",
            f"  Total Output Tokens: {totals['output_tokens']:,}",
            f"  Total Tokens: {totals['input_tokens'] + totals['output_tokens']:,}",
            "",
            f"  Estimated Total Cost: ${totals['cost']:.4f}",
            "",
            f"Log file: {self.log_file_path}",
            "="*60
        ]

        return "\n".join(summary_lines)

    def _read_csv_totals(self) -> dict:
        """Re-derive totals from the CSV file so subprocess contributions are included."""
        totals = {
            "llm_calls": 0, "function_calls": 0, "conversations": 0,
            "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
        }
        if not self.log_file_path or not self.log_file_path.exists():
            return totals
        try:
            with open(self.log_file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    event_type = row.get("event_type", "")
                    if event_type == "llm_call":
                        totals["llm_calls"] += 1
                    elif event_type == "function_call":
                        totals["function_calls"] += 1
                    elif event_type == "conversation":
                        totals["conversations"] += 1
                    totals["input_tokens"] += int(row.get("input_tokens", 0))
                    totals["output_tokens"] += int(row.get("output_tokens", 0))
                    totals["cost"] += float(row.get("cost_usd", 0))
        except Exception:
            pass
        return totals

    def get_log_path(self) -> Path:
        """Get path to CSV log file"""
        return self.log_file_path
