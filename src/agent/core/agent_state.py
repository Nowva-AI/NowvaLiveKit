"""
Agent State Management
Handles persistent state for the Nova AI voice agent across different modes
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Anchor state files to the project root so main.py and the voice agent
# (which run with different CWDs) read/write the same file.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALID_MODES = frozenset({
    "onboarding",
    "main_menu",
    "workout",
    "program_creation",
    "schedule",
})

# Top-level state keys writable via set(): the default schema keys plus
# runtime-only sections that agents create on the fly.
VALID_TOP_LEVEL_KEYS = frozenset({
    "mode",
    "athlete",
    "user",
    "session",
    "workout",
    "quick_exercise",
    "program_creation",
    "program_update",
    "schedule",
    "calibration",
    "shutdown_requested",
})

_notify_fd: int | None = None


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


@contextmanager
def _file_lock(filepath: str) -> Iterator[None]:
    # Advisory lock on a sibling .lock file. We lock a sibling instead of the
    # state file itself because os.replace() swaps the inode, so a lock held
    # on the old inode would not block a writer replacing the file.
    lock_fd = os.open(filepath + ".lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def set_state_notify_fd(fd: int) -> None:
    global _notify_fd
    _notify_fd = fd


class AgentState:
    """
    Global state manager for the Nova AI agent

    Manages transitions between modes:
    - onboarding: New user setup
    - main_menu: Primary interaction hub
    - workout: Active workout session
    """

    def __init__(self, user_id: Optional[str] = None, state_dir: str | Path | None = None):
        """
        Initialize agent state

        Args:
            user_id: Optional user ID to load existing state
            state_dir: Directory for the state file, defaults to the project root
        """
        self._state_dir = Path(state_dir) if state_dir is not None else PROJECT_ROOT
        self._user_loaded_from_db = False  # Track if we've already loaded user info from DB
        self.state = {
            "mode": "onboarding",  # Current mode: onboarding, main_menu, workout
            "user": {
                "id": user_id,
                "username": None,
                "name": None,
                "email": None,
                "first_time_main_menu": True,
                "created_at": None,
            },
            "session": {
                "started_at": datetime.now().isoformat(),
                "last_mode_switch": None,
            },
            "workout": {
                "active": False,
                "current_session": None,  # WorkoutSession.to_dict() stored here during active workout
                "exercise": None,  # Legacy field (kept for compatibility)
                "reps": 0,  # Legacy field
                "sets": 0,  # Legacy field
            },
            "quick_exercise": {
                "exercise_name": None,
                "gathering_params": False,
            },
            "program_creation": {
                "has_vbt_capability": False,  # Automatically set based on fitness level + goal + sport
                # State tracking fields for state-driven flow:
                "height_cm": None,
                "weight_kg": None,
                "age": None,
                "sex": None,
                "goal_raw": None,
                "goal_category": None,
                "goal_confirmation": None,  # For prompt to use in confirmation
                "recommended_duration": None,  # For prompt to suggest
                "duration_weeks": None,
                "days_per_week": None,
                "session_duration": None,
                "injury_history": None,
                "specific_sport": None,
                "user_notes": None,
                "fitness_level": None,
                "vbt_enabled": None,
                "all_params_collected": False,
            }
        }

        # Load existing state if user_id provided
        if user_id:
            self.load_state(user_id)

        # Reset session-scoped flags so they fire once per session, not once ever
        self.state.setdefault("session", {})
        self.state["session"]["main_menu_greeted"] = False
        self.state["session"]["last_mode_switch"] = None
        self.state["session"]["started_at"] = datetime.now().isoformat()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get state value by key path (supports dot notation)

        Args:
            key: Key path (e.g., "user.name" or "mode")
            default: Default value if key not found

        Returns:
            Value at key path or default
        """
        keys = key.split(".")
        value = self.state

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set state value by key path (supports dot notation)

        Args:
            key: Key path (e.g., "user.name" or "mode")
            value: Value to set
        """
        keys = key.split(".")
        if keys[0] not in VALID_TOP_LEVEL_KEYS:
            raise ValueError(
                f"Unknown top-level state key: {keys[0]!r}. "
                f"Valid keys: {sorted(VALID_TOP_LEVEL_KEYS)}"
            )
        target = self.state

        # Navigate to parent
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            elif target[k] is None:
                # If the key exists but is None, replace it with an empty dict
                target[k] = {}
            target = target[k]

        # Set final value
        target[keys[-1]] = value

    def switch_mode(self, new_mode: str):
        """
        Switch to a new mode

        Args:
            new_mode: Mode to switch to (must be one of VALID_MODES)
        """
        if new_mode not in VALID_MODES:
            raise ValueError(
                f"Unknown mode: {new_mode!r}. Valid modes: {sorted(VALID_MODES)}"
            )
        old_mode = self.state["mode"]
        self.state["mode"] = new_mode
        self.state["session"]["last_mode_switch"] = {
            "from": old_mode,
            "to": new_mode,
            "timestamp": datetime.now().isoformat()
        }

        print(f"[STATE] Mode switched: {old_mode} → {new_mode}")

    def update_user(self, **kwargs):
        """
        Update user information

        Args:
            **kwargs: User fields to update (name, email, username, etc.)
        """
        for key, value in kwargs.items():
            if key in self.state["user"]:
                self.state["user"][key] = value
                print(f"[STATE] User.{key} updated")

    def mark_main_menu_visited(self):
        """Mark that user has visited main menu (no longer first time)"""
        self.state["user"]["first_time_main_menu"] = False

    def is_first_time_main_menu(self) -> bool:
        """Check if this is the user's first time in main menu"""
        return self.state["user"].get("first_time_main_menu", True)

    def get_mode(self) -> str:
        """Get current mode"""
        return self.state["mode"]

    def get_user(self) -> Dict:
        """Get user information"""
        return self.state["user"]

    def get_session(self) -> Dict:
        """Get session information"""
        return self.state["session"]

    def _state_filepath(self, user_id: str) -> str:
        return str(self._state_dir / f".agent_state_{user_id}.json")

    def save_state(self, filepath: Optional[str] = None):
        """
        Save state to file using atomic write (write to temp, then rename).
        This prevents partial reads when main.py polls the state file.

        Args:
            filepath: Optional custom filepath, defaults to .agent_state.json
        """
        if filepath is None:
            user_id = self.state["user"].get("id", "guest")
            filepath = self._state_filepath(user_id)

        try:
            # Atomic write: write to temp file, then rename.
            # The advisory lock serializes writers across processes
            # (main.py and the voice agent share this file).
            tmp_filepath = filepath + ".tmp"
            with _file_lock(filepath):
                with open(tmp_filepath, 'w') as f:
                    json.dump(self.state, f, indent=2)
                os.replace(tmp_filepath, filepath)
            print(f"[STATE] Saved to {filepath}")
        except Exception as e:
            print(f"[STATE] Failed to save state: {e}")

        if _notify_fd is not None:
            try:
                os.write(_notify_fd, b'\x00')
            except OSError:
                pass

    def load_state(self, user_id: str):
        """
        Load state from file AND populate user info from database

        Args:
            user_id: User ID to load state for
        """
        filepath = self._state_filepath(user_id)

        if not os.path.exists(filepath):
            print(f"[STATE] No saved state found for user {user_id}")
            # Still continue to load user info from database
        else:
            try:
                with _file_lock(filepath):
                    with open(filepath, 'r') as f:
                        content = f.read()
                if not content or content.strip() == '':
                    print(f"[STATE] Empty state file, skipping load")
                    return
                loaded_state = json.loads(content)
                # Deep-merge so a partial nested dict in the file does not
                # wipe sibling default keys
                _deep_merge(self.state, loaded_state)
                # Suppressed verbose logging - uncomment for debugging
                # print(f"[STATE] Loaded state for user {user_id}")
            except json.JSONDecodeError as e:
                logger.warning(f"[STATE] Corrupted state file {filepath}: {e}")
            except Exception as e:
                print(f"[STATE] Failed to load state: {e}")

        # Load user info from database ONLY if not already loaded (prevent spam)
        if not self._user_loaded_from_db:
            self._load_user_from_database(user_id)

    def _load_user_from_database(self, user_id: str):
        """
        Load user information from database (cached - only loads once)

        Args:
            user_id: User ID to load
        """
        if self._user_loaded_from_db:
            return  # Already loaded, skip to prevent DB spam

        try:
            from db.database import SessionLocal
            from db.models import User

            db = SessionLocal()
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    self.state["user"]["id"] = str(user.id)
                    self.state["user"]["username"] = user.username
                    self.state["user"]["name"] = user.name
                    self.state["user"]["email"] = user.email
                    self.state["user"]["height_cm"] = float(user.height_cm) if user.height_cm else None
                    self.state["user"]["created_at"] = user.created_at.isoformat() if user.created_at else None
                    self._user_loaded_from_db = True  # Mark as loaded
                    print("[STATE] Loaded user info from database")
                else:
                    print(f"[STATE] User {user_id} not found in database")
                    self._user_loaded_from_db = True  # Mark as attempted
            finally:
                db.close()
        except Exception as e:
            print(f"[STATE] Failed to load user from database: {e}")
            self._user_loaded_from_db = True  # Mark as attempted even if failed

    def reload_state(self):
        """
        Reload state from file for the current user
        """
        user_id = self.state["user"].get("id")
        if not user_id:
            print(f"[STATE] Cannot reload - no user ID set")
            return

        self.load_state(user_id)

    def to_dict(self) -> Dict:
        """Get state as dictionary"""
        return self.state

    def __repr__(self) -> str:
        """String representation of state"""
        return f"AgentState(mode={self.state['mode']}, user={self.state['user']['name']})"
