"""
Workout Session State Management
Handles real-time workout state during active training sessions
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class SetProgress:
    """Represents progress on a single set"""
    set_id: int
    set_number: int
    target_reps: int
    target_weight: Optional[float]
    intensity_percent: Optional[float]
    rpe_target: Optional[float]
    rest_seconds: int
    velocity_threshold: Optional[float]
    completed: bool = False
    performed_reps: Optional[int] = None
    performed_weight: Optional[float] = None
    actual_rpe: Optional[float] = None
    measured_velocity: Optional[float] = None
    completed_at: Optional[str] = None  # ISO format timestamp
    weight_unit: Optional[str] = None  # "kg" | "lb"; target_weight 0 = bodyweight


@dataclass
class ExerciseProgress:
    """Represents progress on a single exercise"""
    workout_exercise_id: int
    exercise_id: int
    exercise_name: str
    muscle_group: Optional[str]
    category: Optional[str]
    order_number: int
    notes: Optional[str]
    sets: List[SetProgress]
    current_set_index: int = 0

    def get_current_set(self) -> Optional[SetProgress]:
        """Get the currently active set"""
        if 0 <= self.current_set_index < len(self.sets):
            return self.sets[self.current_set_index]
        return None

    def is_complete(self) -> bool:
        """Check if all sets are completed"""
        return all(s.completed for s in self.sets)

    def count_completed_sets(self) -> int:
        """Count how many sets have been completed"""
        return sum(1 for s in self.sets if s.completed)


class WorkoutSession:
    """
    Manages state for an active workout session.
    Tracks current exercise, set, completion status, and progress logging.
    """

    def __init__(self, user_id: str, schedule_id: Optional[int], workout_data: Dict,
                 is_quick_exercise: bool = False):
        """
        Initialize a new workout session.

        Args:
            user_id: User's UUID as string
            schedule_id: Schedule entry ID for this workout (None for quick exercises)
            workout_data: Full workout structure from get_todays_workout()
            is_quick_exercise: True if this is an ad-hoc single exercise session
        """
        self.user_id = user_id
        self.schedule_id = schedule_id
        self.is_quick_exercise = is_quick_exercise
        self.workout_id = workout_data.get("workout_id")
        self.workout_name = workout_data["workout_name"]
        self.description = workout_data.get("description", "")
        self.week_number = workout_data.get("week_number")
        self.day_number = workout_data.get("day_number")
        self.phase = workout_data.get("phase")

        # Parse exercises and sets into structured objects
        self.exercises: List[ExerciseProgress] = []
        for ex_data in workout_data.get("exercises", []):
            sets = []
            for set_data in ex_data.get("sets", []):
                sets.append(SetProgress(
                    set_id=set_data["set_id"],
                    set_number=set_data["set_number"],
                    target_reps=set_data["reps"],
                    target_weight=set_data.get("weight"),
                    intensity_percent=set_data.get("intensity_percent"),
                    rpe_target=set_data.get("rpe"),
                    rest_seconds=set_data.get("rest_seconds", 120),
                    velocity_threshold=set_data.get("velocity_threshold"),
                    weight_unit=set_data.get("weight_unit"),
                ))

            self.exercises.append(ExerciseProgress(
                workout_exercise_id=ex_data["workout_exercise_id"],
                exercise_id=ex_data["exercise_id"],
                exercise_name=ex_data["exercise_name"],
                muscle_group=ex_data.get("muscle_group"),
                category=ex_data.get("category"),
                order_number=ex_data["order_number"],
                notes=ex_data.get("notes"),
                sets=sets
            ))

        # Sort exercises by order number
        self.exercises.sort(key=lambda x: x.order_number)

        # Session tracking
        self.current_exercise_index = 0
        self.start_time = datetime.utcnow().isoformat()
        self.end_time: Optional[str] = None
        self.is_active = True
        self.notes: List[str] = []

    def get_current_exercise(self) -> Optional[ExerciseProgress]:
        """Get the currently active exercise"""
        if 0 <= self.current_exercise_index < len(self.exercises):
            return self.exercises[self.current_exercise_index]
        return None

    def get_current_set(self) -> Optional[SetProgress]:
        """Get the currently active set"""
        exercise = self.get_current_exercise()
        if exercise:
            return exercise.get_current_set()
        return None

    def last_completed_set(self) -> Optional[SetProgress]:
        """The set finished most recently (sets complete in workout order)."""
        completed = [s for ex in self.exercises for s in ex.sets if s.completed]
        return completed[-1] if completed else None

    def mark_set_complete(
        self,
        performed_reps: int,
        performed_weight: Optional[float] = None,
        rpe: Optional[float] = None,
        measured_velocity: Optional[float] = None
    ) -> bool:
        """
        Mark the current set as complete and record performance.

        Args:
            performed_reps: Actual reps completed
            performed_weight: Actual weight used
            rpe: Rate of perceived exertion (1-10)
            measured_velocity: Measured bar velocity (m/s)

        Returns:
            True if successful, False if no current set
        """
        current_set = self.get_current_set()
        if not current_set:
            return False

        current_set.completed = True
        current_set.performed_reps = performed_reps
        current_set.performed_weight = performed_weight
        current_set.actual_rpe = rpe
        current_set.measured_velocity = measured_velocity
        current_set.completed_at = datetime.utcnow().isoformat()

        return True

    def advance_to_next_set(self) -> bool:
        """
        Move to the next set or exercise.

        Returns:
            True if advanced successfully, False if workout is complete
        """
        exercise = self.get_current_exercise()
        if not exercise:
            return False

        # Try to advance to next set in current exercise
        if exercise.current_set_index < len(exercise.sets) - 1:
            exercise.current_set_index += 1
            return True

        # Current exercise is done, move to next exercise
        if self.current_exercise_index < len(self.exercises) - 1:
            self.current_exercise_index += 1
            return True

        # Workout is complete
        return False

    def skip_current_set(self, reason: Optional[str] = None) -> bool:
        """
        Skip the current set without logging it.

        Args:
            reason: Optional reason for skipping

        Returns:
            True if skipped successfully
        """
        current_set = self.get_current_set()
        if not current_set:
            return False

        # Mark as completed with zero performance
        current_set.completed = True
        current_set.performed_reps = 0
        current_set.completed_at = datetime.utcnow().isoformat()

        if reason:
            self.notes.append(f"Skipped set {current_set.set_number}: {reason}")

        return True

    def skip_current_exercise(self, reason: Optional[str] = None) -> bool:
        """
        Skip the entire current exercise and move to the next one.

        Args:
            reason: Optional reason for skipping

        Returns:
            True if skipped successfully
        """
        exercise = self.get_current_exercise()
        if not exercise:
            return False

        # Mark all remaining sets as skipped
        for set_obj in exercise.sets:
            if not set_obj.completed:
                set_obj.completed = True
                set_obj.performed_reps = 0
                set_obj.completed_at = datetime.utcnow().isoformat()

        if reason:
            self.notes.append(f"Skipped exercise '{exercise.exercise_name}': {reason}")

        # Move to next exercise
        if self.current_exercise_index < len(self.exercises) - 1:
            self.current_exercise_index += 1
            return True

        return False

    def get_next_exercise(self) -> Optional[ExerciseProgress]:
        """Preview the next exercise without advancing"""
        next_index = self.current_exercise_index + 1
        if next_index < len(self.exercises):
            return self.exercises[next_index]
        return None

    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get a summary of workout progress.

        Returns:
            Dict with progress statistics
        """
        total_exercises = len(self.exercises)
        completed_exercises = sum(1 for ex in self.exercises if ex.is_complete())

        total_sets = sum(len(ex.sets) for ex in self.exercises)
        completed_sets = sum(ex.count_completed_sets() for ex in self.exercises)

        current_exercise = self.get_current_exercise()

        return {
            "workout_name": self.workout_name,
            "total_exercises": total_exercises,
            "completed_exercises": completed_exercises,
            "total_sets": total_sets,
            "completed_sets": completed_sets,
            "percent_complete": round((completed_sets / total_sets * 100) if total_sets > 0 else 0, 1),
            "current_exercise_name": current_exercise.exercise_name if current_exercise else None,
            "current_exercise_number": self.current_exercise_index + 1,
            "current_set_number": current_exercise.current_set_index + 1 if current_exercise else None,
            "is_active": self.is_active
        }

    def is_workout_complete(self) -> bool:
        """Check if all exercises and sets are completed"""
        return all(ex.is_complete() for ex in self.exercises)

    def end_session(self) -> None:
        """Mark the session as ended"""
        self.is_active = False
        self.end_time = datetime.utcnow().isoformat()

    def get_completed_sets_for_logging(self) -> List[Dict]:
        """
        Get all completed sets that need to be logged to ProgressLog.
        Returns list of dicts ready for database insertion.

        Returns:
            List of set data for logging
        """
        completed = []
        for exercise in self.exercises:
            for set_obj in exercise.sets:
                if set_obj.completed and set_obj.performed_reps is not None:
                    completed.append({
                        "set_id": set_obj.set_id,
                        "performed_reps": set_obj.performed_reps,
                        "performed_weight": set_obj.performed_weight,
                        "rpe": set_obj.actual_rpe,
                        "measured_velocity": set_obj.measured_velocity,
                        "completed_at": set_obj.completed_at
                    })
        return completed

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the session to a dictionary for storage in AgentState.

        Returns:
            Dict representation of the session
        """
        return {
            "user_id": self.user_id,
            "schedule_id": self.schedule_id,
            "is_quick_exercise": self.is_quick_exercise,
            "workout_id": self.workout_id,
            "workout_name": self.workout_name,
            "description": self.description,
            "week_number": self.week_number,
            "day_number": self.day_number,
            "phase": self.phase,
            "exercises": [
                {
                    "workout_exercise_id": ex.workout_exercise_id,
                    "exercise_id": ex.exercise_id,
                    "exercise_name": ex.exercise_name,
                    "muscle_group": ex.muscle_group,
                    "category": ex.category,
                    "order_number": ex.order_number,
                    "notes": ex.notes,
                    "current_set_index": ex.current_set_index,
                    "sets": [asdict(s) for s in ex.sets]
                }
                for ex in self.exercises
            ],
            "current_exercise_index": self.current_exercise_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_active": self.is_active,
            "notes": self.notes
        }

    @classmethod
    def create_quick_session(
        cls,
        user_id: str,
        exercise_name: str,
        sets: int,
        reps: int,
        weight: float = 0.0,
        rest_seconds: int = 120,
        weight_unit: Optional[str] = "lb",
    ) -> 'WorkoutSession':
        set_list = [
            SetProgress(
                set_id=None,
                set_number=i + 1,
                target_reps=reps,
                target_weight=weight,
                intensity_percent=None,
                rpe_target=None,
                rest_seconds=rest_seconds,
                velocity_threshold=None,
                weight_unit=weight_unit,
            )
            for i in range(sets)
        ]

        exercise = ExerciseProgress(
            workout_exercise_id=None,
            exercise_id=None,
            exercise_name=exercise_name,
            muscle_group=None,
            category="Strength",
            order_number=1,
            notes=None,
            sets=set_list,
        )

        session = cls(
            user_id=user_id,
            schedule_id=None,
            workout_data={
                "workout_id": None,
                "workout_name": f"Quick {exercise_name}",
                "description": f"Ad-hoc {exercise_name} session",
                "exercises": [],
            },
            is_quick_exercise=True,
        )
        session.exercises = [exercise]
        return session

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkoutSession':
        """
        Deserialize a session from a dictionary (e.g., loaded from AgentState).

        Args:
            data: Dict representation of the session

        Returns:
            WorkoutSession instance
        """
        # Create a minimal workout_data dict for __init__
        workout_data = {
            "workout_id": data.get("workout_id"),
            "workout_name": data["workout_name"],
            "description": data.get("description", ""),
            "week_number": data.get("week_number"),
            "day_number": data.get("day_number"),
            "phase": data.get("phase"),
            "exercises": []
        }

        # Create instance
        session = cls(
            user_id=data["user_id"],
            schedule_id=data.get("schedule_id"),
            workout_data=workout_data,
            is_quick_exercise=data.get("is_quick_exercise", False)
        )

        # Restore exercises and sets from saved state
        session.exercises = []
        for ex_data in data.get("exercises", []):
            sets = [
                SetProgress(**set_data)
                for set_data in ex_data["sets"]
            ]

            exercise = ExerciseProgress(
                workout_exercise_id=ex_data["workout_exercise_id"],
                exercise_id=ex_data["exercise_id"],
                exercise_name=ex_data["exercise_name"],
                muscle_group=ex_data.get("muscle_group"),
                category=ex_data.get("category"),
                order_number=ex_data["order_number"],
                notes=ex_data.get("notes"),
                sets=sets,
                current_set_index=ex_data.get("current_set_index", 0)
            )
            session.exercises.append(exercise)

        # Restore session state
        session.current_exercise_index = data.get("current_exercise_index", 0)
        session.start_time = data.get("start_time")
        session.end_time = data.get("end_time")
        session.is_active = data.get("is_active", True)
        session.notes = data.get("notes", [])

        return session

    def get_current_exercise_description(self) -> str:
        """
        Get a natural language description of the current exercise and set.

        Returns:
            Human-readable string describing what to do next
        """
        exercise = self.get_current_exercise()
        if not exercise:
            return "Workout complete!"

        current_set = exercise.get_current_set()
        if not current_set:
            return f"Moving to next exercise: {exercise.exercise_name}"

        # Build description
        desc = f"{exercise.exercise_name}"

        # Add set info
        desc += f" - Set {current_set.set_number} of {len(exercise.sets)}"

        # Add target reps
        desc += f", {current_set.target_reps} reps"

        # Add intensity if available
        if current_set.intensity_percent:
            desc += f" at {current_set.intensity_percent}%"

        # Add velocity target if VBT
        if current_set.velocity_threshold:
            desc += f" (target velocity: {current_set.velocity_threshold} m/s)"

        # Add rest time
        rest_min = current_set.rest_seconds // 60
        rest_sec = current_set.rest_seconds % 60
        if rest_min > 0:
            desc += f". Rest: {rest_min}:{rest_sec:02d}"
        else:
            desc += f". Rest: {rest_sec}s"

        # Add exercise notes/coaching cues if present
        if exercise.notes:
            desc += f". Note: {exercise.notes}"

        return desc
