# memory/lesson_pool.py
from typing import List, Optional, Dict
from pathlib import Path
import json
from collections import defaultdict

from memory.lesson_types import Lesson, SolutionLesson, DebugLesson, LessonType
from core.config import Config
from llm.llm_client import get_client
from llm.prompt_manager import get_prompt_manager

class LessonPool:
    """
    Manages the pool of lessons learned during search.
    Implements lesson management from Section 4.3.
    
    Features:
    - Separate pools for Solution and Debug lessons
    - Deduplication to avoid redundant lessons
    - Retrieval of most recent K lessons
    """
    
    def __init__(self, max_lessons: int = None):
        self.max_lessons = max_lessons or Config.KM
        
        # Separate pools by type
        self.solution_lessons: List[SolutionLesson] = []
        self.debug_lessons: List[DebugLesson] = []
        
        # ID tracking
        self._lesson_counter = defaultdict(int)
        
        # Statistics
        self.total_added = 0
        self.total_deduplicated = 0
    
    def add_lesson(self, lesson: Lesson, check_duplicate: bool = True) -> bool:
        """
        Add a lesson to the pool.

        Enforces KM limit by removing oldest lessons when pool is full.
        This prevents context overflow as specified in the MARS paper.

        Args:
            lesson: Lesson to add
            check_duplicate: Whether to check for duplicates

        Returns:
            True if added, False if rejected as duplicate
        """
        if check_duplicate:
            if self._is_duplicate(lesson):
                self.total_deduplicated += 1
                print(f"  ⊘ Lesson rejected as duplicate: {lesson.title}")
                return False

        # Add to appropriate pool
        if lesson.type == LessonType.SOLUTION:
            self.solution_lessons.append(lesson)
            # Enforce KM limit - remove oldest if exceeds max
            if len(self.solution_lessons) > self.max_lessons:
                removed = self.solution_lessons.pop(0)
                print(f"  ⊖ Evicted oldest lesson to maintain KM={self.max_lessons}: {removed.id}")
        elif lesson.type == LessonType.DEBUG:
            self.debug_lessons.append(lesson)
            # Enforce KM limit for debug lessons too
            if len(self.debug_lessons) > self.max_lessons:
                removed = self.debug_lessons.pop(0)
                print(f"  ⊖ Evicted oldest debug lesson to maintain KM={self.max_lessons}: {removed.id}")
        else:
            raise ValueError(f"Unknown lesson type: {lesson.type}")

        self.total_added += 1
        print(f"  ✓ Lesson added: {lesson.id} - {lesson.title}")

        return True
    
    def _is_duplicate(self, new_lesson: Lesson) -> bool:
        """
        Check if lesson is a semantic duplicate using LLM-based reasoning.
        Uses the lesson_deduplication prompt from Appendix F.

        Falls back to title matching if the LLM call fails.
        """
        # Get existing lessons of same type
        if new_lesson.type == LessonType.SOLUTION:
            existing = self.solution_lessons
        else:
            existing = self.debug_lessons

        if not existing:
            return False

        # Quick check: exact title match (cheap, no LLM call)
        for lesson in existing:
            if lesson.title.lower().strip() == new_lesson.title.lower().strip():
                return True

        # LLM-based semantic deduplication
        try:
            client = get_client()
            pm = get_prompt_manager()

            existing_text = "\n\n".join(l.format_for_prompt() for l in existing[-15:])
            new_text = new_lesson.format_for_prompt()

            prompt = pm.get_prompt(
                "lesson_deduplication",
                existing_lessons=existing_text,
                new_lesson=new_text,
            )

            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You maintain a knowledge base of technical lessons."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=8000,
            )

            content = response.get("content", "")

            import re, json as _json
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                data = _json.loads(json_match.group(1))
                is_dup = data.get("duplicate", False)
                if is_dup:
                    print(f"  [Dedup] LLM says duplicate: {data.get('reasoning', '')[:80]}")
                return bool(is_dup)

        except Exception as e:
            print(f"  [Dedup] LLM dedup failed ({e}), falling back to title match")

        return False
    
    def get_recent_lessons(
        self,
        lesson_type: Optional[LessonType] = None,
        k: Optional[int] = None
    ) -> List[Lesson]:
        """
        Get K most recent lessons of specified type.
        
        Args:
            lesson_type: Type of lessons to retrieve (None = all)
            k: Number of lessons (None = use max_lessons)
            
        Returns:
            List of lessons, most recent first
        """
        k = k or self.max_lessons
        
        if lesson_type == LessonType.SOLUTION:
            lessons = self.solution_lessons[-k:]
        elif lesson_type == LessonType.DEBUG:
            lessons = self.debug_lessons[-k:]
        else:
            # Combine both, sorted by timestamp
            all_lessons = self.solution_lessons + self.debug_lessons
            all_lessons.sort(key=lambda x: x.timestamp, reverse=True)
            lessons = all_lessons[:k]
        
        return list(reversed(lessons))  # Most recent first
    
    def format_for_prompt(
        self,
        lesson_type: Optional[LessonType] = None,
        k: Optional[int] = None
    ) -> str:
        """
        Format lessons for inclusion in agent prompts.
        
        Returns:
            Formatted string of lessons
        """
        lessons = self.get_recent_lessons(lesson_type, k)
        
        if not lessons:
            return "No lessons available yet."
        
        formatted = []
        for lesson in lessons:
            formatted.append(lesson.format_for_prompt())
            formatted.append("")  # Empty line between lessons
        
        return "\n".join(formatted)
    
    def get_statistics(self) -> dict:
        """Get pool statistics"""
        return {
            "solution_lessons": len(self.solution_lessons),
            "debug_lessons": len(self.debug_lessons),
            "total_lessons": len(self.solution_lessons) + len(self.debug_lessons),
            "total_added": self.total_added,
            "total_deduplicated": self.total_deduplicated,
        }
    
    def generate_lesson_id(self, lesson_type: LessonType) -> str:
        """Generate unique lesson ID"""
        self._lesson_counter[lesson_type] += 1
        count = self._lesson_counter[lesson_type]
        
        if lesson_type == LessonType.SOLUTION:
            return f"lesson_{count:05d}"
        else:
            return f"lesson_debug_{count:05d}"
    
    def save_to_file(self, filepath: Path):
        """Save all lessons to JSON file"""
        data = {
            "solution_lessons": [l.to_dict() for l in self.solution_lessons],
            "debug_lessons": [l.to_dict() for l in self.debug_lessons],
            "statistics": self.get_statistics(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Lessons saved to {filepath}")
    
    def load_from_file(self, filepath: Path):
        """Load lessons from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load solution lessons
        for lesson_data in data.get("solution_lessons", []):
            lesson = SolutionLesson.from_dict(lesson_data)
            self.solution_lessons.append(lesson)
        
        # Load debug lessons
        for lesson_data in data.get("debug_lessons", []):
            lesson = DebugLesson.from_dict(lesson_data)
            self.debug_lessons.append(lesson)
        
        print(f"Loaded {len(self.solution_lessons)} solution lessons")
        print(f"Loaded {len(self.debug_lessons)} debug lessons")
    
    def clear(self):
        """Clear all lessons"""
        self.solution_lessons.clear()
        self.debug_lessons.clear()
        self._lesson_counter.clear()
        self.total_added = 0
        self.total_deduplicated = 0