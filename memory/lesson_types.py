# memory/lesson_types.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import json
import time

class LessonType(Enum):
    """Types of lessons"""
    SOLUTION = "solution"
    DEBUG = "debug"

@dataclass
class Lesson:
    """Base class for lessons learned during search"""
    id: str
    type: LessonType
    title: str
    timestamp: float = field(default_factory=time.time)
    source_node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "timestamp": self.timestamp,
            "source_node_id": self.source_node_id,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Lesson':
        """Create from dictionary"""
        lesson_type = LessonType(data["type"])
        
        if lesson_type == LessonType.SOLUTION:
            return SolutionLesson.from_dict(data)
        elif lesson_type == LessonType.DEBUG:
            return DebugLesson.from_dict(data)
        else:
            raise ValueError(f"Unknown lesson type: {lesson_type}")

@dataclass
class SolutionLesson(Lesson):
    """
    Lesson learned from comparing solutions (Section 4.3).
    
    Distilled from comparing:
    - New solution vs Current best solution
    - Identifies causal factors for performance changes
    """
    summary: str = ""
    empirical_findings: str = ""
    key_lesson: str = ""
    
    # Metrics comparison
    old_metric: Optional[float] = None
    new_metric: Optional[float] = None
    metric_delta: Optional[float] = None
    
    old_time: Optional[float] = None
    new_time: Optional[float] = None
    time_delta: Optional[float] = None
    
    def __post_init__(self):
        if self.type != LessonType.SOLUTION:
            self.type = LessonType.SOLUTION
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "summary": self.summary,
            "empirical_findings": self.empirical_findings,
            "key_lesson": self.key_lesson,
            "old_metric": self.old_metric,
            "new_metric": self.new_metric,
            "metric_delta": self.metric_delta,
            "old_time": self.old_time,
            "new_time": self.new_time,
            "time_delta": self.time_delta,
        })
        return base
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SolutionLesson':
        return cls(
            id=data["id"],
            type=LessonType.SOLUTION,
            title=data["title"],
            timestamp=data.get("timestamp", time.time()),
            source_node_id=data.get("source_node_id"),
            metadata=data.get("metadata", {}),
            summary=data.get("summary", ""),
            empirical_findings=data.get("empirical_findings", ""),
            key_lesson=data.get("key_lesson", ""),
            old_metric=data.get("old_metric"),
            new_metric=data.get("new_metric"),
            metric_delta=data.get("metric_delta"),
            old_time=data.get("old_time"),
            new_time=data.get("new_time"),
            time_delta=data.get("time_delta"),
        )
    
    def format_for_prompt(self) -> str:
        """Format lesson for inclusion in agent prompts"""
        text = f"[Lesson {self.id}] {self.title}\n"
        text += f"Summary: {self.summary}\n"
        text += f"Findings: {self.empirical_findings}\n"
        text += f"Key Lesson: {self.key_lesson}\n"
        
        if self.metric_delta is not None:
            sign = "+" if self.metric_delta >= 0 else ""
            text += f"Impact: {sign}{self.metric_delta:.4f} metric change\n"
        
        return text

@dataclass
class DebugLesson(Lesson):
    """
    Lesson learned from debugging errors (Section 4.3).
    
    Captures:
    - Root cause of the error
    - How it was fixed
    - How to detect similar errors in future
    """
    explanation: str = ""
    detection: str = ""
    fix_description: str = ""
    error_type: str = ""
    
    # Code context
    buggy_code_snippet: str = ""
    fixed_code_snippet: str = ""
    
    def __post_init__(self):
        if self.type != LessonType.DEBUG:
            self.type = LessonType.DEBUG
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "explanation": self.explanation,
            "detection": self.detection,
            "fix_description": self.fix_description,
            "error_type": self.error_type,
            "buggy_code_snippet": self.buggy_code_snippet,
            "fixed_code_snippet": self.fixed_code_snippet,
        })
        return base
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DebugLesson':
        return cls(
            id=data["id"],
            type=LessonType.DEBUG,
            title=data["title"],
            timestamp=data.get("timestamp", time.time()),
            source_node_id=data.get("source_node_id"),
            metadata=data.get("metadata", {}),
            explanation=data.get("explanation", ""),
            detection=data.get("detection", ""),
            fix_description=data.get("fix_description", ""),
            error_type=data.get("error_type", ""),
            buggy_code_snippet=data.get("buggy_code_snippet", ""),
            fixed_code_snippet=data.get("fixed_code_snippet", ""),
        )
    
    def format_for_prompt(self) -> str:
        """Format lesson for inclusion in agent prompts"""
        text = f"[Lesson {self.id}] {self.title}\n"
        text += f"Error Type: {self.error_type}\n"
        text += f"Explanation: {self.explanation}\n"
        text += f"Detection: {self.detection}\n"
        text += f"Fix: {self.fix_description}\n"
        return text