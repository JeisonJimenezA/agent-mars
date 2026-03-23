# core/error_patterns.py
"""
Error Pattern Registry for MARS.

Provides quick pattern matching for common Python/ML errors.
Enables fast diagnosis without LLM calls for known error types.

Each pattern includes:
- regex: Pattern to match in error traceback
- error_type: Classification of the error
- quick_diagnosis: Human-readable explanation
- quick_fix: Suggested fix (can be applied automatically in some cases)
- auto_fixable: Whether the fix can be applied without LLM
- fix_function: Optional function to apply the fix
"""

from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import re


@dataclass
class ErrorPattern:
    """Definition of an error pattern with diagnosis and fix."""
    name: str
    regex: str
    error_type: str
    quick_diagnosis: str
    quick_fix: str
    auto_fixable: bool = False
    fix_function: Optional[Callable[[str, str], str]] = None
    priority: int = 0  # Higher = check first
    tags: List[str] = field(default_factory=list)  # ["dataclass", "import", etc.]


def fix_mutable_default(code: str, error: str) -> str:
    """Fix mutable default argument in dataclass."""
    # Pattern: field: List[X] = []
    pattern = r'(\w+):\s*(List\[[\w\[\], ]+\])\s*=\s*\[\]'
    replacement = r'\1: \2 = field(default_factory=list)'

    fixed = re.sub(pattern, replacement, code)

    # Ensure field is imported
    if 'field(default_factory' in fixed and 'from dataclasses import' in fixed:
        if 'field' not in re.search(r'from dataclasses import ([^)]+)', fixed).group(1):
            fixed = re.sub(
                r'from dataclasses import ([\w, ]+)',
                r'from dataclasses import \1, field',
                fixed
            )
    elif 'field(default_factory' in fixed and 'from dataclasses import' not in fixed:
        # Add import
        fixed = 'from dataclasses import field\n' + fixed

    return fixed


def fix_encoding_utf8(code: str, error: str) -> str:
    """Add encoding='utf-8' to open() calls."""
    # Pattern: open(file, 'r') or open(file, 'w') without encoding
    pattern = r"open\(([^)]+),\s*['\"]([rwa])['\"](?!\s*,\s*encoding)"
    replacement = r"open(\1, '\2', encoding='utf-8'"
    return re.sub(pattern, replacement, code)


def fix_remove_emojis(code: str, error: str) -> str:
    """Remove emojis from code."""
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F9FF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", code)


def fix_dict_default(code: str, error: str) -> str:
    """Fix mutable dict default in dataclass."""
    pattern = r'(\w+):\s*(Dict\[[\w\[\], ]+\])\s*=\s*\{\}'
    replacement = r'\1: \2 = field(default_factory=dict)'
    return re.sub(pattern, replacement, code)


# =============================================================================
# ERROR PATTERN REGISTRY
# =============================================================================

ERROR_PATTERNS: List[ErrorPattern] = [
    # -------------------------------------------------------------------------
    # DATACLASS ERRORS (very common)
    # -------------------------------------------------------------------------
    ErrorPattern(
        name="dataclass_mutable_list_default",
        regex=r"mutable default .* is not allowed|unhashable type: 'list'",
        error_type="ValueError",
        quick_diagnosis="Dataclass field has a mutable default value (list). "
                       "Python dataclasses require immutable defaults or field(default_factory=...).",
        quick_fix="Change `field: List[X] = []` to `field: List[X] = field(default_factory=list)`",
        auto_fixable=True,
        fix_function=fix_mutable_default,
        priority=100,
        tags=["dataclass", "mutable_default"],
    ),
    ErrorPattern(
        name="dataclass_mutable_dict_default",
        regex=r"mutable default .* is not allowed|unhashable type: 'dict'",
        error_type="ValueError",
        quick_diagnosis="Dataclass field has a mutable default value (dict). "
                       "Use field(default_factory=dict) instead.",
        quick_fix="Change `field: Dict[K,V] = {}` to `field: Dict[K,V] = field(default_factory=dict)`",
        auto_fixable=True,
        fix_function=fix_dict_default,
        priority=99,
        tags=["dataclass", "mutable_default"],
    ),

    # -------------------------------------------------------------------------
    # ENCODING ERRORS
    # -------------------------------------------------------------------------
    ErrorPattern(
        name="unicode_encode_charmap",
        regex=r"UnicodeEncodeError.*charmap.*can't encode",
        error_type="UnicodeEncodeError",
        quick_diagnosis="Code contains characters (likely emojis) that cannot be encoded "
                       "with the default Windows encoding.",
        quick_fix="Remove emojis from output strings or use UTF-8 encoding.",
        auto_fixable=True,
        fix_function=fix_remove_emojis,
        priority=95,
        tags=["encoding", "windows"],
    ),
    ErrorPattern(
        name="utf8_decode_error",
        regex=r"UnicodeDecodeError.*'utf-8' codec can't decode",
        error_type="UnicodeDecodeError",
        quick_diagnosis="File contains non-UTF-8 bytes. Usually a binary file "
                       "or a file with different encoding (latin-1, cp1252).",
        quick_fix="Try reading with encoding='latin-1' or 'cp1252', or use 'rb' mode.",
        auto_fixable=False,
        priority=90,
        tags=["encoding", "file_io"],
    ),

    # -------------------------------------------------------------------------
    # IMPORT ERRORS
    # -------------------------------------------------------------------------
    ErrorPattern(
        name="no_module_named",
        regex=r"ModuleNotFoundError: No module named '(\w+)'",
        error_type="ModuleNotFoundError",
        quick_diagnosis="Python cannot find the specified module. Either it's not installed "
                       "or the import path is wrong.",
        quick_fix="Install the module with pip install <module> or fix the import path.",
        auto_fixable=False,  # Handled by auto-install in executor
        priority=85,
        tags=["import", "dependency"],
    ),
    ErrorPattern(
        name="cannot_import_name",
        regex=r"ImportError: cannot import name '(\w+)' from '(\w+)'",
        error_type="ImportError",
        quick_diagnosis="The specified name doesn't exist in the module. "
                       "Check spelling or API version compatibility.",
        quick_fix="Verify the correct import name for your library version.",
        auto_fixable=False,
        priority=80,
        tags=["import"],
    ),
    ErrorPattern(
        name="circular_import",
        regex=r"ImportError: cannot import name.*\(most likely due to a circular import\)",
        error_type="ImportError",
        quick_diagnosis="Two modules are trying to import each other, creating a cycle.",
        quick_fix="Move the import inside the function that needs it, or restructure modules.",
        auto_fixable=False,
        priority=82,
        tags=["import", "architecture"],
    ),

    # -------------------------------------------------------------------------
    # FILE I/O ERRORS
    # -------------------------------------------------------------------------
    ErrorPattern(
        name="file_not_found",
        regex=r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'",
        error_type="FileNotFoundError",
        quick_diagnosis="The specified file or directory does not exist.",
        quick_fix="Check the file path. Use os.path.exists() to verify before accessing.",
        auto_fixable=False,
        priority=75,
        tags=["file_io"],
    ),
    ErrorPattern(
        name="permission_denied",
        regex=r"PermissionError: \[Errno 13\] Permission denied",
        error_type="PermissionError",
        quick_diagnosis="No permission to read/write the file or directory.",
        quick_fix="Check file permissions or try a different location.",
        auto_fixable=False,
        priority=70,
        tags=["file_io", "permissions"],
    ),

    # -------------------------------------------------------------------------
    # TYPE ERRORS
    # -------------------------------------------------------------------------
    ErrorPattern(
        name="none_type_not_subscriptable",
        regex=r"TypeError: 'NoneType' object is not subscriptable",
        error_type="TypeError",
        quick_diagnosis="Trying to index (e.g., x[0]) a variable that is None. "
                       "Usually means a function returned None unexpectedly.",
        quick_fix="Add a None check before indexing, or fix the function that returns None.",
        auto_fixable=False,
        priority=65,
        tags=["none", "indexing"],
    ),
    ErrorPattern(
        name="none_type_not_iterable",
        regex=r"TypeError: 'NoneType' object is not iterable",
        error_type="TypeError",
        quick_diagnosis="Trying to iterate over a variable that is None.",
        quick_fix="Check if the variable is None before iterating, or return empty list instead of None.",
        auto_fixable=False,
        priority=64,
        tags=["none", "iteration"],
    ),
    ErrorPattern(
        name="unsupported_operand",
        regex=r"TypeError: unsupported operand type\(s\) for ([^:]+): '(\w+)' and '(\w+)'",
        error_type="TypeError",
        quick_diagnosis="Mathematical operation between incompatible types.",
        quick_fix="Convert types before operation (int(), float(), str()).",
        auto_fixable=False,
        priority=60,
        tags=["types", "operators"],
    ),

    # -------------------------------------------------------------------------
    # ATTRIBUTE ERRORS
    # -------------------------------------------------------------------------
    ErrorPattern(
        name="attribute_error_none",
        regex=r"AttributeError: 'NoneType' object has no attribute '(\w+)'",
        error_type="AttributeError",
        quick_diagnosis="Calling a method on None. The variable is None when it shouldn't be.",
        quick_fix="Check why the variable is None and add proper None handling.",
        auto_fixable=False,
        priority=68,
        tags=["none", "attribute"],
    ),
    ErrorPattern(
        name="attribute_error_module",
        regex=r"AttributeError: module '(\w+)' has no attribute '(\w+)'",
        error_type="AttributeError",
        quick_diagnosis="The module doesn't have the specified attribute. "
                       "Could be API change or wrong import.",
        quick_fix="Check the module's documentation for the correct attribute name.",
        auto_fixable=False,
        priority=55,
        tags=["module", "attribute"],
    ),

    # -------------------------------------------------------------------------
    # VALUE ERRORS
    # -------------------------------------------------------------------------
    ErrorPattern(
        name="value_error_unpack",
        regex=r"ValueError: (not enough|too many) values to unpack",
        error_type="ValueError",
        quick_diagnosis="Tuple/list unpacking mismatch - wrong number of variables.",
        quick_fix="Check the number of values being unpacked matches the variables.",
        auto_fixable=False,
        priority=50,
        tags=["unpacking"],
    ),
    ErrorPattern(
        name="value_error_invalid_literal",
        regex=r"ValueError: invalid literal for int\(\) with base \d+: '([^']*)'",
        error_type="ValueError",
        quick_diagnosis="Trying to convert a non-numeric string to int.",
        quick_fix="Validate input before conversion or handle the exception.",
        auto_fixable=False,
        priority=48,
        tags=["conversion"],
    ),

    # -------------------------------------------------------------------------
    # ML-SPECIFIC ERRORS
    # -------------------------------------------------------------------------
    ErrorPattern(
        name="cuda_out_of_memory",
        regex=r"CUDA out of memory|OutOfMemoryError.*CUDA",
        error_type="CUDAError",
        quick_diagnosis="GPU memory exhausted. Model or batch size too large.",
        quick_fix="Reduce batch_size, use gradient accumulation, or use a smaller model.",
        auto_fixable=False,
        priority=90,
        tags=["cuda", "memory", "ml"],
    ),
    ErrorPattern(
        name="shape_mismatch",
        regex=r"RuntimeError: .*shape.*mismatch|size mismatch|shapes.*not aligned",
        error_type="RuntimeError",
        quick_diagnosis="Tensor/array shape mismatch in operation.",
        quick_fix="Print shapes before operation to debug. Check transpose/reshape.",
        auto_fixable=False,
        priority=85,
        tags=["shapes", "ml"],
    ),
    ErrorPattern(
        name="nan_loss",
        regex=r"(loss|Loss).*(nan|NaN)|RuntimeError.*nan.*loss",
        error_type="RuntimeError",
        quick_diagnosis="Training loss became NaN. Usually gradient explosion.",
        quick_fix="Lower learning rate, add gradient clipping, or check for div-by-zero in data.",
        auto_fixable=False,
        priority=80,
        tags=["training", "ml"],
    ),
    ErrorPattern(
        name="sklearn_not_fitted",
        regex=r"NotFittedError: This .* instance is not fitted yet",
        error_type="NotFittedError",
        quick_diagnosis="Sklearn estimator used before calling fit().",
        quick_fix="Call fit() on training data before predict/transform.",
        auto_fixable=False,
        priority=75,
        tags=["sklearn", "ml"],
    ),

    # -------------------------------------------------------------------------
    # PANDAS/DATA ERRORS
    # -------------------------------------------------------------------------
    ErrorPattern(
        name="key_error_column",
        regex=r"KeyError: ['\"](\w+)['\"]",
        error_type="KeyError",
        quick_diagnosis="Column/key not found in DataFrame/dict.",
        quick_fix="Check column names with df.columns. Verify spelling and case.",
        auto_fixable=False,
        priority=70,
        tags=["pandas", "dict", "data"],
    ),
    ErrorPattern(
        name="pyarrow_not_installed",
        regex=r"pyarrow|ArrowNotImplementedError|parquet.*not supported",
        error_type="ImportError",
        quick_diagnosis="pyarrow not installed but needed for parquet files.",
        quick_fix="Install pyarrow or use CSV format instead.",
        auto_fixable=False,
        priority=65,
        tags=["parquet", "data"],
    ),

    # -------------------------------------------------------------------------
    # SYNTAX ERRORS
    # -------------------------------------------------------------------------
    ErrorPattern(
        name="syntax_error_indent",
        regex=r"IndentationError: (unexpected indent|unindent)",
        error_type="IndentationError",
        quick_diagnosis="Incorrect indentation - mixing tabs and spaces or wrong level.",
        quick_fix="Fix indentation. Use 4 spaces consistently.",
        auto_fixable=False,
        priority=95,
        tags=["syntax"],
    ),
    ErrorPattern(
        name="syntax_error_colon",
        regex=r"SyntaxError: expected ':'",
        error_type="SyntaxError",
        quick_diagnosis="Missing colon after if/for/while/def/class statement.",
        quick_fix="Add ':' at the end of the statement.",
        auto_fixable=False,
        priority=93,
        tags=["syntax"],
    ),
]


class ErrorPatternMatcher:
    """
    Matches errors against known patterns for quick diagnosis.
    """

    def __init__(self):
        # Sort patterns by priority (highest first)
        self.patterns = sorted(ERROR_PATTERNS, key=lambda p: p.priority, reverse=True)
        # Compile regexes for performance
        self._compiled = [(p, re.compile(p.regex, re.IGNORECASE)) for p in self.patterns]

    def match(self, error_text: str) -> Optional[ErrorPattern]:
        """
        Find the first matching pattern for an error.

        Args:
            error_text: Error traceback text

        Returns:
            Matching ErrorPattern or None
        """
        for pattern, compiled_re in self._compiled:
            if compiled_re.search(error_text):
                return pattern
        return None

    def match_all(self, error_text: str) -> List[ErrorPattern]:
        """
        Find all matching patterns for an error.

        Args:
            error_text: Error traceback text

        Returns:
            List of matching ErrorPatterns
        """
        matches = []
        for pattern, compiled_re in self._compiled:
            if compiled_re.search(error_text):
                matches.append(pattern)
        return matches

    def get_diagnosis(self, error_text: str) -> Optional[Dict]:
        """
        Get diagnosis info for an error.

        Args:
            error_text: Error traceback text

        Returns:
            Dict with diagnosis info or None
        """
        pattern = self.match(error_text)
        if not pattern:
            return None

        return {
            "pattern_name": pattern.name,
            "error_type": pattern.error_type,
            "diagnosis": pattern.quick_diagnosis,
            "suggested_fix": pattern.quick_fix,
            "auto_fixable": pattern.auto_fixable,
            "tags": pattern.tags,
        }

    def try_auto_fix(
        self,
        error_text: str,
        code: str,
        target_file: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Attempt to automatically fix a known error pattern.

        Args:
            error_text: Error traceback
            code: Code to fix
            target_file: Optional filename for context

        Returns:
            Tuple of (success, fixed_code or None)
        """
        pattern = self.match(error_text)

        if not pattern or not pattern.auto_fixable or not pattern.fix_function:
            return False, None

        try:
            fixed = pattern.fix_function(code, error_text)
            if fixed != code:
                return True, fixed
            return False, None
        except Exception:
            return False, None

    def format_diagnosis_for_prompt(self, error_text: str) -> str:
        """
        Format diagnosis for inclusion in LLM prompt.

        Provides quick context about known error patterns.

        Args:
            error_text: Error traceback

        Returns:
            Formatted string for prompt injection
        """
        matches = self.match_all(error_text)

        if not matches:
            return ""

        lines = ["KNOWN ERROR PATTERN DETECTED:"]
        for pattern in matches[:3]:  # Max 3 patterns
            lines.append(f"- Type: {pattern.error_type}")
            lines.append(f"  Diagnosis: {pattern.quick_diagnosis}")
            lines.append(f"  Fix: {pattern.quick_fix}")
            lines.append("")

        return "\n".join(lines)


# Global instance
_matcher: Optional[ErrorPatternMatcher] = None


def get_error_matcher() -> ErrorPatternMatcher:
    """Get or create global error pattern matcher."""
    global _matcher
    if _matcher is None:
        _matcher = ErrorPatternMatcher()
    return _matcher
