# execution/validator.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

from core.tree_node import ExecutionResult
from utils.code_parser import CodeParser

class SolutionValidator:
    """
    Validates solutions before and after execution.
    Checks for common issues and requirements.
    """
    
    def __init__(self):
        self.parser = CodeParser()
    
    def validate_solution_structure(
        self,
        files: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate solution file structure.
        
        Checks:
        - Has main.py or runfile.py
        - All files have valid syntax
        - No obvious security issues
        
        Args:
            files: Dictionary of {filename: code}
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for main entry point
        main_files = ['main.py', 'runfile.py', 'run.py']
        has_main = any(f in files for f in main_files)
        
        if not has_main:
            issues.append("No main entry point found (main.py or runfile.py)")
        
        # Validate syntax for each file
        for filename, code in files.items():
            if not filename.endswith('.py'):
                continue
            
            is_valid, error = self.parser.validate_syntax(code)
            if not is_valid:
                issues.append(f"Syntax error in {filename}: {error}")
        
        # Check for dangerous operations
        dangerous_patterns = [
            (r'os\.system\(["\']rm\s+-rf', "Dangerous system command detected"),
            (r'shutil\.rmtree\(["\']/', "Dangerous file deletion detected"),
            (r'subprocess\.(run|call|Popen).*rm\s+-rf', "Dangerous subprocess command"),
        ]
        
        for filename, code in files.items():
            for pattern, message in dangerous_patterns:
                if re.search(pattern, code):
                    issues.append(f"{message} in {filename}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_execution_result(
        self,
        result: ExecutionResult,
        expected_outputs: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate execution result.
        
        Checks:
        - Metric was extracted correctly
        - Required output files were created
        - No critical errors in stderr
        
        Args:
            result: Execution result to validate
            expected_outputs: List of expected output files
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not result.success:
            issues.append(f"Execution failed: {result.error_message}")
        
        if result.success and result.metric_value is None:
            issues.append("No validation metric found in output")
        
        if result.success and not result.validation_metric_valid:
            issues.append("Validation metric is invalid")
        
        # Check for critical errors in stderr
        if result.stderr:
            critical_errors = [
                "CUDA out of memory",
                "MemoryError",
                "Segmentation fault",
                "core dumped"
            ]
            
            for error in critical_errors:
                if error in result.stderr:
                    issues.append(f"Critical error detected: {error}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def check_required_files(
        self,
        solution_dir: Path,
        required_files: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Check if required files exist in solution directory.
        
        Args:
            solution_dir: Solution directory path
            required_files: List of required file paths
            
        Returns:
            Tuple of (all_exist, list_of_missing)
        """
        missing = []
        
        for filename in required_files:
            filepath = solution_dir / filename
            if not filepath.exists():
                missing.append(filename)
        
        return len(missing) == 0, missing
    
    def extract_validation_metric(self, stdout: str) -> Optional[float]:
        """
        Extract validation metric from stdout.
        Implements the required format from paper.
        
        Args:
            stdout: Standard output text
            
        Returns:
            Metric value or None
        """
        if not stdout:
            return None
        
        # Pattern: "Final Validation Metric: <value>"
        pattern = r'Final Validation Metric:\s*([\d.]+)'
        match = re.search(pattern, stdout, re.IGNORECASE)
        
        if match:
            try:
                value = float(match.group(1))
                return value
            except ValueError:
                pass
        
        return None
    
    def check_data_leakage(self, code: str) -> List[str]:
        """
        Check for potential data leakage issues.
        
        Args:
            code: Python code to check
            
        Returns:
            List of potential issues
        """
        issues = []
        
        # Check for using test data in training
        leakage_patterns = [
            (r'pd\.read_csv\(["\'].*test.*["\'].*\).*\.fit\(', 
             "Potential data leakage: using test data in training"),
            (r'train.*test.*concat', 
             "Potential data leakage: concatenating train and test"),
        ]
        
        for pattern, message in leakage_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(message)
        
        return issues