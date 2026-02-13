# utils/code_parser.py
import re
import ast
from typing import Dict, List, Optional, Tuple

class CodeParser:
    """
    Utilities for parsing and analyzing Python code.
    """
    
    @staticmethod
    def extract_imports(code: str) -> List[str]:
        """
        Extract import statements from code.
        
        Args:
            code: Python code as string
            
        Returns:
            List of import statements
        """
        imports = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join([alias.name for alias in node.names])
                    imports.append(f"from {module} import {names}")
        except SyntaxError:
            # Fallback to regex if AST parsing fails
            import_pattern = r'^\s*(import|from)\s+[\w\.]+'
            for line in code.split('\n'):
                if re.match(import_pattern, line):
                    imports.append(line.strip())
        
        return imports
    
    @staticmethod
    def extract_functions(code: str) -> List[Dict[str, str]]:
        """
        Extract function definitions from code.
        
        Args:
            code: Python code as string
            
        Returns:
            List of dicts with 'name' and 'signature'
        """
        functions = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function signature
                    args = [arg.arg for arg in node.args.args]
                    signature = f"{node.name}({', '.join(args)})"
                    
                    functions.append({
                        "name": node.name,
                        "signature": signature,
                        "lineno": node.lineno
                    })
        except SyntaxError:
            pass
        
        return functions
    
    @staticmethod
    def extract_classes(code: str) -> List[Dict[str, str]]:
        """
        Extract class definitions from code.
        
        Args:
            code: Python code as string
            
        Returns:
            List of dicts with 'name' and 'methods'
        """
        classes = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                    
                    classes.append({
                        "name": node.name,
                        "methods": methods,
                        "lineno": node.lineno
                    })
        except SyntaxError:
            pass
        
        return classes
    
    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if code has valid Python syntax.
        
        Args:
            code: Python code as string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    @staticmethod
    def extract_code_from_markdown(text: str) -> List[str]:
        """
        Extract code blocks from markdown-formatted text.
        
        Args:
            text: Text containing markdown code blocks
            
        Returns:
            List of code blocks
        """
        # Match ```python or ``` code blocks
        pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[str]:
        """
        Extract JSON object from text (handles markdown wrapping).
        
        Args:
            text: Text containing JSON
            
        Returns:
            JSON string or None
        """
        # Try to find JSON in markdown code block
        json_pattern = r'```(?:json)?\n(\{.*?\})\n```'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            return match.group(1)
        
        # Try to find raw JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            return match.group(0)
        
        return None
    
    @staticmethod
    def count_lines(code: str) -> int:
        """Count non-empty lines in code"""
        return len([line for line in code.split('\n') if line.strip()])
    
    @staticmethod
    def extract_main_execution(code: str) -> Optional[str]:
        """
        Extract the main execution block from code.
        
        Looks for: if __name__ == "__main__":
        
        Args:
            code: Python code as string
            
        Returns:
            Main block code or None
        """
        pattern = r'if\s+__name__\s*==\s*["\']__main__["\']\s*:(.*?)(?=\n(?:def|class|\Z))'
        match = re.search(pattern, code, re.DOTALL)
        
        if match:
            return match.group(1)
        
        return None