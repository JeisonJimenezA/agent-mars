# execution/executor.py
import subprocess
import time
import signal
import os
import sys
import re
import threading
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import psutil

from core.tree_node import ExecutionResult
from core.config import Config

# Project root directory (where env folder is located)
PROJECT_ROOT = Path(__file__).parent.parent

# Mapping from Python module names to pip package names
# Some packages have different import names than their pip names
MODULE_TO_PACKAGE = {
    # Deep Learning / Neural Networks
    "pytorch_tabnet": "pytorch-tabnet",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "transformers": "transformers",

    # ML Libraries
    "sklearn": "scikit-learn",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "imblearn": "imbalanced-learn",

    # Computer Vision
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",

    # NLP
    "spacy": "spacy",
    "nltk": "nltk",
    "gensim": "gensim",

    # Data Processing
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "bs4": "beautifulsoup4",
    "category_encoders": "category-encoders",
    "featuretools": "featuretools",

    # Time Series
    "tsfresh": "tsfresh",
    "prophet": "prophet",
    "statsmodels": "statsmodels",

    # Hyperparameter Optimization
    "optuna": "optuna",
    "hyperopt": "hyperopt",

    # Explainability
    "shap": "shap",
    "lime": "lime",
    "eli5": "eli5",

    # MLOps / Tracking
    "wandb": "wandb",
    "mlflow": "mlflow",

    # Visualization
    "plotly": "plotly",
    "seaborn": "seaborn",

    # Other common packages
    "tqdm": "tqdm",
    "joblib": "joblib",
    "numba": "numba",
    "scipy": "scipy",
    "pandas": "pandas",
    "numpy": "numpy",
}

@dataclass
class ExecutionConfig:
    """Configuration for code execution"""
    timeout: int = 10800  # 2 hours default
    max_memory_mb: int = 30000  # 30GB
    capture_output: bool = True
    working_dir: Optional[Path] = None
    env_vars: Optional[Dict[str, str]] = None
    use_venv: bool = True  # Use project's virtual environment

class Executor:
    """
    Executes Python code in isolated environment with resource limits.
    Implements execution with timeout and monitoring.
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.current_process: Optional[subprocess.Popen] = None
        self._python_executable = self._find_python_executable()

    def _find_python_executable(self) -> str:
        """
        Find the Python executable to use.
        Prioritizes the project's virtual environment (env folder).

        Returns:
            Path to Python executable
        """
        if not self.config.use_venv:
            return sys.executable

        # Check for virtual environment in project root
        venv_paths = [
            PROJECT_ROOT / "env" / "Scripts" / "python.exe",  # Windows
            PROJECT_ROOT / "env" / "bin" / "python",          # Linux/Mac
            PROJECT_ROOT / "venv" / "Scripts" / "python.exe", # Windows alt
            PROJECT_ROOT / "venv" / "bin" / "python",         # Linux/Mac alt
        ]

        for venv_python in venv_paths:
            if venv_python.exists():
                print(f"[Executor] Using virtual environment: {venv_python}")
                return str(venv_python)

        # Fallback to current Python
        print(f"[Executor] No venv found, using system Python: {sys.executable}")
        return sys.executable
    
    def execute_python_file(
        self,
        filepath: Path,
        timeout: Optional[int] = None,
        working_dir: Optional[Path] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute a Python file with resource monitoring.
        
        Args:
            filepath: Path to Python file
            timeout: Execution timeout in seconds
            working_dir: Working directory for execution
            
        Returns:
            ExecutionResult with execution details
        """
        timeout = timeout or self.config.timeout
        working_dir = working_dir or self.config.working_dir or filepath.parent
        
        print(f"\n[Executor] Running: {filepath}")
        print(f"  Working dir: {working_dir}")
        print(f"  Timeout: {timeout}s")
        
        start_time = time.time()
        
        # Prepare environment
        env = os.environ.copy()
        if self.config.env_vars:
            env.update(self.config.env_vars)
        if env_vars:
            env.update(env_vars)

        # Add Python path
        env['PYTHONPATH'] = str(working_dir)
        # Disable output buffering so training progress prints in real time
        env['PYTHONUNBUFFERED'] = '1'
        
        try:
            # Start process using virtual environment Python
            self.current_process = subprocess.Popen(
                [self._python_executable, "-u", str(filepath)],
                stdout=subprocess.PIPE if self.config.capture_output else None,
                stderr=subprocess.PIPE if self.config.capture_output else None,
                cwd=str(working_dir),
                env=env,
                text=True,
                bufsize=1,
            )

            stdout_lines: List[str] = []
            stderr_lines: List[str] = []

            def _stream(pipe, lines: List[str], prefix: str):
                for line in pipe:
                    lines.append(line)
                    print(f"{prefix}{line}", end="", flush=True)

            if self.config.capture_output:
                t_out = threading.Thread(target=_stream, args=(self.current_process.stdout, stdout_lines, ""), daemon=True)
                t_err = threading.Thread(target=_stream, args=(self.current_process.stderr, stderr_lines, "[STDERR] "), daemon=True)
                t_out.start()
                t_err.start()

            # Wait for process with timeout
            try:
                self.current_process.wait(timeout=timeout)
                if self.config.capture_output:
                    t_out.join()
                    t_err.join()
                return_code = self.current_process.returncode

                stdout = "".join(stdout_lines)
                stderr = "".join(stderr_lines)
                execution_time = time.time() - start_time

                success = return_code == 0
                metric_value = self._extract_metric_from_output(stdout)
                validation_metric_valid = metric_value is not None

                result = ExecutionResult(
                    success=success,
                    metric_value=metric_value,
                    execution_time=execution_time,
                    stdout=stdout,
                    stderr=stderr,
                    error_message="" if success else f"Exit code: {return_code}",
                    validation_metric_valid=validation_metric_valid
                )

                print(f"\n  ✓ Completed in {execution_time:.1f}s")
                if metric_value:
                    print(f"  ✓ Metric: {metric_value:.6f}")

                return result

            except subprocess.TimeoutExpired:
                # Kill process if timeout
                self._kill_process_tree(self.current_process.pid)
                if self.config.capture_output:
                    t_out.join(timeout=2)
                    t_err.join(timeout=2)

                execution_time = time.time() - start_time

                return ExecutionResult(
                    success=False,
                    metric_value=None,
                    execution_time=execution_time,
                    stdout="".join(stdout_lines),
                    stderr="".join(stderr_lines),
                    error_message=f"Execution timeout after {timeout}s",
                    validation_metric_valid=False
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                metric_value=None,
                execution_time=execution_time,
                stdout="",
                stderr="",
                error_message=f"Execution error: {str(e)}",
                validation_metric_valid=False
            )
        
        finally:
            self.current_process = None
    
    def _extract_metric_from_output(self, stdout: str) -> Optional[float]:
        """
        Extract validation metric from stdout using multiple strategies.

        Strategy 1 (primary): canonical pattern "Final Validation Metric: X"
        Strategy 2: common metric label patterns (Score, Accuracy, F1, AUC, etc.)
        Strategy 3: heuristic — last reasonable float in last 300 chars of output
        """
        if not stdout:
            return None

        # ── Strategy 1: canonical pattern ──────────────────────────────
        match = re.search(r'Final\s+Validation\s+Metric\s*[:=]\s*([\d.eE+\-]+)',
                          stdout, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        # ── Strategy 2: labelled metric patterns ───────────────────────
        labelled_patterns = [
            r'(?:val(?:idation)?|best|final)\s+(?:score|metric|loss|accuracy|acc|auc|f1|mae|mse|rmse|r2)\s*[:=]\s*([\d.eE+\-]+)',
            r'(?:score|metric|loss|accuracy|acc|auc|f1|mae|mse|rmse|r2)\s*[:=]\s*([\d.eE+\-]+)',
            r'(?:cv|cross.?val)\s+(?:score|mean)\s*[:=]\s*([\d.eE+\-]+)',
        ]
        for pat in labelled_patterns:
            # Search the last 2000 chars to prefer final values
            tail = stdout[-2000:]
            m = None
            for candidate in re.finditer(pat, tail, re.IGNORECASE):
                m = candidate  # keep last match
            if m:
                try:
                    v = float(m.group(1))
                    if not (v != v):  # not NaN
                        return v
                except ValueError:
                    pass

        # ── Strategy 3: heuristic — last plausible float ───────────────
        # Scan the last 300 chars for floats that look like a metric
        tail = stdout[-300:]
        floats = re.findall(r'\b(0\.\d+|\d+\.\d+)\b', tail)
        if floats:
            # Prefer values in common metric ranges [0, 1] or small positives
            candidates = [float(f) for f in floats]
            in_range = [v for v in candidates if 0.0 <= v <= 1.0]
            if in_range:
                return in_range[-1]  # Last one printed wins

        return None
    
    def _kill_process_tree(self, pid: int):
        """
        Kill process and all its children.
        
        Args:
            pid: Process ID to kill
        """
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # Kill children first
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            
            # Kill parent
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass
            
            # Wait for termination
            psutil.wait_procs(children + [parent], timeout=3)
            
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            print(f"  Warning: Error killing process tree: {e}")
    
    def cancel_execution(self):
        """Cancel currently running execution"""
        if self.current_process and self.current_process.poll() is None:
            print("\n[Executor] Cancelling execution...")
            self._kill_process_tree(self.current_process.pid)

    def detect_missing_modules(self, stderr: str) -> List[str]:
        """
        Detect missing modules from error output.

        Args:
            stderr: Standard error output from execution

        Returns:
            List of missing module names
        """
        missing_modules = []

        # Pattern 1: ModuleNotFoundError: No module named 'xyz'
        pattern1 = r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]"
        matches1 = re.findall(pattern1, stderr)
        missing_modules.extend(matches1)

        # Pattern 2: ImportError: No module named xyz
        pattern2 = r"ImportError: No module named ['\"]?([^\s'\"]+)['\"]?"
        matches2 = re.findall(pattern2, stderr)
        missing_modules.extend(matches2)

        # Pattern 3: ModuleNotFoundError for submodules like 'pytorch_tabnet.tab_model'
        # Extract just the top-level module
        cleaned = []
        for mod in missing_modules:
            # Get top-level module (before first dot)
            top_level = mod.split('.')[0]
            if top_level and top_level not in cleaned:
                cleaned.append(top_level)

        return cleaned

    def get_pip_executable(self) -> str:
        """
        Get the pip executable for the virtual environment.

        Returns:
            Path to pip executable
        """
        # Check for virtual environment in project root
        pip_paths = [
            PROJECT_ROOT / "env" / "Scripts" / "pip.exe",  # Windows
            PROJECT_ROOT / "env" / "bin" / "pip",          # Linux/Mac
            PROJECT_ROOT / "venv" / "Scripts" / "pip.exe", # Windows alt
            PROJECT_ROOT / "venv" / "bin" / "pip",         # Linux/Mac alt
        ]

        for pip_path in pip_paths:
            if pip_path.exists():
                return str(pip_path)

        # Fallback to system pip
        return "pip"

    def install_package(self, package_name: str, timeout: int = 300) -> Tuple[bool, str]:
        """
        Install a package using pip in the virtual environment.

        Args:
            package_name: Name of the package to install (pip name)
            timeout: Installation timeout in seconds

        Returns:
            Tuple of (success, message)
        """
        pip_exe = self.get_pip_executable()

        print(f"[Executor] Installing package: {package_name}")
        print(f"  Using pip: {pip_exe}")

        try:
            result = subprocess.run(
                [pip_exe, "install", package_name],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode == 0:
                print(f"  [OK] Successfully installed {package_name}")
                return True, f"Successfully installed {package_name}"
            else:
                error_msg = result.stderr or result.stdout
                print(f"  [FAIL] Failed to install {package_name}: {error_msg[:200]}")
                return False, f"Failed to install {package_name}: {error_msg[:500]}"

        except subprocess.TimeoutExpired:
            print(f"  [FAIL] Installation timeout for {package_name}")
            return False, f"Installation timeout for {package_name}"
        except Exception as e:
            print(f"  [FAIL] Installation error: {e}")
            return False, f"Installation error: {e}"

    def install_missing_modules(self, stderr: str) -> Tuple[bool, List[str]]:
        """
        Detect and install missing modules from error output.

        Args:
            stderr: Standard error output from execution

        Returns:
            Tuple of (any_installed, list_of_installed_packages)
        """
        missing_modules = self.detect_missing_modules(stderr)

        if not missing_modules:
            return False, []

        print(f"[Executor] Detected missing modules: {missing_modules}")

        installed = []
        for module in missing_modules:
            # Map module name to pip package name
            package_name = MODULE_TO_PACKAGE.get(module, module)

            success, msg = self.install_package(package_name)
            if success:
                installed.append(package_name)

        return len(installed) > 0, installed

    def execute_with_auto_install(
        self,
        filepath: Path,
        timeout: Optional[int] = None,
        working_dir: Optional[Path] = None,
        max_install_attempts: int = 3,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute a Python file with automatic dependency installation.

        If execution fails due to missing modules, installs them and retries.

        Args:
            filepath: Path to Python file
            timeout: Execution timeout in seconds
            working_dir: Working directory for execution
            max_install_attempts: Maximum number of install+retry cycles

        Returns:
            ExecutionResult with execution details
        """
        install_attempts = 0

        while install_attempts < max_install_attempts:
            # Execute the file
            result = self.execute_python_file(filepath, timeout, working_dir, env_vars)

            # If successful or no missing module error, return
            if result.success:
                return result

            # Check for missing modules
            error_output = result.stderr or result.error_message or ""
            if "ModuleNotFoundError" not in error_output and "ImportError" not in error_output:
                # Not a missing module error, return as-is
                return result

            # Try to install missing modules
            any_installed, installed_packages = self.install_missing_modules(error_output)

            if not any_installed:
                # Could not install any packages, return the error
                print(f"[Executor] Could not install missing packages, returning error")
                return result

            install_attempts += 1
            print(f"[Executor] Installed {installed_packages}, retrying execution (attempt {install_attempts}/{max_install_attempts})")

        # Max attempts reached, return last result
        return result