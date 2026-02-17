# execution/executor.py
import subprocess
import time
import signal
import os
import sys
import re
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
    timeout: int = 3600  # 1 hour default
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
        working_dir: Optional[Path] = None
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
        
        # Add Python path
        env['PYTHONPATH'] = str(working_dir)
        
        try:
            # Start process using virtual environment Python
            self.current_process = subprocess.Popen(
                [self._python_executable, str(filepath)],
                stdout=subprocess.PIPE if self.config.capture_output else None,
                stderr=subprocess.PIPE if self.config.capture_output else None,
                cwd=str(working_dir),
                env=env,
                text=True,
                bufsize=1,
            )
            
            # Monitor execution
            try:
                stdout, stderr = self.current_process.communicate(timeout=timeout)
                return_code = self.current_process.returncode
                
                execution_time = time.time() - start_time
                
                # Check success
                success = return_code == 0
                
                # Extract validation metric if present
                metric_value = self._extract_metric_from_output(stdout)
                validation_metric_valid = metric_value is not None
                
                result = ExecutionResult(
                    success=success,
                    metric_value=metric_value,
                    execution_time=execution_time,
                    stdout=stdout or "",
                    stderr=stderr or "",
                    error_message="" if success else f"Exit code: {return_code}",
                    validation_metric_valid=validation_metric_valid
                )
                
                print(f"  ✓ Completed in {execution_time:.1f}s")
                if metric_value:
                    print(f"  ✓ Metric: {metric_value:.6f}")
                
                return result
                
            except subprocess.TimeoutExpired:
                # Kill process if timeout
                self._kill_process_tree(self.current_process.pid)
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=False,
                    metric_value=None,
                    execution_time=execution_time,
                    stdout="",
                    stderr="",
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
        Extract validation metric from stdout.
        
        Looks for pattern: "Final Validation Metric: <value>"
        
        Args:
            stdout: Standard output from execution
            
        Returns:
            Metric value or None
        """
        if not stdout:
            return None
        
        import re
        
        # Pattern from paper's requirement
        pattern = r'Final Validation Metric:\s*([\d.]+)'
        match = re.search(pattern, stdout, re.IGNORECASE)
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
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
        max_install_attempts: int = 3
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
            result = self.execute_python_file(filepath, timeout, working_dir)

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