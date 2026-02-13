# execution/executor.py
import subprocess
import time
import signal
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import psutil

from core.tree_node import ExecutionResult
from core.config import Config

@dataclass
class ExecutionConfig:
    """Configuration for code execution"""
    timeout: int = 3600  # 1 hour default
    max_memory_mb: int = 30000  # 30GB
    capture_output: bool = True
    working_dir: Optional[Path] = None
    env_vars: Optional[Dict[str, str]] = None

class Executor:
    """
    Executes Python code in isolated environment with resource limits.
    Implements execution with timeout and monitoring.
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.current_process: Optional[subprocess.Popen] = None
    
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
            # Start process
            self.current_process = subprocess.Popen(
                ['python', str(filepath)],
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