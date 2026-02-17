# utils/file_manager.py
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import subprocess

class FileManager:
    """
    Manages file operations for solution execution.
    Handles creating, copying, and managing solution files.
    """
    
    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    def create_solution_directory(self, node_id: str) -> Path:
        """
        Create a directory for a specific node's solution.
        
        Args:
            node_id: Unique node identifier
            
        Returns:
            Path to the created directory
        """
        solution_dir = self.working_dir / f"node_{node_id}"
        solution_dir.mkdir(parents=True, exist_ok=True)
        return solution_dir
    
    def write_files(self, solution_dir: Path, files: Dict[str, str]) -> List[Path]:
        """
        Write solution files to directory.
        
        Args:
            solution_dir: Directory to write files to
            files: Dictionary of {filename: content}
            
        Returns:
            List of created file paths
        """
        created_files = []
        
        for filename, content in files.items():
            filepath = solution_dir / filename
            
            # Create subdirectories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            created_files.append(filepath)
        
        return created_files
    
    def copy_metadata(self, solution_dir: Path, metadata_source: Path):
        """
        Copy metadata directory to solution directory.
        
        Args:
            solution_dir: Target directory
            metadata_source: Source metadata directory
        """
        if not metadata_source.exists():
            return
        
        target_metadata = solution_dir / "metadata"
        if target_metadata.exists():
            shutil.rmtree(target_metadata)
        
        shutil.copytree(metadata_source, target_metadata)
    
    def copy_input_data(self, solution_dir: Path, input_source: Path):
        """
        Copy input data files directly to solution directory root.

        Args:
            solution_dir: Target directory
            input_source: Source input data directory
        """
        if not input_source.exists():
            return

        # Copy CSV files directly to solution root for easy access
        # This matches the expected paths in generated code (train.csv, test.csv, etc.)
        for file_path in input_source.iterdir():
            if file_path.is_file():
                target_file = solution_dir / file_path.name
                if not target_file.exists():
                    shutil.copy2(file_path, target_file)

        # Also create input/ symlink for backward compatibility
        target_input = solution_dir / "input"
        if not target_input.exists():
            try:
                target_input.symlink_to(input_source.absolute(), target_is_directory=True)
            except OSError:
                # Fallback to copying if symlink fails (Windows)
                shutil.copytree(input_source, target_input)
    
    def create_working_subdirs(self, solution_dir: Path):
        """
        Create standard subdirectories for solution execution.
        
        Creates:
        - working/: For cached intermediate data
        - outputs/: For final outputs
        - submission/: For submission files
        """
        subdirs = ["working", "outputs", "submission"]
        for subdir in subdirs:
            (solution_dir / subdir).mkdir(exist_ok=True)
    
    def get_solution_files(self, solution_dir: Path) -> Dict[str, str]:
        """
        Read all Python files from solution directory.
        
        Args:
            solution_dir: Directory to read from
            
        Returns:
            Dictionary of {filename: content}
        """
        files = {}
        
        for filepath in solution_dir.rglob("*.py"):
            relative_path = filepath.relative_to(solution_dir)
            with open(filepath, 'r', encoding='utf-8') as f:
                files[str(relative_path)] = f.read()
        
        return files
    
    def cleanup_solution_directory(self, node_id: str):
        """
        Remove a solution directory.
        
        Args:
            node_id: Node identifier
        """
        solution_dir = self.working_dir / f"node_{node_id}"
        if solution_dir.exists():
            shutil.rmtree(solution_dir)
    
    def get_disk_usage(self, solution_dir: Path) -> int:
        """
        Get total disk usage of solution directory in bytes.
        
        Args:
            solution_dir: Directory to measure
            
        Returns:
            Total size in bytes
        """
        total = 0
        for dirpath, dirnames, filenames in os.walk(solution_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
        return total