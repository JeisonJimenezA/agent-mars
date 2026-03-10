# utils/file_manager.py
import os
import shutil
import sys
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
    
    def _create_directory_link(self, link_path: Path, target: Path):
        """
        Create a directory link (junction on Windows, symlink on Unix).
        Falls back to copying if linking fails.

        Args:
            link_path: Path where the link will be created
            target: Path to the target directory
        """
        target_abs = target.absolute()
        try:
            link_path.symlink_to(target_abs, target_is_directory=True)
            return
        except OSError:
            pass

        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link_path), str(target_abs)],
                    capture_output=True,
                    check=True,
                )
                return
            except Exception:
                pass

        # Final fallback: physical copy
        shutil.copytree(target_abs, link_path)

    def _link_file(self, source: Path, target: Path):
        """
        Create a hardlink from source to target.
        Falls back to copying if hardlink fails.

        Args:
            source: Existing file to link from
            target: Path for the new link
        """
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def copy_metadata(self, solution_dir: Path, metadata_source: Path):
        """
        Link metadata directory into solution directory (no data copy).
        Uses a junction on Windows or a symlink on Unix so that data is
        read directly from the original metadata path.

        Args:
            solution_dir: Target directory
            metadata_source: Source metadata directory
        """
        if not metadata_source.exists():
            return

        target_metadata = solution_dir / "metadata"
        if target_metadata.exists() or target_metadata.is_symlink():
            if target_metadata.is_symlink():
                target_metadata.unlink()
            elif sys.platform == "win32":
                # On Windows, junctions appear as dirs but rmtree fails on them.
                # os.rmdir removes the junction without deleting the target contents.
                try:
                    os.rmdir(target_metadata)
                except OSError:
                    shutil.rmtree(target_metadata)
            else:
                shutil.rmtree(target_metadata)

        self._create_directory_link(target_metadata, metadata_source)

    def copy_input_data(self, solution_dir: Path, input_source: Path):
        """
        No-op: raw input data is no longer placed inside node directories.
        Generated code reads directly from the original path via the DATA_DIR
        environment variable injected by the executor.
        """
        pass
    
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