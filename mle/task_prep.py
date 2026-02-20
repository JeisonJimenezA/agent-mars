# mle/task_prep.py
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import json

from llm.llm_client import get_client
from llm.prompt_manager import get_prompt_manager
from core.config import Config

class TaskPreparation:
    """
    Prepares MLE tasks by extracting metadata and setting up the environment.
    Implements task preparation from Section 4.5.
    """
    
    def __init__(self, task_dir: Path):
        self.task_dir = Path(task_dir)
        self.client = get_client()
        self.prompt_manager = get_prompt_manager()
        
        self.metadata_dir = self.task_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Task information
        self.metric_name: Optional[str] = None
        self.lower_is_better: bool = False
        self.train_path: Optional[Path] = None
        self.test_path: Optional[Path] = None
        self.validation_path: Optional[Path] = None
    
    def extract_metric(self, problem_description: str) -> Tuple[str, bool]:
        """
        Extract evaluation metric from problem description.
        
        Args:
            problem_description: Task description
            
        Returns:
            Tuple of (metric_name, lower_is_better)
        """
        print("\n[Task Prep] Extracting metric information...")
        
        prompt = self.prompt_manager.get_prompt(
            "metric_parsing",
            problem_description=problem_description
        )
        
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from task descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=8000
        )
        
        # Parse JSON response
        content = response["content"]
        
        # Extract JSON
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                metric_name = data.get("metric_name", "accuracy")
                lower_is_better = data.get("lower_is_better", False)
                
                self.metric_name = metric_name
                self.lower_is_better = lower_is_better
                
                print(f"Metric: {metric_name}")
                print(f"Lower is better: {lower_is_better}")
                
                return metric_name, lower_is_better
            except json.JSONDecodeError:
                pass
        
        # Fallback
        print("Could not parse metric, using default: accuracy")
        self.metric_name = "accuracy"
        self.lower_is_better = False
        return "accuracy", False
    
    def prepare_data_splits(
        self,
        train_csv: Path,
        test_csv: Path,
        target_column: str = "Survived",
        validation_ratio: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Path]:
        """
        Prepare train/val/test splits and save metadata.
        
        Args:
            train_csv: Path to training CSV
            test_csv: Path to test CSV
            target_column: Name of target column
            validation_ratio: Ratio for validation split
            random_state: Random seed
            
        Returns:
            Dict with paths to metadata files
        """
        print("\n[Task Prep] Preparing data splits...")
        
        # Load data
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        print(f"  Training samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")
        
        # Check if target exists
        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in training data")
        
        # Create stratified validation split
        from sklearn.model_selection import train_test_split
        
        train_split, val_split = train_test_split(
            train_df,
            test_size=validation_ratio,
            random_state=random_state,
            stratify=train_df[target_column]
        )
        
        print(f"  Train split: {len(train_split)}")
        print(f"  Val split: {len(val_split)}")
        
        # Save splits to metadata
        train_meta_path = self.metadata_dir / "train.csv"
        val_meta_path = self.metadata_dir / "val.csv"
        test_meta_path = self.metadata_dir / "test.csv"
        
        train_split.to_csv(train_meta_path, index=False)
        val_split.to_csv(val_meta_path, index=False)
        test_df.to_csv(test_meta_path, index=False)
        
        print(f"Saved train metadata: {train_meta_path}")
        print(f"Saved val metadata: {val_meta_path}")
        print(f"Saved test metadata: {test_meta_path}")
        
        # Store paths
        self.train_path = train_meta_path
        self.validation_path = val_meta_path
        self.test_path = test_meta_path
        
        # Save split info
        split_info = {
            "train_samples": len(train_split),
            "val_samples": len(val_split),
            "test_samples": len(test_df),
            "target_column": target_column,
            "validation_ratio": validation_ratio,
            "random_state": random_state,
        }
        
        info_path = self.metadata_dir / "split_info.json"
        with open(info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        return {
            "train": train_meta_path,
            "val": val_meta_path,
            "test": test_meta_path,
            "info": info_path
        }
    
    def generate_documentation(self, data_description: str = "") -> str:
        """
        Generate documentation about the data.
        
        Args:
            data_description: Optional description of the data
            
        Returns:
            Documentation string
        """
        print("\n[Task Prep] Generating documentation...")
        
        # Load metadata
        if not self.train_path or not self.train_path.exists():
            return "No data available for documentation."
        
        train_df = pd.read_csv(self.train_path)
        
        # Generate basic statistics
        doc = "# Dataset Documentation\n\n"
        doc += f"## Overview\n"
        doc += f"- Training samples: {len(train_df)}\n"
        doc += f"- Features: {len(train_df.columns)}\n"
        doc += f"- Target: {self.metric_name}\n\n"
        
        doc += f"## Columns\n"
        for col in train_df.columns:
            dtype = train_df[col].dtype
            missing = train_df[col].isnull().sum()
            doc += f"- **{col}** ({dtype}): {missing} missing values\n"
        
        if data_description:
            doc += f"\n## Description\n{data_description}\n"
        
        # Save documentation
        doc_path = self.metadata_dir / "documentation.md"
        with open(doc_path, 'w') as f:
            f.write(doc)
        
        print(f"Saved documentation: {doc_path}")
        
        return doc