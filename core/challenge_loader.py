# core/challenge_loader.py
"""
Unified challenge loader - reads challenge context from text files
Supports multiple data formats:
- Simple CSV (train.csv, test.csv)
- JSON subjects with gold labels (subjects/*.json + gold.txt)
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
import json
import re

class ChallengeLoader:
    """
    Loads challenge information from text files.
    NO MODEL SUGGESTIONS - Pure context only.
    """
    
    def __init__(self, challenge_name: str, data_dir: Path):
        self.challenge_name = challenge_name
        self.data_dir = Path(data_dir)
        self.challenge_file = Path("challenges") / f"{challenge_name}.txt"
        
        if not self.challenge_file.exists():
            raise FileNotFoundError(f"Challenge file not found: {self.challenge_file}")
        
        # Load challenge description
        with open(self.challenge_file, 'r', encoding='utf-8') as f:
            self.description = f.read()
        
        # Parse key information
        self._parse_metadata()
    
    def _parse_metadata(self):
        """Extract metadata from description text"""
        
        # Extract metric info
        if "lower is better" in self.description.lower() or "minimize" in self.description.lower():
            self.metric_direction = "minimize"
        else:
            self.metric_direction = "maximize"
        
        # Extract metric name
        metric_match = re.search(r'Metric:\s*([^\n]+)', self.description)
        if metric_match:
            self.metric_name = metric_match.group(1).split('(')[0].strip()
        else:
            self.metric_name = "unknown"
        
        # Detect data format: JSON subjects or simple CSV
        self.data_format = self._detect_data_format()

        # Extract file names based on format
        if self.data_format == "json_subjects":
            self.train_file = None  # Will load from subjects/
            self.test_file = None
            self.gold_file = "gold.txt"
        else:
            train_match = re.search(r'train\.csv', self.description)
            test_match = re.search(r'test\.csv', self.description)
            self.train_file = "train.csv" if train_match else None
            self.test_file = "test.csv" if test_match else None

        # Extract target column
        if self.data_format == "json_subjects":
            # For JSON subjects, extract target column from gold file headers
            # Common patterns: Type, Risk, label
            type_match = re.search(r'Columns:\s*Subject\s*,\s*(\w+)', self.description)
            if type_match:
                self.target_column = type_match.group(1)
            elif "Type" in self.description:
                self.target_column = "Type"
            else:
                self.target_column = None
        else:
            target_match = re.search(r'target', self.description, re.IGNORECASE)
            self.target_column = "target" if target_match else None

    def _detect_data_format(self) -> str:
        """Detect if data is JSON subjects format or simple CSV"""
        # Check if subjects folder exists
        subjects_path = self.data_dir / "train" / "subjects"
        if subjects_path.exists() and subjects_path.is_dir():
            return "json_subjects"

        # Check description for hints
        if "subjects/" in self.description or "gold.txt" in self.description:
            return "json_subjects"

        return "csv"
    
    def get_problem_description(self) -> str:
        """
        Return the complete challenge description.
        This is the ONLY information given to agents.
        """
        return self.description
    
    def get_metric_info(self) -> Tuple[str, bool]:
        """
        Return (metric_name, lower_is_better)
        """
        lower_is_better = (self.metric_direction == "minimize")
        return self.metric_name, lower_is_better
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load train and test data based on detected format"""

        if self.data_format == "json_subjects":
            return self._load_json_subjects_data()
        else:
            return self._load_csv_data()

    def _load_csv_data(self) -> Dict[str, pd.DataFrame]:
        """Load simple CSV format (train.csv, test.csv)"""
        data = {}

        if self.train_file:
            train_path = self.data_dir / self.train_file
            if train_path.exists():
                data['train'] = pd.read_csv(train_path)
                print(f"  Loaded train: {len(data['train'])} samples")

        if self.test_file:
            test_path = self.data_dir / self.test_file
            if test_path.exists():
                data['test'] = pd.read_csv(test_path)
                print(f"  Loaded test: {len(data['test'])} samples")

        return data

    def _load_json_subjects_data(self) -> Dict[str, pd.DataFrame]:
        """Load JSON subjects format (subjects/*.json + gold.txt)"""
        data = {}

        for split in ['train', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue

            gold_path = split_dir / "gold.txt"
            subjects_dir = split_dir / "subjects"

            if not gold_path.exists() or not subjects_dir.exists():
                continue

            # Load gold labels
            gold_df = pd.read_csv(gold_path)
            print(f"  Loaded {split} gold: {len(gold_df)} subjects")

            # Load each subject's messages
            rows = []
            for _, row in gold_df.iterrows():
                subject_id = row['Subject']
                json_path = subjects_dir / f"{subject_id}.json"

                if json_path.exists():
                    text = self._load_subject_messages(json_path)
                else:
                    text = ""
                    print(f"  Warning: Missing {json_path}")

                # Build row with all gold columns + text
                new_row = row.to_dict()
                new_row['text'] = text
                rows.append(new_row)

            df = pd.DataFrame(rows)
            data[split] = df
            print(f"  Loaded {split}: {len(df)} samples")

        return data

    def _load_subject_messages(self, json_path: Path, separator: str = " [SEP] ") -> str:
        """
        Load messages from a subject JSON file and concatenate them.
        Handles null/NaN messages.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)

            # Extract valid message texts
            texts = []
            for msg in messages:
                text = msg.get('message')
                # Handle null, NaN, None
                if text is not None and text == text:  # NaN check: NaN != NaN
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())

            return separator.join(texts)

        except Exception as e:
            print(f"  Error loading {json_path}: {e}")
            return ""
    
    def prepare_splits(
        self,
        train_df: pd.DataFrame,
        metadata_dir: Path,
        validation_ratio: float = 0.2
    ) -> Dict[str, Path]:
        """Prepare train/validation splits"""
        
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect if classification (for stratification)
        if self.target_column and self.target_column in train_df.columns:
            target = train_df[self.target_column]
            
            # Check if classification
            n_unique = target.nunique()
            is_classification = n_unique < 50
            
            if is_classification:
                # Stratified split
                from sklearn.model_selection import train_test_split
                
                # Convert target to numeric if needed
                if target.dtype == 'object':
                    target_numeric = target.astype('category').cat.codes
                    train_df['_target_numeric'] = target_numeric
                    stratify_col = '_target_numeric'
                else:
                    stratify_col = self.target_column
                
                train_split, val_split = train_test_split(
                    train_df,
                    test_size=validation_ratio,
                    random_state=42,
                    stratify=train_df[stratify_col]
                )
                
                # Remove temp column
                if '_target_numeric' in train_split.columns:
                    train_split = train_split.drop('_target_numeric', axis=1)
                    val_split = val_split.drop('_target_numeric', axis=1)
            else:
                # Random split for regression
                from sklearn.model_selection import train_test_split
                train_split, val_split = train_test_split(
                    train_df,
                    test_size=validation_ratio,
                    random_state=42
                )
        else:
            # No target, just random split
            from sklearn.model_selection import train_test_split
            train_split, val_split = train_test_split(
                train_df,
                test_size=validation_ratio,
                random_state=42
            )
        
        # Save splits
        train_path = metadata_dir / "train.csv"
        val_path = metadata_dir / "val.csv"
        
        train_split.to_csv(train_path, index=False)
        val_split.to_csv(val_path, index=False)
        
        print(f"Train: {len(train_split)} | Val: {len(val_split)}")
        
        return {
            "train": train_path,
            "val": val_path
        }