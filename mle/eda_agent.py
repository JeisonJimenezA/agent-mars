# mle/eda_agent.py
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

from llm.llm_client import get_client
from llm.prompt_manager import get_prompt_manager

class EDAAgent:
    """
    Performs Exploratory Data Analysis on training data.
    Generates insights to guide feature engineering.
    """
    
    def __init__(self):
        self.client = get_client()
        self.prompt_manager = get_prompt_manager()
    
    def analyze_data(
        self,
        train_csv: Path,
        target_column: str = "Survived",
        problem_description: str = ""
    ) -> str:
        """
        Perform EDA and generate report.
        
        Args:
            train_csv: Path to training data
            target_column: Name of target column
            problem_description: Task description
            
        Returns:
            EDA report as string
        """
        print("\n[EDA] Analyzing training data...")
        
        # Load data
        df = pd.read_csv(train_csv)
        
        # Generate statistics
        report = self._generate_statistics(df, target_column)
        
        # For now, return the statistical report
        # In full implementation, would use LLM to generate insights
        
        print(f"  EDA complete ({len(report)} chars)")
        
        return report
    
    def _generate_statistics(self, df: pd.DataFrame, target_column: str) -> str:
        """Generate statistical report"""
        
        report = "# Exploratory Data Analysis Report\n\n"
        
        # Basic info
        report += f"## Dataset Overview\n"
        report += f"- Total samples: {len(df)}\n"
        report += f"- Total features: {len(df.columns)}\n"
        report += f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n\n"
        
        # Target distribution
        if target_column in df.columns:
            report += f"## Target Variable: {target_column}\n"
            target_counts = df[target_column].value_counts()
            report += f"Distribution:\n"
            for val, count in target_counts.items():
                pct = count / len(df) * 100
                report += f"  - {val}: {count} ({pct:.1f}%)\n"
            report += "\n"
        
        # Feature analysis
        report += "## Feature Analysis\n\n"
        
        # Numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if numeric_cols:
            report += "### Numerical Features\n"
            for col in numeric_cols:
                report += f"\n**{col}:**\n"
                report += f"  - Mean: {df[col].mean():.2f}\n"
                report += f"  - Std: {df[col].std():.2f}\n"
                report += f"  - Min: {df[col].min():.2f}\n"
                report += f"  - Max: {df[col].max():.2f}\n"
                report += f"  - Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)\n"
        
        # Categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            report += "\n### Categorical Features\n"
            for col in categorical_cols:
                unique = df[col].nunique()
                report += f"\n**{col}:**\n"
                report += f"  - Unique values: {unique}\n"
                report += f"  - Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)\n"
                
                if unique <= 10:
                    report += f"  - Value counts:\n"
                    for val, count in df[col].value_counts().head(5).items():
                        report += f"    - {val}: {count}\n"
        
        # Missing data summary
        report += "\n## Missing Data Summary\n"
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) > 0:
            for col, count in missing.items():
                pct = count / len(df) * 100
                report += f"  - {col}: {count} ({pct:.1f}%)\n"
        else:
            report += "  No missing data.\n"
        
        return report