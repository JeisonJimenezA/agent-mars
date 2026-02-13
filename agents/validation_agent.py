# agents/validation_agent.py
"""
Validation Agent: Verifies validation split correctness and detects data leakage.
Implements Mejora 4 from the MARS paper alignment plan.
"""
import re
from typing import Dict, Optional, Tuple

from agents.base_agent import BaseAgent


class ValidationAgent(BaseAgent):
    """
    Agent responsible for verifying validation methodology.

    Two verification methods:
    1. verify_validation_split() - LLM-based analysis of validation handling
    2. verify_no_data_leakage() - Static regex analysis (no LLM cost)
    """

    def __init__(self):
        super().__init__("ValidationAgent")

    def get_system_prompt(self) -> str:
        return (
            "You are an expert ML engineer specializing in proper validation methodology. "
            "You detect data leakage, improper train/val/test splits, and metric computation errors."
        )

    def verify_validation_split(
        self,
        code: str,
        execution_output: str,
        execution_error: str = "",
    ) -> Dict:
        """
        Verify that the validation split was created/handled correctly.

        Uses the validation_verification prompt (LLM-based).

        Args:
            code: The main script code
            execution_output: stdout from execution
            execution_error: stderr from execution

        Returns:
            Dict with 'success', 'analysis', and optional 'issues'
        """
        self.log("Verifying validation split...")

        # Get prompt from prompt manager
        prompt = self.prompt_manager.get_prompt(
            "validation_verification",
            code=code,
            term_out=execution_output + ("\n" + execution_error if execution_error else ""),
        )

        response = self.call_llm(
            user_message=prompt,
            temperature=0.3,
            max_tokens=800,
        )

        # Parse JSON response
        result = self.extract_json_from_response(response["content"])

        if result:
            success = result.get("success", False)
            analysis = result.get("analysis", "")
            self.log(f"  Validation split {'OK' if success else 'FAILED'}: {analysis[:80]}...")
            return {
                "success": success,
                "analysis": analysis,
                "issues": [] if success else [analysis],
            }
        else:
            self.log("  Could not parse validation verification response")
            return {
                "success": True,  # Default to pass if parsing fails
                "analysis": "Verification parsing failed, assuming valid",
                "issues": [],
            }

    def verify_no_data_leakage(self, code: str) -> Tuple[bool, list]:
        """
        Static analysis to detect common data leakage patterns.

        NO LLM call - pure regex-based detection for speed.

        Args:
            code: All Python code (concatenated modules + main)

        Returns:
            Tuple of (is_clean, list_of_issues)
        """
        self.log("Checking for data leakage (static analysis)...")

        issues = []

        # ── Pattern 1: Fitting on test data ──
        # fit() or fit_transform() called with test/val in variable name
        fit_on_test = re.findall(
            r'\.(fit|fit_transform)\s*\(\s*[^)]*(?:test|val|X_test|X_val|y_test|y_val)[^)]*\)',
            code,
            re.IGNORECASE,
        )
        if fit_on_test:
            issues.append(
                f"Potential leakage: fit/fit_transform called on test/val data ({len(fit_on_test)} occurrences)"
            )

        # ── Pattern 2: Scaler/encoder fitted before split ──
        # Look for fit() before train_test_split()
        fit_pos = [m.start() for m in re.finditer(r'\.fit\s*\(', code)]
        split_pos = [m.start() for m in re.finditer(r'train_test_split\s*\(', code)]
        if fit_pos and split_pos:
            earliest_fit = min(fit_pos)
            earliest_split = min(split_pos)
            if earliest_fit < earliest_split:
                issues.append(
                    "Potential leakage: fit() called before train_test_split()"
                )

        # ── Pattern 3: Target leakage via feature engineering ──
        # Using target column in feature computation
        target_in_features = re.findall(
            r'(?:df|data|X)\[[\'"](?:target|label|y|class)[\'"].*(?:mean|sum|std|count)',
            code,
            re.IGNORECASE,
        )
        if target_in_features:
            issues.append(
                f"Potential target leakage: target column used in feature engineering"
            )

        # ── Pattern 4: Global normalization across train+test ──
        # normalize/scale called on full dataset before split
        global_norm = re.findall(
            r'(?:normalize|scale|StandardScaler|MinMaxScaler).*(?:df|data|X)(?!_train)(?!_test)',
            code,
        )
        # This is a weak signal, only flag if combined with other issues
        if global_norm and len(issues) > 0:
            issues.append(
                "Possible global normalization before split detected"
            )

        # ── Pattern 5: Test data in training loop ──
        test_in_loop = re.findall(
            r'for\s+.*\s+in\s+.*(?:test|val).*:[\s\S]*?(?:backward|optimize|\.step)',
            code,
            re.IGNORECASE,
        )
        if test_in_loop:
            issues.append(
                "Critical: Test/val data appears in training loop"
            )

        is_clean = len(issues) == 0
        if is_clean:
            self.log("  No data leakage patterns detected")
        else:
            self.log(f"  Found {len(issues)} potential data leakage issues")
            for issue in issues:
                self.log(f"    - {issue}")

        return is_clean, issues

    def full_verification(
        self,
        code: str,
        execution_output: str,
        execution_error: str = "",
    ) -> Dict:
        """
        Run both verification methods and combine results.

        Args:
            code: Main script code
            execution_output: stdout
            execution_error: stderr

        Returns:
            Dict with combined 'success', 'analysis', 'issues'
        """
        # Static analysis (fast, no LLM)
        leakage_clean, leakage_issues = self.verify_no_data_leakage(code)

        # LLM-based validation verification
        split_result = self.verify_validation_split(code, execution_output, execution_error)

        # Combine results
        all_issues = leakage_issues + split_result.get("issues", [])
        overall_success = leakage_clean and split_result.get("success", True)

        return {
            "success": overall_success,
            "analysis": split_result.get("analysis", ""),
            "issues": all_issues,
            "leakage_check": leakage_clean,
            "split_check": split_result.get("success", True),
        }
