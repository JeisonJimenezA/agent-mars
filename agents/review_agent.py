# agents/review_agent.py
from typing import Dict, Optional
from agents.base_agent import BaseAgent

class ReviewAgent(BaseAgent):
    """
    Agent responsible for reviewing execution results.
    Extracts empirical findings from execution logs.
    """
    
    def __init__(self):
        super().__init__("ReviewAgent")
    
    def get_system_prompt(self) -> str:
        return """You are an expert ML engineer specializing in analyzing experimental results. You excel at extracting actionable insights from execution logs."""
    
    def review_execution(
        self,
        problem_description: str,
        code: str,
        execution_output: str = "",
        execution_error: str = "",
    ) -> Optional[Dict]:
        """
        Review execution results and extract findings.
        Implements the ExecuteAndReview step from Algorithm 2.

        Returns dict with:
        - summary: Brief summary of execution
        - metric: Extracted metric value (float or None)
        - valid_metric: Whether metric is valid (bool)

        Args:
            problem_description: Task description
            code: Executed code (main script)
            execution_output: Standard output from execution
            execution_error: Standard error from execution

        Returns:
            Review dict or None
        """
        self.log("Reviewing execution results...")

        term_out = execution_output or ""
        if execution_error:
            term_out += "\n\nSTDERR:\n" + execution_error

        prompt = self.prompt_manager.get_prompt(
            "execution_review",
            problem_description=problem_description,
            code=code[:6000],
            term_out=term_out[-6000:],
        )

        response = self.call_llm(
            user_message=prompt,
            temperature=0,
            max_tokens=8000,
        )

        review = self.extract_json_from_response(response["content"])

        if review:
            self.log(f"Review complete: valid_metric={review.get('valid_metric', False)}")
            return review

        self.log("Failed to parse review response")
        return None