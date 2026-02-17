# agents/idea_agent.py
from typing import Optional, List, Dict
from enum import Enum
from pathlib import Path
import json

from agents.base_agent import BaseAgent
from agents.search_agent import SearchAgent
from memory.lesson_pool import LessonPool
from memory.lesson_types import LessonType
from core.config import Config


class CurriculumStage(Enum):
    """
    Curriculum stages for structured exploration (Mejora 3).

    BASELINE -> STANDARD -> ADVANCED -> ENSEMBLE

    Each stage guides the complexity and style of generated ideas.
    """
    BASELINE = "baseline"      # Ideas 0-1: Simple, proven approaches
    STANDARD = "standard"      # Ideas 2-3: Standard ML techniques
    ADVANCED = "advanced"      # Ideas 4-6: Advanced optimizations
    ENSEMBLE = "ensemble"      # Ideas 7+: Ensemble, stacking, meta-learning

class IdeaAgent(BaseAgent):
    """
    Agent responsible for generating solution ideas.
    Implements curriculum-based idea generation from Section 4.5.
    
    Now includes SOTA model discovery via SearchAgent.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__("IdeaAgent")
        self.generated_ideas: List[str] = []
        self.search_agent = SearchAgent()  # ← NEW: Web search capability
        self.discovered_models: List[Dict[str, str]] = []  # ← Cache discovered models

        # Curriculum exploration (Mejora 3)
        self.current_stage: CurriculumStage = CurriculumStage.BASELINE
        self.stage_history: List[CurriculumStage] = []

        # Persistent cache for discovered models
        self._cache_dir = cache_dir or Config.WORKING_DIR / Config.CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._models_cache_file = self._cache_dir / "discovered_models.json"
        self._load_models_cache()

    def _load_models_cache(self):
        """Load discovered models from cache."""
        if self._models_cache_file.exists():
            try:
                with open(self._models_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.discovered_models = data.get("models", [])
                    self.generated_ideas = data.get("ideas", [])
                    # Restore curriculum state
                    stage_name = data.get("current_stage", "baseline")
                    self.current_stage = CurriculumStage(stage_name)
                    self.log(f"Loaded {len(self.discovered_models)} models, {len(self.generated_ideas)} ideas from cache")
            except Exception as e:
                self.log(f"Failed to load models cache: {e}")

    def save_cache(self):
        """Save discovered models and ideas to cache."""
        try:
            with open(self._models_cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "models": self.discovered_models,
                    "ideas": self.generated_ideas,
                    "current_stage": self.current_stage.value,
                    "stage_history": [s.value for s in self.stage_history],
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"Failed to save models cache: {e}")
    
    def get_system_prompt(self) -> str:
        return """You are an expert Machine Learning Engineer specializing in developing innovative and efficient ML solutions. Your role is to propose practical, well-reasoned approaches to solve ML problems."""
    
    def generate_initial_idea(
        self,
        problem_description: str,
        eda_report: str,
        model_architectures: List[Dict[str, str]] = None  # ← Now optional
    ) -> Optional[str]:
        """
        Generate initial lightweight baseline idea.
        
        NOW INCLUDES: Automatic SOTA model discovery via web search.
        
        Args:
            problem_description: Task description
            eda_report: EDA insights
            model_architectures: Optional pre-discovered models (legacy)
            
        Returns:
            Natural language idea description
        """
        self.log("Generating initial baseline idea...")
        
        # ========================================
        # STEP 1: Discover SOTA Models (NEW!)
        # ========================================
        if not model_architectures:
            self.log("  Discovering SOTA models via search...")
            self.discovered_models = self.search_agent.search_sota_models(
                problem_description=problem_description,
                num_candidates=5
            )
            model_architectures = self.discovered_models
        else:
            self.discovered_models = model_architectures
        
        # ========================================
        # STEP 2: Select Lightweight Baseline
        # ========================================
        # Pick the simplest/most efficient model for initial baseline
        if model_architectures:
            # Heuristic: prefer models with "efficient", "light", or first in list
            baseline_model = self._select_baseline_model(model_architectures)
            self.log(f"  Selected baseline: {baseline_model['name']}")
        else:
            baseline_model = None
            self.log("  No models discovered, agent will propose generic approach")
        
        # ========================================
        # STEP 3: Generate Idea with Model Context
        # ========================================
        prompt = self._create_initial_idea_prompt(
            problem_description,
            eda_report,
            baseline_model,
            model_architectures
        )
        
        response = self.call_llm(
            user_message=prompt,
            temperature=0.7,
            max_tokens=1500
        )
        
        idea = response["content"].strip()
        self.generated_ideas.append(idea)
        
        self.log(f"Generated initial idea ({len(idea)} chars)")
        
        return idea
    
    def improve_idea(
        self,
        problem_description: str,
        eda_report: str,
        lesson_pool: LessonPool,
        previous_ideas: Optional[List[str]] = None,
        valid_solutions_count: int = 0,
        stagnation_count: int = 0,
    ) -> Optional[str]:
        """
        Generate improved idea based on lessons learned.

        Implements evolutionary growth using lesson pool.
        NOW INCLUDES:
        - Reference to discovered models for advanced techniques
        - Curriculum-based exploration (BASELINE -> STANDARD -> ADVANCED -> ENSEMBLE)

        Args:
            problem_description: Task description
            eda_report: EDA report
            lesson_pool: Pool of learned lessons
            previous_ideas: List of previously tried ideas
            valid_solutions_count: Number of valid solutions found so far
            stagnation_count: Number of iterations without improvement

        Returns:
            Natural language idea description
        """
        self.log("Generating improved idea from lessons...")

        # ══════════════════════════════════════════════════════════════
        # CURRICULUM STAGE DETERMINATION (Mejora 3)
        # ══════════════════════════════════════════════════════════════
        num_lessons = len(lesson_pool.solution_lessons)
        num_ideas = len(self.generated_ideas)
        self.current_stage = self._determine_curriculum_stage(
            num_lessons=num_lessons,
            valid_count=valid_solutions_count,
            stagnation=stagnation_count,
            num_ideas=num_ideas,
        )
        self.stage_history.append(self.current_stage)

        # Get curriculum guidance text
        curriculum_guidance = self._get_curriculum_guidance(self.current_stage)
        self.log(f"  Curriculum stage: {self.current_stage.value}")

        # Get recent solution lessons
        lessons_text = lesson_pool.format_for_prompt(
            lesson_type=LessonType.SOLUTION,
            k=15  # Use more lessons for improvement
        )

        # Format previous ideas
        prev_ideas_text = self._format_previous_ideas(previous_ideas)

        # Include discovered models for context
        models_context = self._format_models_context(self.discovered_models)

        # Get prompt with curriculum guidance injected
        prompt = self.prompt_manager.get_prompt(
            "idea_improvement",
            problem_description=problem_description,
            eda_report=eda_report[:500],  # Truncate for brevity
            previous_ideas=prev_ideas_text,
            lessons=lessons_text,
            model_architectures=models_context  # ← NEW: Model context
        )

        # Inject curriculum guidance at the start of the prompt
        prompt = f"{curriculum_guidance}\n\n{prompt}"

        # ══════════════════════════════════════════════════════════════
        # TEMPERATURE ADJUSTMENT BY STAGE
        # ══════════════════════════════════════════════════════════════
        temperature = self._get_stage_temperature(self.current_stage)

        # Call LLM
        response = self.call_llm(
            user_message=prompt,
            temperature=temperature,
            max_tokens=2500
        )

        idea = response["content"].strip()

        # Extract cited lessons
        cited_lessons = self._extract_citations(idea)
        if cited_lessons:
            self.log(f"Idea cites {len(cited_lessons)} lessons: {cited_lessons}")

        # Store idea
        self.generated_ideas.append(idea)

        self.log(f"Generated improved idea ({len(idea)} chars)")

        return idea

    def _determine_curriculum_stage(
        self,
        num_lessons: int,
        valid_count: int,
        stagnation: int,
        num_ideas: int,
    ) -> CurriculumStage:
        """
        Determine the current curriculum stage based on progress.

        Logic:
        - BASELINE: ideas 0-1 (starting out)
        - STANDARD: ideas 2-3, or >= 3 lessons learned
        - ADVANCED: ideas 4-6
        - ENSEMBLE: ideas 7+

        Stagnation accelerates progression to more advanced stages.
        """
        # Stagnation acceleration: if stuck, jump ahead
        if stagnation >= 3:
            if num_ideas < 4:
                return CurriculumStage.ADVANCED
            else:
                return CurriculumStage.ENSEMBLE

        # Normal progression based on number of ideas
        if num_ideas <= 1:
            return CurriculumStage.BASELINE
        elif num_ideas <= 3:
            # Accelerate if many lessons learned
            if num_lessons >= 3:
                return CurriculumStage.STANDARD
            return CurriculumStage.STANDARD
        elif num_ideas <= 6:
            return CurriculumStage.ADVANCED
        else:
            return CurriculumStage.ENSEMBLE

    def _get_curriculum_guidance(self, stage: CurriculumStage) -> str:
        """
        Get curriculum guidance text to inject into the prompt.

        Each stage guides the agent toward appropriate complexity.
        """
        guidance = {
            CurriculumStage.BASELINE: """
== CURRICULUM STAGE: BASELINE ==
Focus on SIMPLE, PROVEN approaches:
- Choose the most appropriate model family for this specific problem
- Minimal feature engineering
- Standard preprocessing (scaling, encoding)
- Prioritize simplicity and fast iteration
- Goal: Establish a solid baseline that works reliably
""",
            CurriculumStage.STANDARD: """
== CURRICULUM STAGE: STANDARD ==
Now explore STANDARD ML techniques:
- Try different model families based on problem characteristics
- Basic feature engineering (interactions, transformations)
- Hyperparameter tuning (grid/random search)
- Cross-validation strategies
- Goal: Improve upon baseline with standard techniques
""",
            CurriculumStage.ADVANCED: """
== CURRICULUM STAGE: ADVANCED ==
Time for ADVANCED optimizations:
- Advanced feature engineering (target encoding, embeddings)
- Model-specific optimizations (learning rate schedules, regularization)
- Advanced preprocessing (dimensionality reduction, feature selection)
- Domain-specific techniques
- Goal: Push performance with sophisticated techniques
""",
            CurriculumStage.ENSEMBLE: """
== CURRICULUM STAGE: ENSEMBLE ==
Explore ENSEMBLE and META-LEARNING strategies:
- Model stacking (multiple layers of models)
- Blending (weighted averaging of predictions)
- Diverse model ensembles (different algorithms, different features)
- Out-of-fold predictions for second-level models
- Goal: Maximize performance through model combination
""",
        }
        return guidance.get(stage, "")

    def _get_stage_temperature(self, stage: CurriculumStage) -> float:
        """
        Get LLM temperature based on curriculum stage.

        - BASELINE/ENSEMBLE: Lower temperature (0.6) for consistency
        - STANDARD/ADVANCED: Higher temperature (0.8) for exploration
        """
        if stage in (CurriculumStage.BASELINE, CurriculumStage.ENSEMBLE):
            return 0.6
        else:
            return 0.8
    
    def _select_baseline_model(self, models: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Select the most appropriate baseline model.
        
        Heuristic: Prefer efficient/lightweight models for initial baseline.
        """
        if not models:
            return {"name": "Unknown", "reasoning": "", "description": ""}
        
        # Look for "efficient" or "light" keywords
        for model in models:
            name_lower = model.get("name", "").lower()
            desc_lower = model.get("description", "").lower()
            
            if any(keyword in name_lower or keyword in desc_lower 
                   for keyword in ["efficient", "light", "mobile", "distil"]):
                return model
        
        # Fallback: return first model
        return models[0]
    
    def _create_initial_idea_prompt(
        self,
        problem_description: str,
        eda_report: str,
        baseline_model: Optional[Dict[str, str]],
        all_models: List[Dict[str, str]]
    ) -> str:
        """
        Create prompt for initial idea generation.
        
        Includes discovered model context.
        """
        # Format models as context
        models_context = ""
        if baseline_model:
            models_context = f"""
BASELINE MODEL RECOMMENDATION:
Based on automated search, we recommend starting with: {baseline_model['name']}

Reasoning: {baseline_model.get('reasoning', 'Efficient and proven for this domain')}

Description: {baseline_model.get('description', 'Standard architecture')}

OTHER DISCOVERED MODELS (for reference):
"""
            for i, model in enumerate(all_models[:4], 1):
                if model['name'] != baseline_model['name']:
                    models_context += f"{i}. {model['name']} - {model.get('reasoning', 'Alternative approach')}\n"
        
        prompt = f"""You are an expert ML engineer. Based on the challenge description below, propose a simple baseline solution.

CHALLENGE DESCRIPTION:
{problem_description[:800]}

DATA INSIGHTS:
{eda_report[:500]}

{models_context}

YOUR TASK:
Propose a SIMPLE baseline approach. Consider:
- What type of problem is this? (classification, regression, etc.)
- What preprocessing is needed?
- Which model from the recommendations above would work well?
- How to evaluate the solution?

IMPORTANT:
- Start with the BASELINE MODEL recommendation if provided
- Keep it simple and fast to implement
- Focus on a working solution, not perfection
- Be specific about the model architecture to use

Describe your solution idea in 5-8 sentences. Include:
1. Problem type and approach
2. Specific model/architecture to use
3. Key preprocessing steps
4. Training strategy
5. Evaluation approach
"""
        
        return prompt
    
    def _format_previous_ideas(self, ideas: Optional[List[str]] = None) -> str:
        """Format previous ideas for prompt"""
        ideas = ideas or self.generated_ideas
        
        if not ideas:
            return "No previous ideas."
        
        formatted = []
        for i, idea in enumerate(ideas, 1):
            formatted.append(f"[Idea {i}]\n{idea}\n")
        
        return "\n".join(formatted)
    
    def _format_models_context(self, models: List[Dict[str, str]]) -> str:
        """Format discovered models as context with full descriptions (no truncation)"""
        if not models:
            return "No model information available."

        formatted = "DISCOVERED MODELS:\n"
        for i, model in enumerate(models, 1):
            formatted += f"{i}. {model['name']}\n"
            formatted += f"   Reasoning: {model.get('reasoning', 'N/A')}\n"
            formatted += f"   Description: {model.get('description', 'N/A')}\n\n"

        return formatted
    
    def _extract_citations(self, text: str) -> List[str]:
        """
        Extract lesson citations from text.
        
        Looks for pattern: "Cite {lesson_id}"
        
        Args:
            text: Text containing citations
            
        Returns:
            List of cited lesson IDs
        """
        import re
        
        pattern = r'Cite\s+\{([^}]+)\}'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        return matches