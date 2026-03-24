# agents/idea_agent.py
from typing import Optional, List, Dict, Tuple
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


class ExplorationPhase(Enum):
    """
    High-level exploration phases for radical strategy shifts.

    Unlike CurriculumStage (which is incremental), these phases represent
    FUNDAMENTAL changes in exploration approach after many iterations.

    Triggers (configurable):
    - BREADTH_SEARCH: Initial phase (iterations 1-15)
    - DEEP_EXPLOIT: After all models tried once (iterations 16-30)
    - RADICAL_PIVOT: After prolonged stagnation (iterations 31-45)
    - ENSEMBLE_SYNTHESIS: Final phase (iterations 46+)
    """
    BREADTH_SEARCH = "breadth_search"      # Explore all model families
    DEEP_EXPLOIT = "deep_exploit"          # Exploit top 2 models deeply
    RADICAL_PIVOT = "radical_pivot"        # Reframe problem, new SOTA search
    ENSEMBLE_SYNTHESIS = "ensemble_synth"  # Combine best approaches found


# ═══════════════════════════════════════════════════════════════════════════
# CREATIVE APPROACHES BLOCK - Encourages novel solutions
# ═══════════════════════════════════════════════════════════════════════════
CREATIVE_APPROACHES_BLOCK = """
═══════════════════════════════════════════════════════════════════════════════
THINK CREATIVELY - EXPLORE UNCONVENTIONAL APPROACHES
═══════════════════════════════════════════════════════════════════════════════
You are NOT limited to traditional ML pipelines. Consider these creative approaches:

**LLM-BASED SOLUTIONS (often surprisingly effective!):**
- Zero-shot classification: Use an LLM to directly classify/predict via prompting
- Few-shot learning: Include examples in the prompt for in-context learning
- Chain-of-thought: Have the LLM reason step-by-step before predicting
- LLM as feature extractor: Generate embeddings or semantic features
- LLM for data augmentation: Generate synthetic training examples
- LLM ensemble: Multiple prompts voting on the answer

**HYBRID APPROACHES:**
- LLM + Traditional ML: Use LLM features as input to XGBoost/RF
- Two-stage: LLM for preprocessing/feature extraction, ML for prediction
- LLM for hard cases: Route difficult samples to LLM, easy ones to fast model
- Prompt-tuning + fine-tuning combination

**UNCONVENTIONAL TECHNIQUES:**
- Problem reframing: Classification as ranking, regression as classification
- Self-training / pseudo-labeling with confidence thresholds
- Data programming / weak supervision with labeling functions
- Active learning simulation: Focus on uncertain samples
- Retrieval-augmented prediction: Find similar examples at inference
- Rule-based systems + ML fallback
- Symbolic AI + neural hybrid

**CREATIVE DATA STRATEGIES:**
- Synthetic data generation (LLM, SMOTE variants, mixup)
- Cross-domain transfer learning
- Multi-task learning with auxiliary objectives
- Contrastive learning for better representations
- Knowledge distillation from larger models

**META-LEARNING:**
- AutoML-style hyperparameter search within the solution
- Neural architecture search concepts
- Learning to learn from the problem structure

BE BOLD! The best solutions often come from unexpected combinations.
If traditional approaches plateau, TRY SOMETHING RADICALLY DIFFERENT.
═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# LLM AUTONOMOUS CAPABILITIES BLOCK
# ═══════════════════════════════════════════════════════════════════════════
LLM_CAPABILITIES_BLOCK = """
═══════════════════════════════════════════════════════════════════════════════
YOUR AUTONOMOUS CAPABILITIES
═══════════════════════════════════════════════════════════════════════════════
You can request actions by including these markers in your response.
The system will evaluate your suggestions and execute valid ones.

1. REQUEST ADDITIONAL MODEL SEARCH:
   If you believe exploring different models/techniques would help:
   @@ACTION:SEARCH_MODELS:{"keywords": ["keyword1", "keyword2"]}@@

2. SUGGEST MODEL SWITCH:
   If you think a different model family would work better:
   @@ACTION:SWITCH_MODEL:{"model_name": "ModelName", "reason": "why this model fits better"}@@

3. REQUEST PHASE CHANGE:
   If current exploration approach doesn't fit:
   @@ACTION:CHANGE_PHASE:{"phase": "RADICAL_PIVOT", "reason": "why change is needed"}@@
   Valid phases: BREADTH_SEARCH, DEEP_EXPLOIT, RADICAL_PIVOT, ENSEMBLE_SYNTHESIS

4. REQUEST SPECIFIC LESSONS:
   If you need lessons about specific topics:
   @@ACTION:FILTER_LESSONS:{"keywords": ["topic1", "topic2"], "count": 10}@@

5. SIGNAL CONVERGENCE:
   If you believe the search has converged (plateau detected):
   @@ACTION:SIGNAL_CONVERGENCE:{"confidence": 0.9, "reason": "metric stable for N iterations"}@@

6. REQUEST ANALYSIS:
   If you need additional data analysis:
   @@ACTION:REQUEST_ANALYSIS:{"type": "feature_importance"}@@
   Valid types: feature_importance, correlation, distribution, missing_values

GUIDELINES:
- Use actions when you have STRONG REASONING (explain why in the "reason" field)
- You can include UP TO 3 actions per response
- Actions are SUGGESTIONS - the system decides whether to accept
- Continue with your normal response after including actions
═══════════════════════════════════════════════════════════════════════════════
"""

class IdeaAgent(BaseAgent):
    """
    Agent responsible for generating solution ideas.
    Implements curriculum-based idea generation from Section 4.5.

    Now includes:
    - SOTA model discovery via SearchAgent
    - Exploration Phases for radical strategy shifts after many iterations
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPLORATION PHASE THRESHOLDS (configurable)
    # ═══════════════════════════════════════════════════════════════════════════
    PHASE_BREADTH_END: int = 15        # End of breadth search phase
    PHASE_DEEP_EXPLOIT_END: int = 30   # End of deep exploitation phase
    PHASE_RADICAL_PIVOT_END: int = 45  # End of radical pivot phase
    # After this: ENSEMBLE_SYNTHESIS indefinitely

    # Stagnation threshold to force early pivot (iterations without >5% improvement)
    STAGNATION_PIVOT_THRESHOLD: int = 10

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__("IdeaAgent")
        self.generated_ideas: List[str] = []
        self.search_agent = SearchAgent()  # ← NEW: Web search capability
        self.discovered_models: List[Dict[str, str]] = []  # ← Cache discovered models

        # Curriculum exploration (Mejora 3)
        self.current_stage: CurriculumStage = CurriculumStage.BASELINE
        self.stage_history: List[CurriculumStage] = []

        # ═══════════════════════════════════════════════════════════════════════════
        # EXPLORATION PHASES (NEW)
        # ═══════════════════════════════════════════════════════════════════════════
        self.current_phase: ExplorationPhase = ExplorationPhase.BREADTH_SEARCH
        self.phase_history: List[ExplorationPhase] = []
        self.total_iterations: int = 0
        self.iterations_without_significant_improvement: int = 0
        self.best_metric_at_phase_start: Optional[float] = None
        self.top_models_for_exploitation: List[Dict] = []  # Top 2 models for DEEP_EXPLOIT

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
        model_architectures: List[Dict[str, str]] = None,  # ← Now optional
        force_model: Optional[Dict[str, str]] = None,  # ← NEW: Force specific model
        force_model_index: int = -1,  # ← NEW: Index of forced model
    ) -> Optional[str]:
        """
        Generate initial lightweight baseline idea.

        NOW INCLUDES: Automatic SOTA model discovery via web search.

        Args:
            problem_description: Task description
            eda_report: EDA insights
            model_architectures: Optional pre-discovered models (legacy)
            force_model: If provided, use this specific model (from MCTS rotation)
            force_model_index: Index of the forced model in discovered_models

        Returns:
            Natural language idea description
        """
        self.log("Generating initial baseline idea...")

        # ========================================
        # STEP 1: Discover SOTA Models (or use provided)
        # ========================================
        if not model_architectures and not self.discovered_models:
            self.log("  Discovering SOTA models via search...")
            self.discovered_models = self.search_agent.search_sota_models(
                problem_description=problem_description,
                num_candidates=5
            )
            model_architectures = self.discovered_models
        elif not model_architectures:
            model_architectures = self.discovered_models

        # ========================================
        # STEP 2: Select Model (forced or baseline)
        # ========================================
        if force_model:
            # MCTS is forcing a specific model for exploration
            baseline_model = force_model
            self.log(f"  FORCED model by MCTS: {baseline_model['name']} (index {force_model_index})")
        elif model_architectures:
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
            model_architectures,
            force_model=force_model is not None,
        )

        response = self.call_llm(
            user_message=prompt,
            temperature=0.5,
            max_tokens=16000
        )

        idea = response["content"].strip()
        self.generated_ideas.append(idea)

        self.log(f"Generated initial idea ({len(idea)} chars)")

        # Return model info along with idea for tracking
        self._last_used_model = baseline_model
        self._last_used_model_index = force_model_index if force_model else 0

        return idea

    def get_last_used_model_info(self) -> Tuple[Optional[Dict], int]:
        """Get info about the last model used in idea generation"""
        return (
            getattr(self, '_last_used_model', None),
            getattr(self, '_last_used_model_index', -1)
        )
    
    def improve_idea(
        self,
        problem_description: str,
        eda_report: str,
        lesson_pool: LessonPool,
        previous_ideas: Optional[List[str]] = None,
        valid_solutions_count: int = 0,
        stagnation_count: int = 0,
        best_solution_code: Optional[str] = None,
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

        # Format best solution code as context
        best_solution_context = ""
        if best_solution_code:
            best_solution_context = f"\n\nBEST SOLUTION SO FAR (study this code — understand what it does well and what to improve):\n{best_solution_code}"

        # Build novelty constraint: list key strategies already tried
        novelty_block = self._build_novelty_constraint(previous_ideas or self.generated_ideas)

        # Get prompt with curriculum guidance injected
        prompt = self.prompt_manager.get_prompt(
            "idea_improvement",
            problem_description=problem_description,
            eda_report=eda_report,
            previous_ideas=prev_ideas_text,
            lessons=lessons_text,
            model_architectures=models_context
        )

        # Append best solution context after the prompt
        if best_solution_context:
            prompt = prompt + best_solution_context

        # Get exploration phase guidance (higher-level than curriculum)
        phase_guidance = self.get_phase_guidance()

        # Inject phase guidance + curriculum guidance + novelty constraint at the start
        # And LLM capabilities at the end
        prompt = f"{phase_guidance}\n\n{curriculum_guidance}\n\n{novelty_block}\n\n{prompt}\n\n{LLM_CAPABILITIES_BLOCK}"

        # ══════════════════════════════════════════════════════════════
        # TEMPERATURE ADJUSTMENT BY STAGE
        # ══════════════════════════════════════════════════════════════
        temperature = self._get_stage_temperature(self.current_stage)

        # Call LLM
        response = self.call_llm(
            user_message=prompt,
            temperature=temperature,
            max_tokens=16000
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

        Normal progression: BASELINE(0-1) -> STANDARD(2-3) -> ADVANCED(4-6) -> ENSEMBLE(7+)

        Stagnation behaviour (NEW):
        - Instead of jumping forward to harder stages (which often fail the same way),
          step BACK one level so the model re-explores simpler approaches with fresh eyes.
        - After 2+ consecutive stagnation events, jump to ENSEMBLE to force diversity.
        """
        _stages = [
            CurriculumStage.BASELINE,
            CurriculumStage.STANDARD,
            CurriculumStage.ADVANCED,
            CurriculumStage.ENSEMBLE,
        ]

        # Count consecutive stagnation events from stage_history
        consecutive_stagnations = 0
        for past_stage in reversed(self.stage_history[-6:]):
            if past_stage == self.current_stage:
                consecutive_stagnations += 1
            else:
                break

        if stagnation >= 3:
            if consecutive_stagnations >= 2:
                # Stuck in same stage through 2+ stagnation cycles — jump to ENSEMBLE
                return CurriculumStage.ENSEMBLE
            else:
                # Step BACK one level to re-explore simpler strategies
                current_idx = _stages.index(self.current_stage)
                return _stages[max(0, current_idx - 1)]

        # Normal progression based on number of ideas
        if num_ideas <= 1:
            return CurriculumStage.BASELINE
        elif num_ideas <= 3:
            return CurriculumStage.STANDARD
        elif num_ideas <= 6:
            return CurriculumStage.ADVANCED
        else:
            return CurriculumStage.ENSEMBLE

    def _get_curriculum_guidance(self, stage: CurriculumStage) -> str:
        """
        Get curriculum guidance text to inject into the prompt.

        Each stage guides the agent toward appropriate complexity.
        NOW INCLUDES creative and unconventional approaches at every stage.
        """
        guidance = {
            CurriculumStage.BASELINE: """
== CURRICULUM STAGE: BASELINE ==
Focus on SIMPLE approaches that WORK:
- Traditional ML: Choose appropriate model family (tree-based, linear, neural)
- **OR try LLM-based**: Zero-shot classification with good prompting can beat complex ML!
- Minimal preprocessing, let the model handle complexity
- Quick iteration, fail fast
- Goal: Get a WORKING solution, conventional OR creative
""",
            CurriculumStage.STANDARD: """
== CURRICULUM STAGE: STANDARD ==
Explore DIVERSE approaches - both traditional AND creative:

TRADITIONAL PATH:
- Different model families (XGBoost, RandomForest, Neural Nets)
- Feature engineering, hyperparameter tuning

CREATIVE PATH (try these!):
- LLM few-shot learning: Include examples in prompt
- LLM as feature extractor: Generate semantic embeddings
- Hybrid: LLM features + traditional ML classifier
- Problem reframing: Is this really classification? Could be ranking/similarity

Goal: Find what WORKS BEST, don't limit yourself to conventional ML
""",
            CurriculumStage.ADVANCED: """
== CURRICULUM STAGE: ADVANCED ==
Time for SOPHISTICATED and UNCONVENTIONAL techniques:

ADVANCED ML:
- Target encoding, learned embeddings
- Neural architecture customization
- Advanced regularization, learning schedules

CREATIVE TECHNIQUES (highly encouraged!):
- Chain-of-thought prompting: Have LLM reason step-by-step
- Retrieval-augmented: Find similar examples at inference time
- Self-training: Use confident predictions as pseudo-labels
- LLM ensemble: Multiple prompts voting on answer
- Data augmentation via LLM: Generate synthetic examples
- Two-stage: LLM for feature extraction, fast model for prediction

Goal: Push boundaries with creative combinations
""",
            CurriculumStage.ENSEMBLE: """
== CURRICULUM STAGE: ENSEMBLE ==
COMBINE the best of ALL worlds:

TRADITIONAL ENSEMBLES:
- Model stacking, blending, voting
- Diverse base models (different algorithms, features)

CREATIVE ENSEMBLES (the winning edge!):
- LLM + ML ensemble: LLM votes alongside traditional models
- Confidence-based routing: Hard cases to LLM, easy to fast model
- Multi-prompt ensemble: Different prompting strategies
- Knowledge distillation: Train small model on LLM predictions
- Symbolic + Neural: Rule-based for clear cases, ML for fuzzy

Goal: Maximum performance through creative model combination
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
            return 0.3
        else:
            return 0.7
    
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
        all_models: List[Dict[str, str]],
        force_model: bool = False,
    ) -> str:
        """
        Create prompt for initial idea generation.

        Includes discovered model context.

        Args:
            force_model: If True, the model is being forced by MCTS rotation
        """
        # Format models as context
        models_context = ""
        if baseline_model:
            if force_model:
                # Stronger language when MCTS forces a specific model
                models_context = f"""
═══════════════════════════════════════════════════════════════════════════
MANDATORY MODEL (assigned by exploration system):
You MUST use this model: **{baseline_model['name']}**

Reasoning: {baseline_model.get('reasoning', 'Selected for systematic exploration')}

Description: {baseline_model.get('description', 'Standard architecture')}

DO NOT suggest a different model. Build the solution around {baseline_model['name']}.
═══════════════════════════════════════════════════════════════════════════
"""
            else:
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

        model_instruction = ""
        if force_model and baseline_model:
            model_instruction = f"""
CRITICAL: You MUST use {baseline_model['name']} as the primary model.
Do NOT propose alternative models. The exploration system is systematically
testing each model to find the best one for this problem.
"""

        prompt = f"""You are an expert ML engineer. Based on the challenge description below, propose a simple baseline solution.

CHALLENGE DESCRIPTION:
{problem_description}

DATA INSIGHTS:
{eda_report}

{models_context}

YOUR TASK:
Propose a solution approach. You have FULL CREATIVE FREEDOM - consider:
- What type of problem is this? (classification, regression, ranking, etc.)
- What's the BEST approach? Traditional ML? LLM-based? Hybrid?
- Could an LLM solve this directly via prompting? (often surprisingly effective!)
- What preprocessing is needed - or can you skip it entirely?

IMPORTANT:
- You are NOT limited to traditional ML pipelines
- LLM-based solutions (zero-shot, few-shot, chain-of-thought) are VALID and often BETTER
- Hybrid approaches (LLM + ML) can combine the best of both worlds
- If a model is recommended, consider it but feel free to propose alternatives
- Be CREATIVE - unexpected approaches often win
{model_instruction}

Describe your solution idea in 5-10 sentences. Include:
1. Problem type and chosen approach (traditional ML / LLM-based / hybrid / other)
2. Specific model/technique to use and WHY
3. If LLM-based: prompting strategy (zero-shot, few-shot, chain-of-thought)
4. Key preprocessing or data handling
5. Training/inference strategy
6. Why you believe this approach will work well

{CREATIVE_APPROACHES_BLOCK}

{LLM_CAPABILITIES_BLOCK}
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
    
    def _build_novelty_constraint(self, ideas: List[str]) -> str:
        """
        Build a novelty constraint block listing key strategies already tried.

        Extracts short fingerprints (technique keywords) from each idea so the
        LLM knows what NOT to repeat, without flooding the prompt with full text.
        """
        if not ideas:
            return ""

        import re
        # Keywords that identify a distinct ML strategy
        strategy_keywords = [
            r'\b(lgbm|lightgbm|xgboost|xgb|catboost|random.?forest|gradient.?boost'
            r'|logistic.?regression|svm|linear.?model|neural.?net|mlp|lstm|gru|bert'
            r'|transformer|attention|cnn|resnet|ensemble|stacking|blending|bagging'
            r'|optuna|hyperopt|bayesian.?optim|target.?encod|feature.?select'
            r'|pca|svd|embedding|tfidf|word2vec|smote|oversampl|undersampl'
            r'|k.?fold|stratif|pseudo.?label|self.?train)\b',
        ]
        seen = set()
        fingerprints = []
        for idea in ideas[-10:]:  # Only last 10 ideas
            for pat in strategy_keywords:
                for m in re.finditer(pat, idea, re.IGNORECASE):
                    kw = m.group(1).lower().replace(' ', '_').replace('-', '_')
                    if kw not in seen:
                        seen.add(kw)
                        fingerprints.append(kw)

        if not fingerprints:
            return ""

        fp_list = ", ".join(sorted(set(fingerprints)))
        return (
            "== NOVELTY CONSTRAINT ==\n"
            f"The following strategies have ALREADY been tried: [{fp_list}]\n"
            "You MUST propose a FUNDAMENTALLY DIFFERENT approach. "
            "Do NOT use any of the strategies listed above as your primary technique.\n"
        )

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

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPLORATION PHASES (Radical Strategy Shifts)
    # ═══════════════════════════════════════════════════════════════════════════

    def update_exploration_state(
        self,
        iteration: int,
        current_best_metric: Optional[float],
        model_exploration_stats: Dict[int, Dict],
        lower_is_better: bool = False,
    ) -> ExplorationPhase:
        """
        Update exploration phase based on search progress.

        Called by orchestrator after each iteration to track state and
        determine if a phase transition is needed.

        Args:
            iteration: Current MCTS iteration number
            current_best_metric: Best metric found so far
            model_exploration_stats: Per-model stats from MCTS
            lower_is_better: Metric direction

        Returns:
            Current exploration phase (may have changed)
        """
        self.total_iterations = iteration
        old_phase = self.current_phase

        # Track significant improvement (>5%)
        if self.best_metric_at_phase_start is not None and current_best_metric is not None:
            if lower_is_better:
                improvement = (self.best_metric_at_phase_start - current_best_metric) / max(abs(self.best_metric_at_phase_start), 1e-6)
            else:
                improvement = (current_best_metric - self.best_metric_at_phase_start) / max(abs(self.best_metric_at_phase_start), 1e-6)

            if improvement > 0.05:  # 5% improvement threshold
                self.iterations_without_significant_improvement = 0
                self.best_metric_at_phase_start = current_best_metric
            else:
                self.iterations_without_significant_improvement += 1
        else:
            self.best_metric_at_phase_start = current_best_metric

        # Determine new phase
        new_phase = self._determine_exploration_phase(iteration, model_exploration_stats)

        if new_phase != old_phase:
            self._on_phase_transition(old_phase, new_phase, model_exploration_stats)

        self.current_phase = new_phase
        self.phase_history.append(new_phase)

        return new_phase

    def _determine_exploration_phase(
        self,
        iteration: int,
        model_exploration_stats: Dict[int, Dict],
    ) -> ExplorationPhase:
        """
        Determine which exploration phase we're in.

        Phase transitions:
        1. BREADTH_SEARCH → DEEP_EXPLOIT: After PHASE_BREADTH_END iterations OR
           all models have been tried at least once
        2. DEEP_EXPLOIT → RADICAL_PIVOT: After PHASE_DEEP_EXPLOIT_END iterations OR
           prolonged stagnation (STAGNATION_PIVOT_THRESHOLD)
        3. RADICAL_PIVOT → ENSEMBLE_SYNTHESIS: After PHASE_RADICAL_PIVOT_END iterations
        """
        # Check for prolonged stagnation → force RADICAL_PIVOT early
        if (self.iterations_without_significant_improvement >= self.STAGNATION_PIVOT_THRESHOLD
            and self.current_phase in (ExplorationPhase.BREADTH_SEARCH, ExplorationPhase.DEEP_EXPLOIT)):
            self.log(f"  [Phase] Prolonged stagnation ({self.iterations_without_significant_improvement} iters) → RADICAL_PIVOT")
            return ExplorationPhase.RADICAL_PIVOT

        # Check if all models have been explored (triggers early transition to DEEP_EXPLOIT)
        all_models_explored = False
        if model_exploration_stats:
            all_models_explored = all(
                stats.get("valid_count", 0) > 0
                for stats in model_exploration_stats.values()
            )

        # Normal phase progression based on iteration count
        if iteration <= self.PHASE_BREADTH_END and not all_models_explored:
            return ExplorationPhase.BREADTH_SEARCH
        elif iteration <= self.PHASE_DEEP_EXPLOIT_END:
            return ExplorationPhase.DEEP_EXPLOIT
        elif iteration <= self.PHASE_RADICAL_PIVOT_END:
            return ExplorationPhase.RADICAL_PIVOT
        else:
            return ExplorationPhase.ENSEMBLE_SYNTHESIS

    def _on_phase_transition(
        self,
        old_phase: ExplorationPhase,
        new_phase: ExplorationPhase,
        model_exploration_stats: Dict[int, Dict],
    ):
        """
        Handle phase transition actions.

        - DEEP_EXPLOIT: Identify top 2 models to focus on
        - RADICAL_PIVOT: Trigger new SOTA search with different queries
        - ENSEMBLE_SYNTHESIS: Prepare best models for combination
        """
        self.log(f"  [Phase] TRANSITION: {old_phase.value} -> {new_phase.value}")
        self.iterations_without_significant_improvement = 0

        if new_phase == ExplorationPhase.DEEP_EXPLOIT:
            # Identify top 2 models by best_metric
            self.top_models_for_exploitation = self._get_top_models(model_exploration_stats, n=2)
            model_names = [m.get("name", "?") for m in self.top_models_for_exploitation]
            self.log(f"  [Phase] DEEP_EXPLOIT: Focusing on top models: {model_names}")

        elif new_phase == ExplorationPhase.RADICAL_PIVOT:
            self.log("  [Phase] RADICAL_PIVOT: Will re-search SOTA with different queries")
            # Clear discovered models to force new search
            # (But keep the old ones as backup in case new search fails)
            self._backup_models = self.discovered_models.copy()

        elif new_phase == ExplorationPhase.ENSEMBLE_SYNTHESIS:
            self.log("  [Phase] ENSEMBLE_SYNTHESIS: Will combine best approaches")

    def _get_top_models(self, model_exploration_stats: Dict[int, Dict], n: int = 2) -> List[Dict]:
        """Get the top N models by best_metric."""
        if not model_exploration_stats or not self.discovered_models:
            return []

        # Sort models by best_metric (handle None values)
        scored_models = []
        for idx, stats in model_exploration_stats.items():
            if int(idx) < len(self.discovered_models):
                model = self.discovered_models[int(idx)]
                best_metric = stats.get("best_metric")
                if best_metric is not None:
                    scored_models.append((model, best_metric))

        # Sort by metric (descending for maximize, would need to adjust for minimize)
        scored_models.sort(key=lambda x: x[1], reverse=True)

        return [m[0] for m in scored_models[:n]]

    def get_phase_guidance(self) -> str:
        """
        Get guidance text for the current exploration phase.

        This is injected into the idea generation prompt to steer
        the LLM toward the appropriate strategy.

        NOW INCLUDES creative/unconventional approaches at every phase.
        """
        guidance = {
            ExplorationPhase.BREADTH_SEARCH: """
═══════════════════════════════════════════════════════════════════════════════
EXPLORATION PHASE: BREADTH SEARCH
═══════════════════════════════════════════════════════════════════════════════
Strategy: Explore FUNDAMENTALLY DIFFERENT approaches - both traditional AND creative.

MUST TRY (diversity is key):
- Traditional ML: XGBoost, RandomForest, Neural Nets, Linear models
- **LLM-BASED**: Zero-shot prompting, few-shot learning (HIGHLY RECOMMENDED!)
- **HYBRID**: LLM features + traditional classifier

Why try LLM-based?
- Zero-shot can beat fine-tuned models on many tasks
- No training data needed - works immediately
- Understands context and semantics naturally

Rules:
- Each attempt should be FUNDAMENTALLY different
- Don't just tune hyperparameters - try completely different paradigms
- LLM solutions count as valid exploration!
═══════════════════════════════════════════════════════════════════════════════
""",
            ExplorationPhase.DEEP_EXPLOIT: f"""
═══════════════════════════════════════════════════════════════════════════════
EXPLORATION PHASE: DEEP EXPLOITATION
═══════════════════════════════════════════════════════════════════════════════
Strategy: OPTIMIZE the best approaches found so far.

TOP PERFORMERS TO FOCUS ON:
{self._format_top_models()}

For TRADITIONAL ML approaches:
- Hyperparameter optimization (Optuna, grid search)
- Feature engineering specific to this model
- Training tricks (learning rate schedules, regularization)

For LLM-BASED approaches (if they worked well):
- Prompt engineering: Refine the prompt structure
- Few-shot optimization: Better example selection
- Chain-of-thought: Add reasoning steps
- Output parsing: Better structured extraction

For HYBRID approaches:
- Balance between LLM and ML components
- Feature fusion strategies
═══════════════════════════════════════════════════════════════════════════════
""",
            ExplorationPhase.RADICAL_PIVOT: """
═══════════════════════════════════════════════════════════════════════════════
EXPLORATION PHASE: RADICAL PIVOT - BE WILDLY CREATIVE!
═══════════════════════════════════════════════════════════════════════════════
Previous approaches have PLATEAUED. Time to think COMPLETELY DIFFERENTLY.

RADICAL IDEAS TO TRY:

1. **FLIP THE PARADIGM**:
   - If using ML → try pure LLM prompting
   - If using LLM → try simple rule-based + ML
   - Classification → Ranking/Similarity
   - Supervised → Self-supervised + pseudo-labels

2. **LLM CREATIVE TECHNIQUES**:
   - Chain-of-thought with self-consistency (multiple reasoning paths)
   - Tree-of-thought (explore multiple solution branches)
   - LLM as judge (generate, then self-evaluate)
   - Debate: Two LLM instances argue for different answers
   - Reflection: LLM critiques and improves its own answer

3. **UNCONVENTIONAL APPROACHES**:
   - Retrieval-augmented: Find similar examples dynamically
   - Data programming: Write labeling functions
   - Active learning simulation: Focus on hard cases
   - Symbolic AI + Neural hybrid
   - Multi-task: Add auxiliary objectives

4. **CREATIVE DATA STRATEGIES**:
   - LLM-generated synthetic data
   - Cross-domain transfer from unexpected sources
   - Contrastive learning for representations

THE CRAZIER THE IDEA, THE BETTER. Standard approaches have failed!
═══════════════════════════════════════════════════════════════════════════════
""",
            ExplorationPhase.ENSEMBLE_SYNTHESIS: """
═══════════════════════════════════════════════════════════════════════════════
EXPLORATION PHASE: ENSEMBLE SYNTHESIS - COMBINE EVERYTHING
═══════════════════════════════════════════════════════════════════════════════
Strategy: Create POWERFUL COMBINATIONS of all successful approaches.

ENSEMBLE STRATEGIES:

1. **TRADITIONAL ENSEMBLES**:
   - Stacking: Train meta-model on base predictions
   - Blending: Weighted average of models
   - Voting: Majority/soft voting

2. **LLM + ML ENSEMBLES** (often the winning combo!):
   - LLM as one voter alongside ML models
   - Confidence-based routing: Hard cases → LLM, easy → fast ML
   - LLM for edge cases, ML for bulk predictions
   - LLM as tie-breaker when models disagree

3. **MULTI-PROMPT ENSEMBLES**:
   - Different prompting strategies voting
   - Temperature diversity: Same prompt, different temperatures
   - Perspective diversity: Different personas/viewpoints

4. **KNOWLEDGE DISTILLATION**:
   - Train fast model on LLM predictions
   - Combine distilled model with original LLM

Goal: Maximum performance by combining the BEST of traditional AND creative approaches!
═══════════════════════════════════════════════════════════════════════════════
""",
        }
        return guidance.get(self.current_phase, "")

    def _format_top_models(self) -> str:
        """Format top models for deep exploitation prompt."""
        if not self.top_models_for_exploitation:
            return "No top models identified yet."

        lines = []
        for i, model in enumerate(self.top_models_for_exploitation, 1):
            lines.append(f"  {i}. {model.get('name', 'Unknown')}")
        return "\n".join(lines)

    def trigger_radical_search(self, problem_description: str) -> List[Dict[str, str]]:
        """
        Trigger a new SOTA search with DIFFERENT queries.

        Called during RADICAL_PIVOT phase to find alternative approaches
        that weren't discovered in the initial search.
        """
        self.log("  [Phase] Triggering radical SOTA re-search...")

        # Use different search queries to find alternative approaches
        new_models = self.search_agent.search_sota_models(
            problem_description=problem_description,
            num_candidates=5,
            alternative_search=True  # Flag for different query generation
        )

        if new_models:
            # Merge with existing models (avoid duplicates)
            existing_names = {m.get("name", "").lower() for m in self.discovered_models}
            for model in new_models:
                if model.get("name", "").lower() not in existing_names:
                    self.discovered_models.append(model)
                    self.log(f"  [Phase] Discovered new model: {model.get('name')}")

        return new_models