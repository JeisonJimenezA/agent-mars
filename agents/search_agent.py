# agents/search_agent.py
from typing import List, Dict, Optional
from pathlib import Path
from agents.base_agent import BaseAgent
from utils.academic_search import search_all_academic_sources, set_cache_file, save_cache
from core.config import Config
import re
import json

class SearchAgent(BaseAgent):
    """
    Agent responsible for web search and information discovery.
    Implements model architecture search from Appendix C.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__("SearchAgent")
        self.search_cache: Dict[str, List[Dict]] = {}

        # Initialize persistent cache for academic search
        cache_dir = cache_dir or Config.WORKING_DIR / Config.CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "academic_search_cache.json"
        set_cache_file(cache_file)
        self.log(f"Academic search cache: {cache_file}")
    
    def get_system_prompt(self) -> str:
        return """You are an expert ML researcher specializing in finding state-of-the-art models and solutions. You excel at analyzing search results and extracting relevant technical information."""
    
    def search_sota_models(
        self,
        problem_description: str,
        num_candidates: int = 5
    ) -> List[Dict[str, str]]:
        """
        Search for state-of-the-art model architectures for the task.

        This implements the SearchArchitectures function from Algorithm 2.

        Args:
            problem_description: Task description
            num_candidates: Number of model candidates to return

        Returns:
            List of dicts with 'name', 'reasoning', 'description'
        """
        self.log(f"Searching for SOTA models (target: {num_candidates} candidates)")

        # Step 1: Extract problem-specific keywords
        problem_keywords = self._extract_problem_keywords(problem_description)

        # Step 2: Identify task type and domain
        task_info = self._identify_task_type(problem_description)
        self.log(f"  Task type: {task_info.get('task_type', 'unknown')}")
        self.log(f"  Domain: {task_info.get('domain', 'unknown')}")

        # Step 3: Generate targeted search queries using FULL problem context
        queries = self._generate_search_queries(
            task_info,
            problem_keywords,
            problem_description
        )
        self.log(f"  Generated {len(queries)} search queries")
        
        # Step 4: Perform academic searches
        search_results = self._perform_searches(queries)

        # Step 5: Extract and rank model candidates using full context
        candidates = self._extract_model_candidates(
            search_results,
            task_info,
            num_candidates,
            problem_keywords
        )
        
        self.log(f"  Found {len(candidates)} model candidates")
        for i, candidate in enumerate(candidates, 1):
            self.log(f"    {i}. {candidate['name']}")
        
        return candidates
    
    def _extract_problem_keywords(self, problem_description: str) -> Dict[str, str]:
        """
        Extract specific keywords from the problem description for targeted search.

        Returns dict with challenge-specific terms for more relevant searches.
        """
        prompt = f"""Extract specific keywords from this ML challenge for academic paper search.

CHALLENGE DESCRIPTION:
{problem_description[:2000]}

Extract the following (be SPECIFIC, not generic):
1. challenge_name: The name of the competition/challenge (e.g., "Otto Group", "Titanic", "House Prices")
2. metric_name: The evaluation metric (e.g., "logloss", "RMSE", "AUC", "F1")
3. target_variable: What we're predicting (e.g., "product category", "survival", "price")
4. data_characteristics: Key data features (e.g., "93 numerical features", "mixed categorical", "imbalanced classes")
5. industry_domain: Business/scientific domain (e.g., "e-commerce", "healthcare", "finance")
6. num_classes: Number of classes if classification (e.g., "9", "2", "multi-class")
7. key_challenges: Main difficulties (e.g., "high cardinality", "missing values", "class imbalance")

Respond in JSON format:
```json
{{
    "challenge_name": "specific challenge name or 'unknown'",
    "metric_name": "specific metric or 'accuracy'",
    "target_variable": "what is being predicted",
    "data_characteristics": "key data traits",
    "industry_domain": "business/scientific domain",
    "num_classes": "number or 'N/A'",
    "key_challenges": "main difficulties"
}}
```
"""

        response = self.call_llm(
            user_message=prompt,
            temperature=0.2,
            max_tokens=600
        )

        keywords = self.extract_json_from_response(response["content"])

        if keywords:
            self.log(f"  Extracted keywords: {keywords.get('challenge_name', 'unknown')}")
            return keywords
        else:
            return {
                "challenge_name": "unknown",
                "metric_name": "accuracy",
                "target_variable": "target",
                "data_characteristics": "mixed features",
                "industry_domain": "general",
                "num_classes": "N/A",
                "key_challenges": "unknown"
            }

    def _identify_task_type(self, problem_description: str) -> Dict[str, str]:
        """
        Identify task type and domain from problem description.

        Returns dict with 'task_type', 'domain', 'data_type'
        """
        prompt = f"""Analyze this ML task and identify its characteristics:

TASK:
{problem_description[:1000]}

Respond in JSON format:
```json
{{
    "task_type": "classification|regression|ranking|generation",
    "domain": "computer_vision|nlp|tabular|time_series|audio|multi_modal",
    "data_type": "images|text|structured|sequences|mixed",
    "specific_problem": "brief description"
}}
```
"""
        
        response = self.call_llm(
            user_message=prompt,
            temperature=0.3,
            max_tokens=500
        )
        
        # Parse JSON
        task_info = self.extract_json_from_response(response["content"])
        
        if task_info:
            return task_info
        else:
            # Fallback to keyword detection
            return self._fallback_task_detection(problem_description)
    
    def _fallback_task_detection(self, description: str) -> Dict[str, str]:
        """Simple keyword-based task detection"""
        description_lower = description.lower()
        
        # Detect task type
        if any(word in description_lower for word in ["classify", "classification", "predict class"]):
            task_type = "classification"
        elif any(word in description_lower for word in ["regression", "predict value", "estimate"]):
            task_type = "regression"
        else:
            task_type = "classification"  # default
        
        # Detect domain
        if any(word in description_lower for word in ["image", "photo", "picture", "visual"]):
            domain = "computer_vision"
            data_type = "images"
        elif any(word in description_lower for word in ["text", "language", "nlp", "sentiment"]):
            domain = "nlp"
            data_type = "text"
        elif any(word in description_lower for word in ["time series", "temporal", "sequence"]):
            domain = "time_series"
            data_type = "sequences"
        else:
            domain = "tabular"
            data_type = "structured"
        
        return {
            "task_type": task_type,
            "domain": domain,
            "data_type": data_type,
            "specific_problem": "General ML task"
        }
    
    def _generate_search_queries(
        self,
        task_info: Dict[str, str],
        problem_keywords: Optional[Dict[str, str]] = None,
        problem_description: Optional[str] = None
    ) -> List[str]:
        """
        Generate targeted academic search queries using LLM based on full problem context.

        The LLM analyzes the problem and generates optimal search queries
        for Semantic Scholar, ArXiv, and Papers With Code.

        Returns list of search query strings.
        """
        # Build context for LLM
        pk = problem_keywords or {}
        domain = task_info.get("domain", "machine learning")
        task_type = task_info.get("task_type", "classification")

        prompt = f"""Generate 5 highly specific academic search queries for finding state-of-the-art ML models to solve this challenge.

PROBLEM CONTEXT:
- Challenge Name: {pk.get('challenge_name', 'unknown')}
- Task Type: {task_type}
- Domain: {domain}
- Evaluation Metric: {pk.get('metric_name', 'accuracy')}
- Target Variable: {pk.get('target_variable', 'unknown')}
- Data Characteristics: {pk.get('data_characteristics', 'unknown')}
- Industry/Domain: {pk.get('industry_domain', 'general')}
- Number of Classes: {pk.get('num_classes', 'N/A')}
- Key Challenges: {pk.get('key_challenges', 'none')}

SEARCH TARGETS:
The queries will search Semantic Scholar, ArXiv, and Papers With Code.

REQUIREMENTS:
1. Each query should be 4-8 words, optimized for academic paper search
2. Queries must be SPECIFIC to this problem, not generic
3. Include queries for:
   - The specific challenge/competition if known (e.g., "Otto Group product classification Kaggle")
   - The evaluation metric optimization (e.g., "logloss minimization multiclass neural network")
   - The specific data challenges mentioned (e.g., "high cardinality categorical embedding")
   - State-of-the-art models for this exact domain
   - Efficient/practical models for this task type
4. DO NOT use generic queries like "machine learning classification" or "deep learning model"
5. Each query should target different aspects of the problem

Respond with a JSON array of exactly 5 query strings:
```json
["query 1", "query 2", "query 3", "query 4", "query 5"]
```
"""

        response = self.call_llm(
            user_message=prompt,
            temperature=0.4,
            max_tokens=500
        )

        # Parse JSON array from response
        queries = self._parse_query_list(response["content"])

        if queries and len(queries) >= 3:
            self.log(f"  LLM generated {len(queries)} targeted queries")
            for i, q in enumerate(queries[:5], 1):
                self.log(f"    {i}. {q}")
            return queries[:5]
        else:
            # Fallback to template-based queries if LLM fails
            self.log("  LLM query generation failed, using fallback")
            return self._fallback_search_queries(task_info, pk)

    def _parse_query_list(self, response: str) -> List[str]:
        """Parse a JSON array of query strings from LLM response."""
        # Try to extract JSON array
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', response, re.DOTALL)

        if json_match:
            try:
                queries = json.loads(json_match.group(1))
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    return [q.strip() for q in queries if q.strip()]
            except json.JSONDecodeError:
                pass

        # Fallback: try raw JSON
        try:
            queries = json.loads(response)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return [q.strip() for q in queries if q.strip()]
        except json.JSONDecodeError:
            pass

        return []

    def _fallback_search_queries(
        self,
        task_info: Dict[str, str],
        problem_keywords: Dict[str, str]
    ) -> List[str]:
        """
        Fallback template-based queries when LLM generation fails.
        """
        domain = task_info.get("domain", "machine learning")
        task_type = task_info.get("task_type", "classification")

        pk = problem_keywords
        challenge_name = pk.get("challenge_name", "").lower()
        metric_name = pk.get("metric_name", "accuracy")
        key_challenges = pk.get("key_challenges", "")

        queries = []

        # Challenge-specific
        if challenge_name and challenge_name != "unknown":
            queries.append(f"{challenge_name} {task_type} winning solution")

        # Metric-specific
        if metric_name and metric_name not in ["accuracy", "unknown"]:
            queries.append(f"{metric_name} optimization {domain} neural network")

        # Domain-specific SOTA (generic, no specific model names)
        if domain == "tabular":
            queries.append("state-of-the-art tabular data machine learning")
            queries.append("best methods structured data classification regression")
        elif domain == "computer_vision":
            queries.append("state-of-the-art image classification deep learning")
        elif domain == "nlp":
            queries.append("state-of-the-art text classification natural language")
        elif domain == "time_series":
            queries.append("state-of-the-art time series forecasting methods")
        else:
            queries.append(f"state-of-the-art {domain} {task_type}")

        # Challenge-specific queries
        if "imbalance" in key_challenges.lower():
            queries.append("class imbalance focal loss oversampling")
        if "categorical" in key_challenges.lower() or "cardinality" in key_challenges.lower():
            queries.append("categorical embedding high cardinality")

        # Efficient models
        queries.append(f"efficient {domain} model fast inference")

        return queries[:5]
    
    def _perform_searches(self, queries: List[str]) -> List[Dict]:
        """
        Perform real academic searches via Semantic Scholar, ArXiv, Papers With Code.

        Falls back to knowledge base if all API calls fail.
        """
        all_results = []

        for query in queries:
            self.log(f"  Searching: {query[:50]}...")
            try:
                # Real academic search (rate-limited internally)
                academic_results = search_all_academic_sources(
                    query=query,
                    limit_per_source=3,
                )
                if academic_results:
                    self.log(f"    Found {len(academic_results)} papers")
                    all_results.extend(academic_results)
                else:
                    # Fallback to knowledge base for this query
                    self.log("    No academic results, using knowledge base fallback")
                    fallback = self._get_knowledge_based_results(query)
                    all_results.extend(fallback)
            except Exception as e:
                self.log(f"    Academic search error: {e}, using fallback")
                fallback = self._get_knowledge_based_results(query)
                all_results.extend(fallback)

        # If still empty after all queries, use full knowledge base
        if not all_results:
            self.log("  All searches failed, using full knowledge base")
            for query in queries:
                all_results.extend(self._get_knowledge_based_results(query))

        return all_results
    
    def _get_knowledge_based_results(self, query: str) -> List[Dict]:
        """
        Knowledge-based fallback when web search fails.

        Returns generic guidance without specific model names,
        letting the LLM decide the best approach based on the problem.
        """
        query_lower = query.lower()
        results = []

        # Computer Vision
        if any(word in query_lower for word in ["image", "vision", "cnn", "visual"]):
            results.append({
                "title": "Deep Learning for Computer Vision",
                "snippet": "For image tasks, consider convolutional architectures or vision transformers. Choice depends on dataset size, computational budget, and whether transfer learning is applicable.",
                "model_name": "To be determined by problem analysis",
                "type": "Computer Vision"
            })

        # NLP
        elif any(word in query_lower for word in ["text", "nlp", "language", "transformer"]):
            results.append({
                "title": "Natural Language Processing Methods",
                "snippet": "For text tasks, transformer-based models or simpler approaches like TF-IDF with classical ML may work. Choice depends on data size and complexity.",
                "model_name": "To be determined by problem analysis",
                "type": "NLP"
            })

        # Tabular data
        elif any(word in query_lower for word in ["tabular", "structured", "classification", "regression"]):
            results.append({
                "title": "Tabular Data Methods",
                "snippet": "For structured/tabular data, tree-based methods, linear models, or neural approaches can work. Choice depends on data characteristics, feature types, and interpretability needs.",
                "model_name": "To be determined by problem analysis",
                "type": "Tabular"
            })

        # Time series
        elif any(word in query_lower for word in ["time series", "temporal", "forecast", "sequence"]):
            results.append({
                "title": "Time Series Methods",
                "snippet": "For temporal data, consider sequence models, attention mechanisms, or classical statistical methods. Choice depends on seasonality, trend patterns, and forecast horizon.",
                "model_name": "To be determined by problem analysis",
                "type": "Time Series"
            })

        # Default - no specific suggestions
        else:
            results.append({
                "title": "Machine Learning Approach",
                "snippet": "Analyze the problem characteristics (data type, size, target variable) to select an appropriate model family. Start simple and iterate based on results.",
                "model_name": "To be determined by problem analysis",
                "type": "General ML"
            })

        return results
    
    def _extract_model_candidates(
        self,
        search_results: List[Dict],
        task_info: Dict[str, str],
        num_candidates: int,
        problem_keywords: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """
        Extract and rank model candidates from search results.

        Uses LLM to synthesize information and propose candidates,
        now with problem-specific context for better relevance.
        """
        # Format search results
        results_text = self._format_search_results(search_results)

        # Build problem context section
        pk = problem_keywords or {}
        problem_context = ""
        if pk:
            problem_context = f"""
PROBLEM CONTEXT:
- Challenge: {pk.get('challenge_name', 'unknown')}
- Metric to optimize: {pk.get('metric_name', 'accuracy')}
- Target: {pk.get('target_variable', 'target prediction')}
- Data characteristics: {pk.get('data_characteristics', 'mixed features')}
- Industry/Domain: {pk.get('industry_domain', 'general')}
- Number of classes: {pk.get('num_classes', 'N/A')}
- Key challenges: {pk.get('key_challenges', 'none specified')}
"""

        prompt = f"""Based on these search results about ML models, propose {num_candidates} model candidates for this SPECIFIC task.

TASK INFORMATION:
- Task Type: {task_info.get('task_type', 'unknown')}
- Domain: {task_info.get('domain', 'unknown')}
- Data Type: {task_info.get('data_type', 'unknown')}
{problem_context}
SEARCH RESULTS:
{results_text}

Propose {num_candidates} distinct model architectures SPECIFICALLY suited for this problem. For each, provide:
1. Model name
2. Reasoning (why it fits THIS SPECIFIC task and metric)
3. Description (architecture overview and why it addresses the key challenges)

Prioritize:
- Optimization for the SPECIFIC METRIC ({pk.get('metric_name', 'accuracy')})
- Handling of the stated data characteristics
- Efficiency (training speed, memory)
- Diversity (different algorithmic families: tree-based, neural, instance-based)

Respond in JSON format:
```json
[
    {{
        "name": "ModelName",
        "reasoning": "Why this model is appropriate for THIS task and metric",
        "description": "Brief architecture description and key features"
    }}
]
```
"""
        
        response = self.call_llm(
            user_message=prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Parse JSON array
        candidates = self._parse_model_candidates(response["content"])
        
        if candidates and len(candidates) >= num_candidates:
            return candidates[:num_candidates]
        else:
            # Fallback: synthesize from search results
            return self._synthesize_candidates_from_results(
                search_results,
                task_info,
                num_candidates
            )
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results for prompt"""
        formatted = []
        for i, result in enumerate(results[:10], 1):  # Limit to 10 results
            formatted.append(f"[{i}] {result.get('title', 'Unknown')}")
            formatted.append(f"    {result.get('snippet', 'No description')}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _parse_model_candidates(self, response: str) -> List[Dict[str, str]]:
        """Parse model candidates from LLM response"""
        # Try to extract JSON array
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', response, re.DOTALL)
        
        if json_match:
            try:
                candidates = json.loads(json_match.group(1))
                if isinstance(candidates, list):
                    return candidates
            except json.JSONDecodeError:
                pass
        
        # Fallback: try raw JSON
        try:
            candidates = json.loads(response)
            if isinstance(candidates, list):
                return candidates
        except json.JSONDecodeError:
            pass
        
        return []
    
    def _synthesize_candidates_from_results(
        self,
        results: List[Dict],
        task_info: Dict[str, str],
        num_candidates: int
    ) -> List[Dict[str, str]]:
        """Synthesize candidates directly from search results"""
        candidates = []
        seen_models = set()
        
        for result in results:
            model_name = result.get("model_name", "Unknown")
            
            if model_name in seen_models or model_name == "Unknown":
                continue
            
            candidates.append({
                "name": model_name,
                "reasoning": f"Effective for {task_info.get('domain', 'ML')} tasks",
                "description": result.get("snippet", "No description available")
            })
            
            seen_models.add(model_name)
            
            if len(candidates) >= num_candidates:
                break
        
        return candidates