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
        
        # Step 1: Identify task type and domain
        task_info = self._identify_task_type(problem_description)
        self.log(f"  Task type: {task_info.get('task_type', 'unknown')}")
        self.log(f"  Domain: {task_info.get('domain', 'unknown')}")
        
        # Step 2: Generate search queries
        queries = self._generate_search_queries(task_info)
        self.log(f"  Generated {len(queries)} search queries")
        
        # Step 3: Perform searches (simulated for now)
        # In production, this would call actual web_search tool
        search_results = self._perform_searches(queries)
        
        # Step 4: Extract and rank model candidates
        candidates = self._extract_model_candidates(
            search_results,
            task_info,
            num_candidates
        )
        
        self.log(f"  Found {len(candidates)} model candidates")
        for i, candidate in enumerate(candidates, 1):
            self.log(f"    {i}. {candidate['name']}")
        
        return candidates
    
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
    
    def _generate_search_queries(self, task_info: Dict[str, str]) -> List[str]:
        """
        Generate academic search queries based on task characteristics.

        Optimized for Semantic Scholar, ArXiv, and Papers With Code.
        Returns list of search query strings.
        """
        domain = task_info.get("domain", "machine learning")
        task_type = task_info.get("task_type", "classification")
        data_type = task_info.get("data_type", "data")

        queries = []

        # Query 1: Domain + task SOTA (academic style)
        queries.append(f"state-of-the-art {domain} {task_type} deep learning")

        # Query 2: Domain-specific academic queries
        if domain == "computer_vision":
            queries.append("image classification convolutional neural network benchmark")
        elif domain == "nlp":
            queries.append("transformer text classification BERT language model")
        elif domain == "tabular":
            queries.append("gradient boosting tabular data XGBoost LightGBM CatBoost")
        elif domain == "time_series":
            queries.append("time series forecasting LSTM transformer temporal")
        else:
            queries.append(f"{data_type} {task_type} neural network architecture")

        # Query 3: Efficient/lightweight models for practical deployment
        queries.append(f"efficient {domain} model architecture low latency")

        return queries[:3]  # Limit to 3 queries for rate limiting
    
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
        Knowledge-based model suggestions (fallback when no web search).
        
        This provides reasonable defaults based on common patterns.
        """
        query_lower = query.lower()
        results = []
        
        # Computer Vision models
        if any(word in query_lower for word in ["image", "vision", "cnn", "visual"]):
            results.extend([
                {
                    "title": "ResNet - Deep Residual Learning",
                    "snippet": "ResNet uses skip connections to enable training of very deep networks. Variants include ResNet50, ResNet101. Efficient and widely used for image classification.",
                    "model_name": "ResNet",
                    "type": "CNN"
                },
                {
                    "title": "EfficientNet - Efficient CNN Architecture",
                    "snippet": "EfficientNet achieves state-of-the-art accuracy with fewer parameters through compound scaling. Excellent for resource-constrained environments.",
                    "model_name": "EfficientNet",
                    "type": "CNN"
                },
                {
                    "title": "Vision Transformer (ViT)",
                    "snippet": "Vision Transformers apply transformer architecture to images. Competitive with CNNs on large datasets.",
                    "model_name": "ViT",
                    "type": "Transformer"
                }
            ])
        
        # NLP models
        elif any(word in query_lower for word in ["text", "nlp", "language", "transformer"]):
            results.extend([
                {
                    "title": "BERT - Bidirectional Encoder Representations",
                    "snippet": "BERT uses bidirectional transformers for text understanding. Excellent for classification and NLU tasks.",
                    "model_name": "BERT",
                    "type": "Transformer"
                },
                {
                    "title": "DistilBERT - Lightweight BERT",
                    "snippet": "DistilBERT is a smaller, faster version of BERT with 97% of performance. Good for production deployments.",
                    "model_name": "DistilBERT",
                    "type": "Transformer"
                },
                {
                    "title": "RoBERTa - Robustly Optimized BERT",
                    "snippet": "RoBERTa improves BERT training with better hyperparameters and more data. Higher accuracy than base BERT.",
                    "model_name": "RoBERTa",
                    "type": "Transformer"
                }
            ])
        
        # Tabular data models
        elif any(word in query_lower for word in ["tabular", "structured", "gradient boost"]):
            results.extend([
                {
                    "title": "XGBoost - Extreme Gradient Boosting",
                    "snippet": "XGBoost is highly efficient gradient boosting library. Dominates tabular data competitions. Handles mixed data types well.",
                    "model_name": "XGBoost",
                    "type": "GBM"
                },
                {
                    "title": "LightGBM - Light Gradient Boosting",
                    "snippet": "LightGBM uses histogram-based learning for faster training. Memory efficient and scales well to large datasets.",
                    "model_name": "LightGBM",
                    "type": "GBM"
                },
                {
                    "title": "CatBoost - Categorical Boosting",
                    "snippet": "CatBoost handles categorical features natively without encoding. Robust to overfitting and requires less tuning.",
                    "model_name": "CatBoost",
                    "type": "GBM"
                }
            ])
        
        # Time series models
        elif any(word in query_lower for word in ["time series", "temporal", "lstm", "forecast"]):
            results.extend([
                {
                    "title": "LSTM - Long Short-Term Memory",
                    "snippet": "LSTMs are RNNs designed for sequence learning. Effective for time series forecasting and temporal patterns.",
                    "model_name": "LSTM",
                    "type": "RNN"
                },
                {
                    "title": "Temporal Fusion Transformer",
                    "snippet": "TFT combines LSTM and attention mechanisms for multi-horizon forecasting. State-of-the-art for complex time series.",
                    "model_name": "TFT",
                    "type": "Transformer"
                }
            ])
        
        # Default fallback
        else:
            results.extend([
                {
                    "title": "Random Forest - Ensemble Learning",
                    "snippet": "Random Forest is robust ensemble method. Good baseline for many tasks with minimal tuning.",
                    "model_name": "RandomForest",
                    "type": "Ensemble"
                },
                {
                    "title": "Neural Network - Multi-Layer Perceptron",
                    "snippet": "MLPs are flexible neural networks. Can learn complex patterns with sufficient data.",
                    "model_name": "MLP",
                    "type": "Neural Network"
                }
            ])
        
        return results
    
    def _extract_model_candidates(
        self,
        search_results: List[Dict],
        task_info: Dict[str, str],
        num_candidates: int
    ) -> List[Dict[str, str]]:
        """
        Extract and rank model candidates from search results.
        
        Uses LLM to synthesize information and propose candidates.
        """
        # Format search results
        results_text = self._format_search_results(search_results)
        
        prompt = f"""Based on these search results about ML models, propose {num_candidates} model candidates for this task.

TASK INFORMATION:
- Task Type: {task_info.get('task_type', 'unknown')}
- Domain: {task_info.get('domain', 'unknown')}
- Data Type: {task_info.get('data_type', 'unknown')}

SEARCH RESULTS:
{results_text}

Propose {num_candidates} distinct model architectures. For each, provide:
1. Model name
2. Reasoning (why it fits this task)
3. Description (architecture overview)

Prioritize:
- Efficiency (training speed, memory)
- Proven effectiveness for this domain
- Diversity (different algorithmic families)

Respond in JSON format:
```json
[
    {{
        "name": "ModelName",
        "reasoning": "Why this model is appropriate for the task",
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