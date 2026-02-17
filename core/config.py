# core/config.py
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class Config:
    """Global configuration for MARS framework"""
    
    # ============================================================================
    # API Configuration
    # ============================================================================
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # API Limits - Claude Sonnet 4.5 max output = 16000 (modo largo)
    # Claude Opus 4.5 soporta hasta 32000 tokens de salida
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "16000"))
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    
    # ============================================================================
    # Directory Structure
    # ============================================================================
    ROOT_DIR: Path = Path(__file__).parent.parent
    WORKING_DIR: Path = ROOT_DIR / os.getenv("WORKING_DIR", "working")
    OUTPUT_DIR: Path = ROOT_DIR / os.getenv("OUTPUT_DIR", "outputs")
    LOG_DIR: Path = ROOT_DIR / os.getenv("LOG_DIR", "logs")
    PROMPT_DIR: Path = ROOT_DIR / "llm" / "prompts"
    
    # ============================================================================
    # MCTS Hyperparameters (from paper)
    # ============================================================================
    KM: int = int(os.getenv("MCTS_KM", "30"))  # Max lessons in memory
    ND: int = int(os.getenv("MCTS_ND", "10"))  # Max debugging attempts
    NI: int = int(os.getenv("MCTS_NI", "2"))   # Branching factor
    W: float = float(os.getenv("MCTS_W", "-0.07"))  # Latency penalty weight
    C_UCT: float = float(os.getenv("MCTS_CUCT", "1.414"))  # UCT constant
    NS: int = 5  # Valid nodes without improvement before reactivating root
    LOWER_IS_BETTER: bool = False  # Metric direction (set per-task)
    
    # ============================================================================
    # Execution Settings
    # ============================================================================
    MAX_EXECUTION_TIME: int = int(os.getenv("MAX_EXECUTION_TIME", "3600"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    RANDOM_SEED: int = 42
    
    # ============================================================================
    # File Management
    # ============================================================================
    METADATA_DIR: str = "metadata"
    CACHE_DIR: str = "cache"
    SUBMISSION_DIR: str = "submission"
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for directory in [cls.WORKING_DIR, cls.OUTPUT_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories in working dir
        (cls.WORKING_DIR / cls.METADATA_DIR).mkdir(exist_ok=True)
        (cls.WORKING_DIR / cls.CACHE_DIR).mkdir(exist_ok=True)
        (cls.WORKING_DIR / cls.SUBMISSION_DIR).mkdir(exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate critical configuration"""
        # Check that at least one LLM provider is configured
        provider = os.getenv("LLM_PROVIDER", "").lower()
        has_provider = False

        # Ollama (local) - no API key needed
        if provider == "ollama" or os.getenv("USE_OLLAMA", "").lower() == "true":
            has_provider = True
        if provider == "deepseek" or os.getenv("DEEPSEEK_API_KEY"):
            has_provider = True
        if provider == "openai" or os.getenv("OPENAI_API_KEY"):
            has_provider = True
        if provider == "anthropic" or os.getenv("ANTHROPIC_API_KEY"):
            has_provider = True
        if provider == "gemini" or os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY"):
            has_provider = True

        if not has_provider:
            raise ValueError(
                "No LLM provider configured. Set LLM_PROVIDER and the corresponding API key:\n"
                "  - ollama: No API key needed (local)\n"
                "  - deepseek: DEEPSEEK_API_KEY\n"
                "  - openai: OPENAI_API_KEY\n"
                "  - anthropic: ANTHROPIC_API_KEY\n"
                "  - gemini: GOOGLE_GENAI_API_KEY or GEMINI_API_KEY"
            )

        if not cls.PROMPT_DIR.exists():
            raise ValueError(f"Prompt directory not found: {cls.PROMPT_DIR}")
    
    @classmethod
    def to_dict(cls) -> dict:
        """Export config as dictionary"""
        return {
            "api": {
                "model": cls.DEEPSEEK_MODEL,
                "max_tokens": cls.MAX_TOKENS,
                "temperature": cls.TEMPERATURE,
            },
            "mcts": {
                "Km": cls.KM,
                "Nd": cls.ND,
                "Ni": cls.NI,
                "w": cls.W,
                "c_uct": cls.C_UCT,
            },
            "execution": {
                "max_time": cls.MAX_EXECUTION_TIME,
                "debug_mode": cls.DEBUG_MODE,
            }
        }
    
    @classmethod
    def save_to_file(cls, filepath: Path):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(cls.to_dict(), f, indent=2)


# Initialize directories on import
Config.setup_directories()