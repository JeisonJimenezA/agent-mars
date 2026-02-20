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
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")

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
        provider = os.getenv("LLM_PROVIDER", "").lower()
        has_provider = False

        # Ollama (local) - no API key needed
        if provider == "ollama" or os.getenv("USE_OLLAMA", "").lower() == "true":
            has_provider = True

        # OpenRouter (cloud)
        if provider == "openrouter" or os.getenv("OPENROUTER_API_KEY"):
            has_provider = True

        if not has_provider:
            raise ValueError(
                "No LLM provider configured. Set LLM_PROVIDER and the corresponding API key:\n"
                "  - openrouter: OPENROUTER_API_KEY (acceso a todos los modelos cloud)\n"
                "  - ollama: No API key needed (local)"
            )

        if not cls.PROMPT_DIR.exists():
            raise ValueError(f"Prompt directory not found: {cls.PROMPT_DIR}")

    @classmethod
    def to_dict(cls) -> dict:
        """Export config as dictionary"""
        return {
            "api": {
                "model": cls.OPENROUTER_MODEL,
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
