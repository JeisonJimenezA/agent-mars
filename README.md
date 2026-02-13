# MARS Framework

**Modular Agent with Reflective Search** - Autonomous system for solving Machine Learning challenges using Monte Carlo Tree Search (MCTS).

## Project Structure

```
mars/
├── main.py                 # Main entry point
├── orchestrator.py         # MCTS flow coordinator
├── requirements.txt        # Dependencies
├── .env.example            # Configuration template
│
├── agents/                 # Specialized agents
│   ├── base_agent.py       # Base class for agents
│   ├── idea_agent.py       # Generates solution ideas
│   ├── coding_agent.py     # Implements code
│   ├── debug_agent.py      # Analyzes and fixes errors
│   ├── review_agent.py     # Reviews execution results
│   ├── modular_agent.py    # Decomposes ideas into modules
│   ├── search_agent.py     # Academic search
│   ├── solution_improver.py # Improves existing solutions
│   └── validation_agent.py # Verifies validation and data leakage
│
├── core/                   # Core components
│   ├── config.py           # Global configuration
│   ├── mcts.py             # MCTS engine
│   ├── tree_node.py        # Search tree nodes
│   └── challenge_loader.py # Challenge loader
│
├── execution/              # Solution execution
│   ├── executor.py         # Executes Python scripts
│   ├── validator.py        # Validates solutions
│   └── diff_editor.py      # Incremental code editing
│
├── llm/                    # LLM integration
│   ├── deepseek_client.py  # DeepSeek/Anthropic client
│   ├── prompt_manager.py   # Prompt management
│   └── prompts/            # Prompt templates
│
├── memory/                 # Lesson system
│   ├── lesson_pool.py      # Learned lessons pool
│   ├── lesson_extractor.py # Extracts lessons from executions
│   └── lesson_types.py     # Lesson types
│
├── mle/                    # ML preparation
│   ├── eda_agent.py        # Exploratory data analysis
│   └── task_prep.py        # Task preparation
│
├── utils/                  # Utilities
│   ├── file_manager.py     # File management
│   ├── code_parser.py      # Python code parsing
│   └── academic_search.py  # Paper search
│
├── challenges/             # Challenge definitions
│   └── otto_group.txt      # Example: Otto Group Classification
│
├── data/                   # Challenge data (gitignored)
├── outputs/                # Generated outputs (gitignored)
├── logs/                   # Execution logs (gitignored)
└── working/                # Working directory (gitignored)
```

## Requirements

- Python 3.10+
- DeepSeek or Anthropic API Key

## Installation

```bash
# Clone repository
git clone <repo-url>
cd mars

# Create virtual environment
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Edit the `.env` file:

```env
# API Keys (use one of the two)
DEEPSEEK_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-api-key
USE_ANTHROPIC=false  # true to use Anthropic

# Directories
WORKING_DIR=./working
OUTPUT_DIR=./outputs
LOG_DIR=./logs

# MCTS Hyperparameters
MCTS_KM=30       # Maximum iterations
MCTS_ND=10       # Debug attempts per node
MCTS_NI=2        # Improvements per valid node
MAX_EXECUTION_TIME=7200
```

## Usage

### Run a challenge

```bash
python main.py --challenge otto_group --data-dir ./data/otto-group --time-budget 3600
```

**Parameters:**
- `--challenge`: Challenge name (must exist in `challenges/`)
- `--data-dir`: Directory with CSV data
- `--time-budget`: Maximum time in seconds (default: 3600)
- `--output-dir`: Output directory (default: ./working)

### Create a new challenge

1. Create a file in `challenges/challenge_name.txt` with:
   - Objective
   - Dataset structure
   - Evaluation metric
   - Submission format

2. Place data in `data/challenge-name/`:
   - `train.csv`
   - `test.csv`

### Outputs

After execution, you will find:

```
working/
├── metadata/           # EDA and data splits
├── best_solution/      # Best solution found
│   ├── main.py
│   ├── *.py            # Generated modules
│   └── solution_info.json
├── lessons.json        # Learned lessons
└── mcts_log.json       # MCTS tree log
```

## Workflow

1. **Load**: Reads the challenge and prepares data
2. **EDA**: Automatic exploratory analysis
3. **MCTS**: Solution search
   - Generates ideas (IdeaAgent)
   - Decomposes into modules (ModularAgent)
   - Implements code (CodingAgent)
   - Executes and validates (Executor)
   - Debugs if it fails (DebugAgent)
   - Extracts lessons (LessonExtractor)
4. **Improve**: Iterates on valid solutions
5. **Result**: Saves the best solution

## License

MIT
