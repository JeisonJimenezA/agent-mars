# MARS Framework

**Modular Agent with Reflective Search** - Sistema autónomo para resolver desafíos de Machine Learning usando Monte Carlo Tree Search (MCTS).

## Estructura del Proyecto

```
mars/
├── main.py                 # Punto de entrada principal
├── orchestrator.py         # Coordinador del flujo MCTS
├── requirements.txt        # Dependencias
├── .env.example            # Template de configuración
│
├── agents/                 # Agentes especializados
│   ├── base_agent.py       # Clase base para agentes
│   ├── idea_agent.py       # Genera ideas de solución
│   ├── coding_agent.py     # Implementa código
│   ├── debug_agent.py      # Analiza y corrige errores
│   ├── review_agent.py     # Revisa resultados de ejecución
│   ├── modular_agent.py    # Descompone ideas en módulos
│   ├── search_agent.py     # Búsqueda académica
│   ├── solution_improver.py # Mejora soluciones existentes
│   └── validation_agent.py # Verifica validación y data leakage
│
├── core/                   # Componentes centrales
│   ├── config.py           # Configuración global
│   ├── mcts.py             # Motor MCTS
│   ├── tree_node.py        # Nodos del árbol de búsqueda
│   └── challenge_loader.py # Carga de desafíos
│
├── execution/              # Ejecución de soluciones
│   ├── executor.py         # Ejecuta scripts Python
│   ├── validator.py        # Valida soluciones
│   └── diff_editor.py      # Edición incremental de código
│
├── llm/                    # Integración con LLMs
│   ├── deepseek_client.py  # Cliente DeepSeek/Anthropic
│   ├── prompt_manager.py   # Gestión de prompts
│   └── prompts/            # Templates de prompts
│
├── memory/                 # Sistema de lecciones
│   ├── lesson_pool.py      # Pool de lecciones aprendidas
│   ├── lesson_extractor.py # Extrae lecciones de ejecuciones
│   └── lesson_types.py     # Tipos de lecciones
│
├── mle/                    # Preparación ML
│   ├── eda_agent.py        # Análisis exploratorio
│   └── task_prep.py        # Preparación de tareas
│
├── utils/                  # Utilidades
│   ├── file_manager.py     # Gestión de archivos
│   ├── code_parser.py      # Parsing de código Python
│   └── academic_search.py  # Búsqueda en papers
│
├── challenges/             # Definiciones de desafíos
│   └── otto_group.txt      # Ejemplo: Otto Group Classification
│
├── data/                   # Datos de desafíos (gitignored)
├── outputs/                # Salidas generadas (gitignored)
├── logs/                   # Logs de ejecución (gitignored)
└── working/                # Directorio de trabajo (gitignored)
```

## Requisitos

- Python 3.10+
- API Key de DeepSeek o Anthropic

## Instalación

```bash
# Clonar repositorio
git clone <url-del-repo>
cd mars

# Crear entorno virtual
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

## Configuración

Edita el archivo `.env`:

```env
# API Keys (usar uno de los dos)
DEEPSEEK_API_KEY=tu-api-key
ANTHROPIC_API_KEY=tu-api-key
USE_ANTHROPIC=false  # true para usar Anthropic

# Directorios
WORKING_DIR=./working
OUTPUT_DIR=./outputs
LOG_DIR=./logs

# Hiperparámetros MCTS
MCTS_KM=30       # Máximo de iteraciones
MCTS_ND=10       # Intentos de debug por nodo
MCTS_NI=2        # Mejoras por nodo válido
MAX_EXECUTION_TIME=7200
```

## Uso

### Ejecutar un desafío

```bash
python main.py --challenge otto_group --data-dir ./data/otto-group --time-budget 3600
```

**Parámetros:**
- `--challenge`: Nombre del desafío (debe existir en `challenges/`)
- `--data-dir`: Directorio con los datos CSV
- `--time-budget`: Tiempo máximo en segundos (default: 3600)
- `--output-dir`: Directorio de salida (default: ./working)

### Crear un nuevo desafío

1. Crear archivo en `challenges/nombre_desafio.txt` con:
   - Objetivo
   - Estructura del dataset
   - Métrica de evaluación
   - Formato de submission

2. Colocar datos en `data/nombre-desafio/`:
   - `train.csv`
   - `test.csv`

### Salidas

Después de ejecutar, encontrarás:

```
working/
├── metadata/           # EDA y splits de datos
├── best_solution/      # Mejor solución encontrada
│   ├── main.py
│   ├── *.py            # Módulos generados
│   └── solution_info.json
├── lessons.json        # Lecciones aprendidas
└── mcts_log.json       # Log del árbol MCTS
```

## Flujo de Trabajo

1. **Carga**: Lee el desafío y prepara los datos
2. **EDA**: Análisis exploratorio automático
3. **MCTS**: Búsqueda de soluciones
   - Genera ideas (IdeaAgent)
   - Descompone en módulos (ModularAgent)
   - Implementa código (CodingAgent)
   - Ejecuta y valida (Executor)
   - Debug si falla (DebugAgent)
   - Extrae lecciones (LessonExtractor)
4. **Mejora**: Itera sobre soluciones válidas
5. **Resultado**: Guarda la mejor solución

## Licencia

MIT
