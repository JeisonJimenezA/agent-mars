# MARS Framework

**Modular Agent with Reflective Search** - Sistema autónomo para resolver desafíos de Machine Learning usando Monte Carlo Tree Search (MCTS).

## Estructura del Proyecto

```
mars/
├── main.py                 # Punto de entrada principal
├── orchestrator.py         # Coordinador del flujo MCTS
├── requirements.txt        # Dependencias
├── .env.example            # Plantilla de configuración
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
│   └── validation_agent.py # Verifica validación y fuga de datos
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
│   └── prompts/            # Plantillas de prompts
│
├── memory/                 # Sistema de lecciones
│   ├── lesson_pool.py      # Repositorio de lecciones aprendidas
│   ├── lesson_extractor.py # Extrae lecciones de ejecuciones
│   └── lesson_types.py     # Tipos de lecciones
│
├── mle/                    # Preparación ML
│   ├── eda_agent.py        # Análisis exploratorio
│   └── task_prep.py        # Preparación de tareas
│
├── utils/                  # Utilidades
│   ├── file_manager.py     # Gestión de archivos
│   ├── code_parser.py      # Análisis de código Python
│   └── academic_search.py  # Búsqueda en papers
│
├── challenges/             # Definiciones de desafíos
│   └── otto_group.txt      # Ejemplo: Clasificación Otto Group
│
├── data/                   # Datos de desafíos (ignorado por git)
├── outputs/                # Salidas generadas (ignorado por git)
├── logs/                   # Registros de ejecución (ignorado por git)
└── working/                # Directorio de trabajo (ignorado por git)
```

## Requisitos

- Python 3.10+
- Clave API de DeepSeek o Anthropic

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
# Editar .env con tus claves API
```

## Configuración

Edita el archivo `.env`:

```env
# Claves API (usar una de las dos)
DEEPSEEK_API_KEY=tu-clave-api
ANTHROPIC_API_KEY=tu-clave-api
USE_ANTHROPIC=false  # true para usar Anthropic

# Directorios
WORKING_DIR=./working
OUTPUT_DIR=./outputs
LOG_DIR=./logs

# Hiperparámetros MCTS
MCTS_KM=30       # Máximo de iteraciones
MCTS_ND=10       # Intentos de depuración por nodo
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
- `--time-budget`: Tiempo máximo en segundos (por defecto: 3600)
- `--output-dir`: Directorio de salida (por defecto: ./working)

### Crear un nuevo desafío

1. Crear archivo en `challenges/nombre_desafio.txt` con:
   - Objetivo
   - Estructura del conjunto de datos
   - Métrica de evaluación
   - Formato de envío

2. Colocar datos en `data/nombre-desafio/`:
   - `train.csv`
   - `test.csv`

### Salidas

Después de ejecutar, encontrarás:

```
working/
├── metadata/           # EDA y divisiones de datos
├── best_solution/      # Mejor solución encontrada
│   ├── main.py
│   ├── *.py            # Módulos generados
│   └── solution_info.json
├── lessons.json        # Lecciones aprendidas
└── mcts_log.json       # Registro del árbol MCTS
```

## Flujo de Trabajo

1. **Carga**: Lee el desafío y prepara los datos
2. **EDA**: Análisis exploratorio automático
3. **MCTS**: Búsqueda de soluciones
   - Genera ideas (IdeaAgent)
   - Descompone en módulos (ModularAgent)
   - Implementa código (CodingAgent)
   - Ejecuta y valida (Executor)
   - Depura si falla (DebugAgent)
   - Extrae lecciones (LessonExtractor)
4. **Mejora**: Itera sobre soluciones válidas
5. **Resultado**: Guarda la mejor solución

## Licencia

MIT
