# Scryptorum Framework Documentation

Scryptorum is a modern, annotation-driven framework for running and tracking LLM experiments. It's designed to work seamlessly with Python projects, allowing you to focus on your research while it handles experiment management, versioning, and logging.

## Overview

Key features include:
* **Dual run types**: Lightweight trial runs (logs only) vs milestone runs (full versioning)
* **Annotation-driven**: Use `@experiment`, `@metric`, `@timer`, `@llm_call` decorators for automatic tracking
* **CLI-configurable**: Same code runs as trial or milestone based on CLI flags
* **Thread-safe logging**: JSONL format for metrics, timings, and LLM invocations
* **Plugin architecture**: Ready for HTTP server and custom extensions
* **Modern Python**: Built for Python 3.13+ with type hints and clean dependencies

## Core Concepts

Scryptorum organizes your work into a three-level hierarchy:

### Project
A **project** is a workspace that contains multiple related experiments. Think of it as a research grant or lab.

- **Purpose**: Groups related experiments and provides shared resources
- **Contains**: Multiple experiments, shared data, models, and artifacts
- **Example**: "NLP for Healthcare", "Customer Service AI", "Code Analysis Tools"
- **Lifespan**: Long-lived, evolves over months/years

### Experiment  
An **experiment** represents a specific research question or hypothesis you're testing. Think of it as a focused study within your broader research.

- **Purpose**: Organizes all attempts at solving a particular problem
- **Contains**: Multiple runs, experiment-specific configs, prompts, and data
- **Examples**: "sentiment_analysis", "document_summarization", "code_generation"
- **Lifespan**: Medium-lived, contains many runs as you iterate and improve

### Run
A **run** is a single execution of an experiment - one specific attempt with particular parameters, data, or code.

- **Purpose**: Records the results of one specific attempt
- **Types**:
  - **Trial Run**: Quick iterations for development (logs only)
  - **Milestone Run**: Important versions with full snapshots (code, configs, artifacts)
- **Examples**: Testing different prompts, model parameters, or datasets
- **Lifespan**: Immutable once finished

### Integration Example
```
my_llm_project/                      ← Your Python Package
├── src/my_llm_project/
│   ├── experiments/                 ← Your experiment code
│   │   ├── sentiment_analysis.py
│   │   └── entity_extraction.py
│   ├── llm_helpers/                 ← Your reusable code
│   └── business_logic/
├── pyproject.toml                   ← Dependencies (including scryptorum)
├── .scryptorum                      ← Scryptorum config (points to experiments dir)
└── tests/

~/research_experiments/              ← Scryptorum Project (separate location)
├── my_llm_project/                  ← Project within experiments dir
│   ├── experiments/
│   │   ├── sentiment_analysis/      ← Experiment tracking
│   │   │   ├── runs/
│   │   │   │   ├── tr-20250526-143022/  ← Trial Run
│   │   │   │   └── ms-20250526-150000/  ← Milestone Run
│   │   │   ├── config/              ← Agent configs
│   │   │   └── prompts/             ← Prompt templates
│   │   └── entity_extraction/
│   └── data/                        ← Project-wide shared data
```

This structure helps you:
- **Organize** related work at the right level of granularity
- **Compare** different approaches to the same problem (runs within an experiment)
- **Track progress** across multiple research directions (experiments within a project)
- **Reproduce** important results (milestone runs preserve everything needed)

## Installation
```bash
# Clone and install in development mode
git clone <repository-url>
cd scryptorum
uv sync

# Install with optional dependencies
uv sync --extra dev      # Development tools (pytest, black, ruff)
uv sync --extra server   # HTTP server support (fastapi, uvicorn)

# Install the package
pip install -e .
```

## Quick Start

### 1. Set Up Your Python Project
Create your Python project as normal:
```bash
# Create your Python package
uv init my_llm_project
cd my_llm_project
uv add scryptorum
```

### 2. Initialize Scryptorum
Initialize scryptorum within your Python project:
```bash
# Initialize with default experiments directory (./experiments)
scryptorum init

# Or specify a custom experiments directory
scryptorum init --experiments-dir ~/my_research_experiments
```

### 3. Create an Experiment
Create your first experiment (scryptorum auto-detects the project):
```bash
scryptorum create-experiment sentiment_analysis
```

### 4. Write Your Experiment
Write your experiment code in your Python package:
```python
# src/my_llm_project/experiments/sentiment_analysis.py
from scryptorum import experiment, metric, timer, llm_call

@experiment(name="sentiment_analysis")
def main():
    """Analyze sentiment of customer reviews."""
    
    # Load and process data
    reviews = load_reviews()
    results = []
    
    for review in reviews:
        sentiment = analyze_sentiment(review)
        results.append(sentiment)
    
    # Calculate metrics
    accuracy = calculate_accuracy(results)
    return accuracy

@timer("data_loading")
def load_reviews():
    """Load customer reviews."""
    # Your data loading logic
    return ["Great product!", "Terrible service", "Love it!"]

@llm_call(model="gpt-4")
def analyze_sentiment(review: str) -> str:
    """Analyze sentiment using LLM."""
    # Your LLM call logic here
    prompt = f"Analyze sentiment of: {review}"
    # return llm_client.complete(prompt)
    return "positive"  # Placeholder

@metric(name="accuracy", metric_type="accuracy")
def calculate_accuracy(results):
    """Calculate sentiment accuracy."""
    # Your evaluation logic
    return 0.85
```

### 5. Run Your Experiment

```bash
# Trial run (lightweight, logs only) - for development and iteration
scryptorum run sentiment_analysis --module my_llm_project.experiments.sentiment_analysis

# Milestone run (full versioning and artifacts) - for important results  
scryptorum run sentiment_analysis --module my_llm_project.experiments.sentiment_analysis --milestone

# Or run as a Python module directly
python -m my_llm_project.experiments.sentiment_analysis
```

## Usage Patterns

### Pattern 1: Decorator-Based (Recommended)
Use decorators for automatic experiment tracking with minimal code changes.

```python
from scryptorum import experiment, metric, timer, llm_call

@experiment(name="my_experiment")
def run_experiment():
    data = prepare_data()
    results = process_data(data)
    score = evaluate(results)
    return score

@timer("data_preparation")
def prepare_data():
    # Automatically times this function
    return load_data_from_source()

@llm_call(model="gpt-4")
def call_llm(prompt: str):
    # Automatically logs LLM input/output
    return your_llm_client.complete(prompt)

@metric(name="f1_score", metric_type="f1")
def evaluate(results):
    # Automatically logs the return value as a metric
    return calculate_f1(results)
```

### Pattern 2: Class-Based
Extend `BaseRunnable` for more complex experiments with lifecycle methods.

```python
from scryptorum.execution.runner import BaseRunnable

class SentimentAnalysisRunnable(BaseRunnable):
    
    def prepare(self):
        """Setup resources and log experiment configuration."""
        self.model_config = self.config.get("model", {})
        self.run.log_event("model_configured", self.model_config)
        
    def run(self):
        """Execute the main experiment."""
        reviews = self.load_data()
        
        for review in reviews:
            sentiment = self.analyze_sentiment(review)
            self.run.log_llm_call(
                model="gpt-4",
                input_data=review,
                output_data=sentiment
            )
    
    def score(self):
        """Evaluate results and log metrics."""
        accuracy = self.calculate_accuracy()
        self.run.log_metric("accuracy", accuracy, "accuracy")
        
    def cleanup(self):
        """Clean up resources."""
        pass

# Run with CLI
# scryptorum run sentiment_analysis -m my_module.SentimentAnalysisRunnable
```

### Pattern 3: Manual API
Direct control over experiment and run management.

```python
from scryptorum import Experiment, RunType

# Create experiment manually
experiment = Experiment(".", "manual_experiment")
run = experiment.create_run(RunType.MILESTONE)

try:
    # Manual logging
    run.log_metric("custom_metric", 0.92, "f1_score")
    run.log_llm_call("gpt-3.5", "test prompt", "test response", 150.0)
    
    # Time operations manually
    with run.TimerContext(run, "data_processing"):
        process_data()
        
finally:
    run.finish()
```

## CLI Commands

### Project Integration
```bash
# Initialize scryptorum in existing Python project
scryptorum init [--experiments-dir <path>] [--project-name <name>]

# Create new standalone project (optional - for non-Python workflows)
scryptorum create-project <name> [--parent-dir <path>]
```

### Experiment Management
```bash
# Create experiment (auto-detects project from .scryptorum)
scryptorum create-experiment <name>

# List experiments (auto-detects project)
scryptorum list-experiments

# List runs in experiment (auto-detects project)  
scryptorum list-runs <experiment>

# Override auto-detection with explicit project
scryptorum create-experiment <name> --project-root <path>
```

### Running Experiments
```bash
# Run Python module as experiment (auto-detects project)
scryptorum run <experiment-name> --module <package.module>

# Run script as experiment (auto-detects project)
scryptorum run <experiment-name> --script <script.py>

# Milestone run with full versioning
scryptorum run <experiment-name> --module <package.module> --milestone

# Custom run ID (milestone only)
scryptorum run <experiment-name> --module <package.module> --milestone --run-id "v1.0"
```

## Run Types

### Trial Runs (Default)
- **Purpose**: Quick development and testing
- **Storage**: Single `trial_run` directory (reset each time)
- **Artifacts**: Logs only, no file preservation
- **Use case**: Development, debugging, parameter tuning

```bash
scryptorum run my_experiment -s script.py  # Trial run
```

### Milestone Runs
- **Purpose**: Production experiments and important results
- **Storage**: Versioned `run-<id>` directories
- **Artifacts**: Full preservation (code snapshots, artifacts, logs)
- **Use case**: Final experiments, reproducible results, model releases

```bash
scryptorum run my_experiment -s script.py --milestone  # Milestone run
```

## Available Decorators

### `@experiment(name="experiment_name")`
Marks a function as an experiment entry point.
- Automatically creates experiment and run context
- Run type controlled by CLI flags

### `@metric(name="metric_name", metric_type="accuracy")`
Automatically logs function return value as a metric.
```python
@metric(name="accuracy", metric_type="accuracy")
def calculate_accuracy(predictions, labels):
    return accuracy_score(predictions, labels)
```

### `@timer(name="operation_name")`
Automatically times function execution.
```python
@timer("model_inference")
def run_inference(model, data):
    return model.predict(data)
```

### `@llm_call(model="gpt-4", log_input=True, log_output=True)`
Automatically logs LLM invocations with timing.
```python
@llm_call(model="gpt-4")
def get_completion(prompt: str) -> str:
    return openai.Completion.create(prompt=prompt)
```

## Advanced Features

### Batch Processing
```python
from scryptorum.core.decorators import batch_processor

@batch_processor(batch_size=10, parallel=False)
def process_items(item):
    """Process items in batches with automatic logging."""
    return analyze_item(item)

# Usage
items = load_large_dataset()
results = process_items(items)  # Automatically batched and logged
```

### Plugin System
Scryptorum includes a plugin architecture for extensions:

```python
from scryptorum.plugins import registry, HTTPServerPlugin

# Register custom plugins
registry.register(MyCustomPlugin())

# Future: HTTP server for experiment visualization
# registry.register(HTTPServerPlugin())
```

## Logging Format

All logs use JSONL format for easy parsing and analysis:

### Run Logs (`run.jsonl`)
```json
{"timestamp": "2024-01-01T12:00:00", "event_type": "run_started", "run_id": "trial_run", "run_type": "trial"}
{"timestamp": "2024-01-01T12:00:01", "event_type": "llm_call", "model": "gpt-4", "input": "...", "output": "..."}
{"timestamp": "2024-01-01T12:00:05", "event_type": "run_finished", "duration_seconds": 5.2}
```

### Metrics (`metrics.jsonl`)
```json
{"timestamp": "2024-01-01T12:00:03", "run_id": "trial_run", "name": "accuracy", "value": 0.85, "type": "accuracy"}
{"timestamp": "2024-01-01T12:00:04", "run_id": "trial_run", "name": "f1_score", "value": 0.82, "type": "f1"}
```

### Timings (`timings.jsonl`)
```json
{"timestamp": "2024-01-01T12:00:02", "run_id": "trial_run", "operation": "data_loading", "duration_ms": 1250.5}
{"timestamp": "2024-01-01T12:00:03", "run_id": "trial_run", "operation": "model_inference", "duration_ms": 890.2}
```

## Best Practices

1. **Use trial runs for development**: Quick iteration without storage overhead
2. **Use milestone runs for important results**: Full versioning for reproducibility
3. **Leverage decorators**: Minimal code changes for maximum tracking
4. **Structure experiments logically**: Separate data loading, processing, and evaluation
5. **Log meaningful metrics**: Focus on business-relevant measurements
6. **Version your code**: Milestone runs automatically snapshot code for reproducibility

## Examples

See `examples/scryptorum/` for complete examples showing how the same code runs transparently in both trial and milestone modes.

## License

MIT License

Copyright (c) 2025 Scryptorum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.