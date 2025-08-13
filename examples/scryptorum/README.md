# Scryptorum Example: Literary Agent Evaluation

This example demonstrates a complete evaluation experiment using the scryptorum framework to compare two character agents (pirate vs shakespeare) on literary comprehension tasks.

## Overview

The experiment evaluates how well different character agents can answer questions about literature and reading comprehension. It uses:

- **Two test agents**: A literature-hating pirate and a Shakespeare-inspired poet
- **Standardized dataset**: 5 literary questions with golden answers
- **Judge agent**: An impartial evaluator that scores responses
- **Comprehensive logging**: All interactions, metrics, and results are tracked

## Features Demonstrated

- ✅ **@experiment decorator**: Automatic experiment management and logging
- ✅ **@metric decorators**: Structured metric tracking for scores and comparisons  
- ✅ **@timer decorators**: Performance measurement of evaluation steps
- ✅ **Agent loading**: Using existing agent configurations from the agents example
- ✅ **Judge pattern**: Creating a specialized agent for scoring other agents
- ✅ **Dataset evaluation**: Systematic evaluation across multiple questions
- ✅ **Result persistence**: Saving detailed results to experiment data directory
- ✅ **Event logging**: Comprehensive tracking of experiment progress

## Dataset

The evaluation uses 5 literary questions across different categories:

1. **Theme Analysis** (Medium): Shakespeare's Romeo and Juliet themes
2. **Literature Value** (Easy): Importance of reading for personal development  
3. **Literary Devices** (Hard): How metaphors enhance storytelling
4. **Character Analysis** (Medium): What makes characters memorable
5. **Medium Comparison** (Easy): Books vs movies in storytelling

Each question includes:
- Golden answer for comparison
- Difficulty level (easy/medium/hard)
- Category classification
- Unique ID for tracking

## Scoring System

The judge agent evaluates responses on multiple dimensions:

- **Accuracy**: Factual correctness (1-10)
- **Depth**: Insight and analysis quality (1-10)  
- **Clarity**: Communication effectiveness (1-10)
- **Relevance**: Staying on topic (1-10)
- **Overall Score**: Combined assessment (1-10)

## Usage

### Run the Complete Experiment

```bash
# From project root
cd /Users/dan/dev/code/ai/dsat
python examples/scryptorum/literary_evaluation.py
```

### Run via Scryptorum CLI (Trial Mode)

```bash
python -m scryptorum.cli.commands run examples/scryptorum/literary_evaluation.py
```

### Run via Scryptorum CLI (Milestone Mode) 

```bash
python -m scryptorum.cli.commands run examples/scryptorum/literary_evaluation.py --milestone
```

## Expected Output

The experiment will:

1. Load pirate and shakespeare agents from existing configurations
2. Create a judge agent for scoring responses
3. Ask each agent all 5 questions
4. Score each response using the judge
5. Calculate comparative metrics
6. Save detailed results to experiment data directory
7. Display final scores and winner

Example output:
```
🏴‍☠️ Starting Literary Agent Evaluation Experiment 📚
Loading test agents...
Creating judge agent...
🏴‍☠️ Evaluating pirate agent...
🎭 Evaluating shakespeare agent...

📊 Experiment Results:
🏴‍☠️ Pirate Agent Average Score: 3.2/10
🎭 Shakespeare Agent Average Score: 7.8/10
📈 Score Difference: 4.6 (positive = Shakespeare wins)
🏆 Winner: Shakespeare
```

## Files Generated

The experiment creates several outputs:

### Experiment Structure

**Note**: This example configures `SCRYPTORUM_EXPERIMENTS_DIR` to save experiments in `examples/scryptorum/experiments/` to keep example data organized.

```
examples/scryptorum/experiments/
└── literary_agent_evaluation/
    ├── experiment.json          # Experiment metadata
    ├── config/                  # Experiment-specific configs
    ├── data/
    │   └── evaluation_results.json  # Detailed results
    └── runs/
        ├── trial_run/          # Trial mode logs (default)
        │   └── run.jsonl       # Complete event/metric logs
        └── run-<timestamp>/    # Milestone mode logs
            └── run.jsonl       # Versioned logs
```

### Log Entries

The experiment generates comprehensive JSONL logs including:

- Agent loading events
- Question/response pairs for each agent
- Judge scoring events  
- Performance timings
- Metric calculations
- Error handling

### Results Data

Detailed JSON results include:
- Complete question/answer pairs
- Individual scores for each question
- Agent-specific performance summaries
- Overall comparison metrics
- Winner determination

## Dependencies

- Existing pirate and shakespeare agent configurations
- Ollama running with gemma3n model
- scryptorum framework
- agents framework

## Customization

You can easily modify this experiment:

- **Add more questions**: Extend `LITERARY_QUESTIONS` list
- **Change scoring criteria**: Modify judge prompt in `judge.toml`
- **Test different agents**: Update agent configurations
- **Add new metrics**: Use `@metric` decorator on new calculation functions
- **Modify evaluation logic**: Extend the evaluation functions

This example provides a solid foundation for building more complex agent evaluation experiments using the scryptorum framework.