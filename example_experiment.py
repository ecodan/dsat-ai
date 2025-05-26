"""
Example showing how trial vs milestone runs work transparently.

This exact same code can be run as either a trial or milestone run
via CLI flags - no code changes needed!

Usage examples:
    # Run as trial (default - reuses trial_run directory)
    python -m scryptorum.cli.commands run example_experiment.py
    
    # Run as milestone (creates new versioned run directory)
    python -m scryptorum.cli.commands run example_experiment.py --milestone
    
    # Set default run type programmatically
    python -c "from scryptorum import set_default_run_type; set_default_run_type('milestone')"
"""

from scryptorum import experiment, metric, timer, llm_call, set_default_run_type


@experiment(name="transparent_experiment")
def main():
    """
    This experiment runs identically regardless of trial vs milestone mode.
    
    Trial mode (default):
    - Creates logs in experiments/transparent_experiment/runs/trial_run/
    - Resets trial_run directory on each execution
    - Only captures logs, no full artifact versioning
    
    Milestone mode (--milestone flag):
    - Creates logs in experiments/transparent_experiment/runs/run-<id>/
    - Creates unique versioned directory for each run
    - Captures full artifacts, code snapshots, etc.
    """
    print("Running transparent experiment...")
    
    # The same code runs in both modes
    data = prepare_data()
    results = process_batch(data)
    accuracy = evaluate_results(results)
    throughput = calculate_throughput(results)
    
    print(f"Experiment completed with accuracy: {accuracy}, throughput: {throughput} items/sec")
    return accuracy


@timer("data_preparation")
def prepare_data():
    """Prepare experimental data."""
    import time
    time.sleep(0.1)  # Simulate work
    return [f"data_point_{i}" for i in range(50)]


@timer("batch_processing")
def process_batch(data):
    """Process data in batches."""
    results = []
    for i in range(0, len(data), 10):  # Process in batches of 10
        batch = data[i:i+10]
        batch_result = process_single_batch(batch)
        results.extend(batch_result)
    return results


def process_single_batch(batch):
    """Process a single batch of data."""
    processed = []
    for item in batch:
        # Simulate some LLM calls
        response = call_model(f"Process this item: {item}")
        processed.append(response)
    return processed


@llm_call(model="gpt-4")
def call_model(prompt: str) -> str:
    """Simulate an LLM call."""
    import time
    time.sleep(0.02)  # Simulate API latency
    return f"Processed: {prompt.split()[-1]}"


@metric(name="accuracy", metric_type="accuracy")
def evaluate_results(results):
    """Calculate final accuracy."""
    # Simulate evaluation
    correct = len([r for r in results if "data_point" in r])
    total = len(results)
    return correct / total if total > 0 else 0.0


@metric(name="throughput", metric_type="rate")
def calculate_throughput(results):
    """Calculate processing throughput."""
    return len(results) / 10.0  # items per second


if __name__ == "__main__":
    # Example: Set default run type programmatically (optional)
    # Uncomment to make all experiments milestone runs by default
    # set_default_run_type("milestone")
    
    # Run the experiment
    # The run type (trial vs milestone) is determined by CLI flags, not code!
    main()