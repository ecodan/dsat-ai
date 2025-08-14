"""
Example showing how trial vs milestone runs work transparently.

This exact same code can be run as either a trial or milestone run
via CLI flags - no code changes needed!

Usage examples:
    # Run as trial (default - reuses trial_run directory)
    scryptorum run transparent_experiment --script examples/scryptorum/example_experiment_script.py
    
    # Run as milestone (creates new versioned run directory)
    scryptorum run transparent_experiment --script examples/scryptorum/example_experiment_script.py --milestone
    
    # Or run directly as a Python script
    python examples/scryptorum/example_experiment_script.py
"""

from dsat.scryptorum import experiment, metric, timer, llm_call, set_default_run_type
from dsat.scryptorum.core.decorators import get_current_run
import time


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
    
    # Get current run for manual logging
    run = get_current_run()
    
    # Log experiment start
    run.log_event("experiment_started", {
        "experiment_name": "transparent_experiment",
        "description": "Demonstrates transparent trial/milestone execution"
    })
    
    # The same code runs in both modes
    data = prepare_data()
    run.log_event("data_prepared", {"data_size": len(data)})
    
    results = process_batch(data)
    run.log_event("processing_completed", {"results_count": len(results)})
    
    accuracy = evaluate_results(results)
    throughput = calculate_throughput(results)
    
    # Log final summary
    run.log_event("experiment_completed", {
        "final_accuracy": accuracy,
        "final_throughput": throughput,
        "total_data_points": len(data),
        "total_results": len(results)
    })
    
    print(f"Experiment completed with accuracy: {accuracy}, throughput: {throughput} items/sec")
    return accuracy


@timer("data_preparation")
def prepare_data():
    """Prepare experimental data."""
    run = get_current_run()
    
    # Log preparation start
    run.log_event("data_preparation_started", {"target_size": 50})
    
    time.sleep(0.1)  # Simulate work
    data = [f"data_point_{i}" for i in range(50)]
    
    # Log preparation metrics
    run.log_metric("data_points_created", len(data), "count")
    
    return data


@timer("batch_processing")
def process_batch(data):
    """Process data in batches."""
    run = get_current_run()
    results = []
    batch_size = 10
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    run.log_event("batch_processing_started", {
        "total_items": len(data),
        "batch_size": batch_size,
        "total_batches": total_batches
    })
    
    for i in range(0, len(data), batch_size):  # Process in batches of 10
        batch_num = i // batch_size + 1
        batch = data[i:i+batch_size]
        
        run.log_event("batch_started", {
            "batch_number": batch_num,
            "batch_size": len(batch)
        })
        
        batch_result = process_single_batch(batch)
        results.extend(batch_result)
        
        run.log_event("batch_completed", {
            "batch_number": batch_num,
            "batch_results": len(batch_result),
            "total_results_so_far": len(results)
        })
        
    return results


def process_single_batch(batch):
    """Process a single batch of data."""
    run = get_current_run()
    processed = []
    
    for idx, item in enumerate(batch):
        # Simulate some LLM calls
        response = call_model(f"Process this item: {item}")
        processed.append(response)
        
        # Log progress within batch
        if (idx + 1) % 5 == 0:  # Log every 5th item
            run.log_event("batch_progress", {
                "items_processed_in_batch": idx + 1,
                "batch_size": len(batch)
            })
            
    return processed


@llm_call(model="gpt-4")
def call_model(prompt: str) -> str:
    """Simulate an LLM call."""
    # Additional manual logging for LLM context
    run = get_current_run()
    
    # Log the call attempt
    run.log_event("llm_call_attempted", {
        "model": "gpt-4",
        "prompt_length": len(prompt),
        "prompt_preview": prompt[:50] + "..." if len(prompt) > 50 else prompt
    })
    
    time.sleep(0.02)  # Simulate API latency
    response = f"Processed: {prompt.split()[-1]}"
    
    # Log successful response
    run.log_event("llm_call_succeeded", {
        "response_length": len(response),
        "response_preview": response[:50] + "..." if len(response) > 50 else response
    })
    
    return response


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