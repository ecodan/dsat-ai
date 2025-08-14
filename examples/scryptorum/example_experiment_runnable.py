"""
Example showing the class-based runnable approach for scryptorum experiments.

This demonstrates how to create experiments using BaseRunnable instead of decorators.
The runnable approach provides more structure with prepare/run/score/cleanup phases.

Usage:
    # Run as trial (default - reuses trial_run directory)
    scryptorum run sentiment_analysis --module examples.scryptorum.example_experiment_runnable.SentimentAnalysisRunnable

    # Run as milestone (creates new versioned run directory)
    scryptorum run sentiment_analysis --module examples.scryptorum.example_experiment_runnable.SentimentAnalysisRunnable --milestone

    # Or use script approach
    scryptorum run sentiment_analysis --script examples/scryptorum/example_experiment_runnable.py
"""

import time
from typing import Dict, Any, List

from dsat.scryptorum.execution.runner import BaseRunnable
from dsat.scryptorum.core.experiment import Experiment
from dsat.scryptorum.core.runs import Run


class SentimentAnalysisRunnable(BaseRunnable):
    """Example sentiment analysis experiment using the runnable approach."""
    
    def __init__(self, experiment: Experiment, run: Run, config: Dict[str, Any]):
        super().__init__(experiment, run, config)
        self.data = []
        self.results = []
        self.accuracy = 0.0
        self.throughput = 0.0
    
    def prepare(self) -> None:
        """Prepare experimental data and setup."""
        self.run.logger.debug("Preparing sentiment analysis experiment...")
        
        # Log preparation start
        self.run.log_event("preparation_started", {
            "experiment": self.experiment.experiment_name,
            "config": self.config
        })
        
        # Prepare data with timing
        start_time = time.time()
        self.data = self._prepare_data()
        prep_duration = time.time() - start_time
        
        # Log preparation metrics
        self.run.log_timing("data_preparation", prep_duration)
        self.run.log_metric("data_size", len(self.data), "count")
        
        self.run.logger.debug(f"Prepared {len(self.data)} data points in {prep_duration:.3f}s")
    
    def execute(self) -> None:
        """Main experiment execution."""
        self.run.logger.debug("Running sentiment analysis...")
        
        # Log run start
        self.run.log_event("main_execution_started", {
            "data_size": len(self.data)
        })
        
        # Process data in batches with timing
        start_time = time.time()
        self.results = self._process_batch(self.data)
        processing_duration = time.time() - start_time
        
        # Log processing metrics
        self.run.log_timing("batch_processing", processing_duration)
        self.run.log_metric("results_count", len(self.results), "count")
        
        self.run.logger.debug(f"Processed {len(self.results)} results in {processing_duration:.3f}s")
    
    def score(self) -> None:
        """Evaluate results and calculate metrics."""
        self.run.logger.debug("Evaluating results...")
        
        # Calculate accuracy
        self.accuracy = self._evaluate_results(self.results)
        self.run.log_metric("accuracy", self.accuracy, "accuracy")
        
        # Calculate throughput
        self.throughput = self._calculate_throughput(self.results)
        self.run.log_metric("throughput", self.throughput, "rate")
        
        # Log final scores
        self.run.log_event("scoring_completed", {
            "accuracy": self.accuracy,
            "throughput": self.throughput
        })
        
        self.run.logger.debug(f"Final accuracy: {self.accuracy:.3f}, throughput: {self.throughput:.1f} items/sec")
    
    def cleanup(self) -> None:
        """Clean up resources and log completion."""
        self.run.logger.debug("Cleaning up experiment...")
        
        # Log final summary
        self.run.log_event("experiment_completed", {
            "total_data_points": len(self.data),
            "total_results": len(self.results),
            "final_accuracy": self.accuracy,
            "final_throughput": self.throughput
        })
        
        # Clear data to free memory
        self.data.clear()
        self.results.clear()
    
    def _prepare_data(self) -> List[str]:
        """Prepare experimental data."""
        time.sleep(0.1)  # Simulate data loading/preparation
        return [f"sentiment_data_point_{i}" for i in range(50)]
    
    def _process_batch(self, data: List[str]) -> List[str]:
        """Process data in batches."""
        results = []
        batch_size = 10
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_result = self._process_single_batch(batch)
            results.extend(batch_result)
            
            # Log batch completion
            self.run.log_event("batch_completed", {
                "batch_number": i // batch_size + 1,
                "batch_size": len(batch),
                "results_so_far": len(results)
            })
        
        return results
    
    def _process_single_batch(self, batch: List[str]) -> List[str]:
        """Process a single batch of data."""
        processed = []
        
        for item in batch:
            # Simulate LLM call with logging
            response = self._call_model(f"Analyze sentiment: {item}")
            processed.append(response)
        
        return processed
    
    def _call_model(self, prompt: str) -> str:
        """Simulate an LLM call with logging."""
        start_time = time.time()
        
        # Simulate API latency
        time.sleep(0.02)
        response = f"Positive sentiment detected in: {prompt.split()[-1]}"
        
        # Log the LLM call
        duration = time.time() - start_time
        self.run.log_llm_call(
            model="gpt-4",
            input_data=prompt,
            output_data=response,
            duration_ms=duration * 1000
        )
        
        return response
    
    def _evaluate_results(self, results: List[str]) -> float:
        """Calculate final accuracy."""
        # Simulate evaluation logic
        correct = len([r for r in results if "sentiment_data_point" in r])
        total = len(results)
        return correct / total if total > 0 else 0.0
    
    def _calculate_throughput(self, results: List[str]) -> float:
        """Calculate processing throughput."""
        return len(results) / 10.0  # items per second


# For script-based execution, create a simple wrapper
def main():
    """Simple wrapper for script-based execution."""
    from dsat.scryptorum.core.experiment import Experiment
    from dsat.scryptorum.core.runs import RunType
    
    # This would normally be handled by the CLI, but for direct script execution:
    experiment = Experiment(".", "sentiment_analysis")
    run = experiment.create_run(RunType.TRIAL)
    
    try:
        runnable = SentimentAnalysisRunnable(experiment, run, {})
        runnable.prepare()
        runnable.execute()
        runnable.score()
        runnable.cleanup()
    finally:
        run.finish()


if __name__ == "__main__":
    main()