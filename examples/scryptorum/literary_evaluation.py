"""
Literary Agent Evaluation Experiment

This experiment evaluates how well different character agents can answer questions
about literature and reading comprehension. It compares a literature-hating pirate
against a Shakespeare-inspired poet using a standardized question set.

The experiment demonstrates:
- Loading multiple agents from configuration
- Creating a labeled dataset with golden answers
- Using a judge agent to score responses
- Comprehensive logging and metrics tracking
- Automatic experiment management via decorators

Usage:
    python examples/scryptorum/literary_evaluation.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Set the experiments directory to be inside examples/scryptorum/
examples_experiments_dir = project_root / "examples" / "scryptorum" / "experiments"
os.environ["SCRYPTORUM_EXPERIMENTS_DIR"] = str(examples_experiments_dir)

from dsat.scryptorum import experiment, metric, timer, llm_call
from dsat.scryptorum.core.decorators import get_current_run
from dsat.agents.agent import Agent


# Dataset of literary questions with golden answers
LITERARY_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the main theme of Shakespeare's Romeo and Juliet?",
        "golden_answer": "The destructive power of feuding and prejudice, and the transformative power of love that transcends social barriers.",
        "category": "theme_analysis",
        "difficulty": "medium"
    },
    {
        "id": 2, 
        "question": "Why is reading literature important for personal development?",
        "golden_answer": "Literature develops empathy, critical thinking, cultural understanding, and emotional intelligence while expanding vocabulary and imagination.",
        "category": "literature_value",
        "difficulty": "easy"
    },
    {
        "id": 3,
        "question": "How does metaphor enhance storytelling?",
        "golden_answer": "Metaphors create vivid imagery, convey complex emotions, make abstract concepts concrete, and add layers of meaning that engage readers intellectually and emotionally.",
        "category": "literary_devices", 
        "difficulty": "hard"
    },
    {
        "id": 4,
        "question": "What makes a character memorable in literature?",
        "golden_answer": "Memorable characters have clear motivations, face meaningful conflicts, show growth or change, have distinctive voices, and represent universal human experiences.",
        "category": "character_analysis",
        "difficulty": "medium"
    },
    {
        "id": 5,
        "question": "How do books differ from movies in storytelling?",
        "golden_answer": "Books allow deeper character psychology, internal monologue, detailed world-building, reader imagination, and personal pacing, while movies excel at visual spectacle and shared experience.",
        "category": "medium_comparison",
        "difficulty": "easy"
    }
]


def load_test_agents():
    """Load the pirate and shakespeare agents for testing."""
    run = get_current_run()
    
    # Use existing agent configurations from demo
    config_dir = project_root / "examples" / "config"
    prompts_dir = config_dir / "prompts"
    
    run.log_event("agents_loading_started", {
        "config_dir": str(config_dir),
        "prompts_dir": str(prompts_dir)
    })
    
    try:
        pirate_agent = Agent.create_from_config(
            config_file=config_dir / "agents.json",
            agent_name="pirate",
            prompts_dir=prompts_dir
        )
        
        shakespeare_agent = Agent.create_from_config(
            config_file=config_dir / "agents.json", 
            agent_name="shakespeare",
            prompts_dir=prompts_dir
        )
        
        run.log_event("agents_loaded_successfully", {
            "pirate_model": pirate_agent.config.model_version,
            "shakespeare_model": shakespeare_agent.config.model_version
        })
        
        return pirate_agent, shakespeare_agent
        
    except Exception as e:
        run.log_event("agents_loading_failed", {"error": str(e)})
        raise


def create_judge_agent():
    """Create a judge agent for scoring responses."""
    run = get_current_run()
    
    # Create a simple judge configuration
    judge_config = {
        "agent_name": "judge",
        "model_provider": "ollama",
        "model_family": "gemma",
        "model_version": "gemma3n",
        "prompt": "judge:v1",
        "custom_configs": {
            "logging": {
                "enabled": True,
                "mode": "standard"
            }
        }
    }

    try:
        # Use the same prompts directory as other agents
        prompts_dir = project_root / "examples" / "config" / "prompts"
        judge_agent = Agent.from_dict(judge_config, prompts_dir=prompts_dir)
        system_prompt = judge_agent.get_system_prompt()
        
        run.log_event("judge_agent_created", {
            "model": judge_config["model_version"],
            "prompt_length": len(system_prompt) if system_prompt else 0
        })
        
        return judge_agent
        
    except Exception as e:
        run.log_event("judge_creation_failed", {"error": str(e)})
        raise


@timer("agent_response")
def get_agent_response(agent: Agent, question: str, question_id: int) -> str:
    """Get response from an agent for a given question."""
    run = get_current_run()
    
    run.log_event("agent_response_started", {
        "agent_type": agent.config.model_version,
        "question_id": question_id,
        "question_preview": question[:50] + "..." if len(question) > 50 else question
    })
    
    try:
        response = agent.invoke(question)
        
        run.log_event("agent_response_completed", {
            "agent_type": agent.config.model_version,
            "question_id": question_id,
            "response_length": len(response),
            "response_preview": response[:100] + "..." if len(response) > 100 else response
        })
        
        return response
        
    except Exception as e:
        run.log_event("agent_response_failed", {
            "agent_type": agent.config.model_version,
            "question_id": question_id,
            "error": str(e)
        })
        raise


@timer("judge_scoring")
def score_response(judge_agent: Agent, question: str, response: str, golden_answer: str) -> dict:
    """Score a response using the judge agent."""
    run = get_current_run()
    
    judge_prompt = f"""
Question: {question}

Golden Answer: {golden_answer}

Response to Evaluate: {response}

Please score this response according to the criteria provided in your system prompt.
"""
    
    run.log_event("judge_scoring_started", {
        "question_length": len(question),
        "response_length": len(response),
        "golden_answer_length": len(golden_answer)
    })
    
    try:
        judge_response = judge_agent.invoke(judge_prompt)
        
        # Try to parse JSON from judge response
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{[^}]+\}', judge_response)
            if json_match:
                score_data = json.loads(json_match.group())
            else:
                # Fallback scoring if JSON parsing fails
                score_data = {
                    "score": 5,
                    "accuracy": 5,
                    "depth": 5,
                    "clarity": 5,
                    "relevance": 5,
                    "reasoning": "Failed to parse judge response"
                }
                
        except json.JSONDecodeError:
            # Fallback scoring
            score_data = {
                "score": 5,
                "accuracy": 5,
                "depth": 5,
                "clarity": 5,
                "relevance": 5,
                "reasoning": "Judge response was not valid JSON"
            }
        
        run.log_event("judge_scoring_completed", {
            "overall_score": score_data.get("score", 0),
            "judge_response_length": len(judge_response)
        })
        
        return score_data
        
    except Exception as e:
        run.log_event("judge_scoring_failed", {"error": str(e)})
        # Return default scores on error
        return {
            "score": 1,
            "accuracy": 1,
            "depth": 1,
            "clarity": 1,
            "relevance": 1,
            "reasoning": f"Scoring failed: {str(e)}"
        }


@timer("full_evaluation")
def evaluate_agent_on_dataset(agent: Agent, judge_agent: Agent, agent_name: str) -> dict:
    """Evaluate a single agent on the full question dataset."""
    run = get_current_run()
    
    run.log_event("agent_evaluation_started", {
        "agent_name": agent_name,
        "agent_model": agent.config.model_version,
        "total_questions": len(LITERARY_QUESTIONS)
    })
    
    results = []
    total_score = 0
    
    for question_data in LITERARY_QUESTIONS:
        question_id = question_data["id"]
        question = question_data["question"]
        golden_answer = question_data["golden_answer"]
        category = question_data["category"]
        difficulty = question_data["difficulty"]
        
        run.log_event("question_started", {
            "agent_name": agent_name,
            "question_id": question_id,
            "category": category,
            "difficulty": difficulty
        })
        
        # Get agent response
        response = get_agent_response(agent, question, question_id)
        
        # Score the response
        score_data = score_response(judge_agent, question, response, golden_answer)
        
        # Compile result
        result = {
            "question_id": question_id,
            "question": question,
            "category": category,
            "difficulty": difficulty,
            "response": response,
            "golden_answer": golden_answer,
            "scores": score_data,
            "agent_name": agent_name
        }
        
        results.append(result)
        total_score += score_data.get("score", 0)
        
        run.log_event("question_completed", {
            "agent_name": agent_name,
            "question_id": question_id,
            "score": score_data.get("score", 0),
            "running_total": total_score
        })
    
    average_score = total_score / len(LITERARY_QUESTIONS)
    
    evaluation_summary = {
        "agent_name": agent_name,
        "agent_model": agent.config.model_version,
        "total_questions": len(LITERARY_QUESTIONS),
        "total_score": total_score,
        "average_score": average_score,
        "results": results
    }
    
    run.log_event("agent_evaluation_completed", {
        "agent_name": agent_name,
        "total_score": total_score,
        "average_score": average_score
    })
    
    return evaluation_summary


@metric(name="pirate_average_score", metric_type="accuracy")
def calculate_pirate_score(pirate_results: dict) -> float:
    """Calculate average score for pirate agent."""
    return pirate_results["average_score"]


@metric(name="shakespeare_average_score", metric_type="accuracy") 
def calculate_shakespeare_score(shakespeare_results: dict) -> float:
    """Calculate average score for shakespeare agent."""
    return shakespeare_results["average_score"]


@metric(name="score_difference", metric_type="difference")
def calculate_score_difference(pirate_results: dict, shakespeare_results: dict) -> float:
    """Calculate the difference in scores between agents."""
    return shakespeare_results["average_score"] - pirate_results["average_score"]


@experiment(name="literary_agent_evaluation")
def main():
    """
    Main experiment function that evaluates pirate vs shakespeare agents
    on literature comprehension questions.
    """
    run = get_current_run()
    
    print("🏴‍☠️ Starting Literary Agent Evaluation Experiment 📚")
    print(f"📁 Experiments will be saved to: {examples_experiments_dir}")
    
    run.log_event("experiment_started", {
        "experiment_name": "literary_agent_evaluation",
        "description": "Comparing pirate and shakespeare agents on literature questions",
        "dataset_size": len(LITERARY_QUESTIONS),
        "evaluation_method": "judge_agent_scoring"
    })
    
    # Load agents
    print("Loading test agents...")
    pirate_agent, shakespeare_agent = load_test_agents()
    
    # Create judge
    print("Creating judge agent...")
    judge_agent = create_judge_agent()
    
    # Evaluate pirate agent
    print("🏴‍☠️ Evaluating pirate agent...")
    pirate_results = evaluate_agent_on_dataset(pirate_agent, judge_agent, "pirate")
    pirate_score = calculate_pirate_score(pirate_results)
    
    # Evaluate shakespeare agent  
    print("🎭 Evaluating shakespeare agent...")
    shakespeare_results = evaluate_agent_on_dataset(shakespeare_agent, judge_agent, "shakespeare")
    shakespeare_score = calculate_shakespeare_score(shakespeare_results)
    
    # Calculate comparison metrics
    score_diff = calculate_score_difference(pirate_results, shakespeare_results)
    
    # Log final experiment results
    run.log_event("experiment_completed", {
        "pirate_average_score": pirate_score,
        "shakespeare_average_score": shakespeare_score,
        "score_difference": score_diff,
        "winner": "shakespeare" if score_diff > 0 else "pirate" if score_diff < 0 else "tie",
        "total_questions_asked": len(LITERARY_QUESTIONS) * 2  # Both agents answer all questions
    })
    
    # Save detailed results to data directory
    experiment_data = {
        "experiment_name": "literary_agent_evaluation",
        "pirate_results": pirate_results,
        "shakespeare_results": shakespeare_results,
        "summary": {
            "pirate_score": pirate_score,
            "shakespeare_score": shakespeare_score,
            "score_difference": score_diff,
            "winner": "shakespeare" if score_diff > 0 else "pirate" if score_diff < 0 else "tie"
        }
    }
    
    # Write results to experiment data directory
    if run.experiment_path:
        results_file = run.experiment_path / "data" / "evaluation_results.json"
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(experiment_data, f, indent=2)
        
        run.log_event("results_saved", {"file_path": str(results_file)})
    
    # Print summary
    print(f"\n📊 Experiment Results:")
    print(f"🏴‍☠️ Pirate Agent Average Score: {pirate_score:.2f}/10")
    print(f"🎭 Shakespeare Agent Average Score: {shakespeare_score:.2f}/10")
    print(f"📈 Score Difference: {score_diff:.2f} (positive = Shakespeare wins)")
    
    winner = "Shakespeare" if score_diff > 0 else "Pirate" if score_diff < 0 else "Tie"
    print(f"🏆 Winner: {winner}")
    
    return {
        "pirate_score": pirate_score,
        "shakespeare_score": shakespeare_score,
        "winner": winner
    }



if __name__ == "__main__":
    main()