#!/usr/bin/env python3
"""
Demo: Pirate vs Shakespeare Conversation

This demonstrates the DSAT agent system with two OllamaAgents having a conversation:
- A pirate who hates literature 
- Shakespeare who tries to convince the pirate of literature's value

Features demonstrated:
- Agent configuration from JSON files
- Prompt loading from TOML files
- Agent logging
- Multi-turn conversation loop
"""

import logging
import sys
import time
from pathlib import Path

# Add the src directory to the path so we can import dsat modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dsat.agents.agent import Agent, AgentConfig


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_agents(config_dir: Path, prompts_dir: Path):
    """Load the pirate and shakespeare agents from configuration."""
    
    # Load agent configurations
    agent_configs = AgentConfig.load_from_file(config_dir / "agents.json")
    
    # Create agents using the factory method
    pirate_agent = Agent.create(
        config=agent_configs["pirate"], 
        prompts_dir=prompts_dir
    )
    
    shakespeare_agent = Agent.create(
        config=agent_configs["shakespeare"], 
        prompts_dir=prompts_dir
    )
    
    return pirate_agent, shakespeare_agent


def print_conversation_header():
    """Print a nice header for the conversation."""
    print("=" * 80)
    print("🏴‍☠️  PIRATE vs SHAKESPEARE: A LITERARY DEBATE  📚")
    print("=" * 80)
    print("The gruff pirate captain faces off against the eloquent bard")
    print("in a battle of wits about the value of literature!")
    print("=" * 80)
    print()


def print_agent_response(agent_name: str, response: str, emoji: str):
    """Print an agent's response with nice formatting."""
    print(f"{emoji} {agent_name.upper()}:")
    print("-" * 40)
    print(response)
    print()


def main():
    """Main conversation loop."""
    
    # Setup
    logger = setup_logging()
    
    # Get paths relative to this script
    script_dir = Path(__file__).parent
    config_dir = script_dir.parent / "config"
    prompts_dir = config_dir / "prompts"
    
    logger.info(f"Loading agents from {config_dir}")
    logger.info(f"Loading prompts from {prompts_dir}")
    
    try:
        # Load agents
        pirate_agent, shakespeare_agent = load_agents(config_dir, prompts_dir)
        logger.info("Successfully loaded pirate and shakespeare agents")
        
        # Print header
        print_conversation_header()
        
        # Initial topic - Shakespeare starts
        current_topic = "Good morrow! I come to speak of literature's might, how books can fill a soul with pure delight!"
        
        # Conversation loop - 5 exchanges
        for round_num in range(1, 6):
            print(f"🔄 ROUND {round_num}")
            print("=" * 40)
            
            # Shakespeare speaks first (except round 1 where he sets the topic)
            if round_num == 1:
                print_agent_response("Shakespeare", current_topic, "📜")
                
                # Pirate responds
                logger.info(f"Round {round_num}: Pirate responding to Shakespeare")
                pirate_response = pirate_agent.invoke(current_topic)
                print_agent_response("Pirate", pirate_response, "🏴‍☠️")
                current_topic = pirate_response
                
            else:
                # Shakespeare responds to pirate
                logger.info(f"Round {round_num}: Shakespeare responding to Pirate")
                shakespeare_response = shakespeare_agent.invoke(current_topic)
                print_agent_response("Shakespeare", shakespeare_response, "📜")
                
                # Pirate responds back
                logger.info(f"Round {round_num}: Pirate responding to Shakespeare")
                pirate_response = pirate_agent.invoke(shakespeare_response)
                print_agent_response("Pirate", pirate_response, "🏴‍☠️")
                current_topic = pirate_response
            
            # Small delay between rounds for readability
            if round_num < 5:
                time.sleep(1)
        
        print("=" * 80)
        print("🎭 THE DEBATE CONCLUDES! 🎭")
        print("Thank you for witnessing this battle of words!")
        print("=" * 80)
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print("\n❌ Error: Missing required dependency")
        print("Make sure you have requests installed: pip install requests")
        print("Also ensure Ollama is running with gemma3n model available")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error during conversation: {e}")
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Ensure gemma3n model is available: ollama pull gemma3n")
        print("3. Check that Ollama is accessible at http://localhost:11434")
        sys.exit(1)


if __name__ == "__main__":
    main()