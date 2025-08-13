#!/usr/bin/env python3
"""
Examples of using the agent logging system.

This script demonstrates different ways to configure and use
agent logging in host applications.
"""

import logging
import logging.config
import os
from pathlib import Path

import sys
sys.path.append('../..')
from src.agents import AgentConfig

def example_1_standard_python_logging():
    """
    Example 1: Standard Python logging integration (default).
    
    This is the most host-app friendly approach where logs go through
    the standard Python logging system and the host app controls routing.
    """
    print("=== Example 1: Standard Python Logging ===")
    
    # Host app sets up its own logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('host_app.log')  # File output
        ]
    )
    
    # Create agent config with standard logging
    config = AgentConfig(
        agent_name="chatbot",
        model_provider="anthropic",
        model_family="claude", 
        model_version="claude-3-5-haiku-latest",
        prompt="assistant:v1",
        provider_auth={"api_key": "your-api-key-here"},
        custom_configs={
            "logging": {
                "enabled": True,
                "mode": "standard",  # Default mode
                "level": "standard"  # Include full prompt/response content
            }
        }
    )
    
    # Agent logs will automatically go through host app's logging config
    print("✓ Agent configured for standard logging")
    print("  - Logs go through 'dsat.agents.chatbot' logger")
    print("  - Host app controls where logs are written")
    print("  - LLM call details in log 'extra' data for structured loggers")
    

def example_2_dedicated_jsonl_file():
    """
    Example 2: Dedicated JSONL file for detailed analysis.
    
    This mode writes detailed LLM call logs to a separate JSONL file
    that can be used for analysis, debugging, or compliance.
    """
    print("\n=== Example 2: Dedicated JSONL File ===")
    
    # Ensure log directory exists
    log_dir = Path("./llm_logs")
    log_dir.mkdir(exist_ok=True)
    
    config = AgentConfig(
        agent_name="research_assistant",
        model_provider="anthropic",
        model_family="claude",
        model_version="claude-3-5-haiku-latest", 
        prompt="researcher:v2",
        provider_auth={"api_key": "your-api-key-here"},
        custom_configs={
            "logging": {
                "enabled": True,
                "mode": "jsonl_file",
                "file_path": str(log_dir / "research_assistant_calls.jsonl"),
                "level": "standard"  # Full details
            }
        }
    )
    
    print("✓ Agent configured for JSONL file logging")
    print(f"  - Detailed logs written to: {log_dir / 'research_assistant_calls.jsonl'}")
    print("  - Each line is a complete JSON object with call details")
    print("  - Perfect for analysis, auditing, or debugging")


def example_3_environment_variable_config():
    """
    Example 3: Configuration via environment variables.
    
    This allows runtime configuration without changing code,
    useful for different deployment environments.
    """
    print("\n=== Example 3: Environment Variable Configuration ===")
    
    # Set environment variables (would typically be done in deployment)
    os.environ["DSAT_AGENT_LOGGING_ENABLED"] = "true"
    os.environ["DSAT_AGENT_LOGGING_MODE"] = "jsonl_file" 
    os.environ["DSAT_AGENT_LOGGING_FILE_PATH"] = "./prod_llm_calls.jsonl"
    os.environ["DSAT_AGENT_LOGGING_LEVEL"] = "minimal"  # For production
    
    # Agent config doesn't need logging section - uses env vars
    config = AgentConfig(
        agent_name="production_agent",
        model_provider="anthropic",
        model_family="claude",
        model_version="claude-3-5-haiku-latest",
        prompt="production:v1",
        provider_auth={"api_key": "your-api-key-here"}
        # No custom_configs.logging - uses environment variables
    )
    
    print("✓ Agent configured via environment variables")
    print("  - DSAT_AGENT_LOGGING_ENABLED=true")
    print("  - DSAT_AGENT_LOGGING_MODE=jsonl_file")
    print("  - DSAT_AGENT_LOGGING_FILE_PATH=./prod_llm_calls.jsonl")
    print("  - DSAT_AGENT_LOGGING_LEVEL=minimal")
    
    # Clean up for this example
    del os.environ["DSAT_AGENT_LOGGING_ENABLED"]
    del os.environ["DSAT_AGENT_LOGGING_MODE"]
    del os.environ["DSAT_AGENT_LOGGING_FILE_PATH"]
    del os.environ["DSAT_AGENT_LOGGING_LEVEL"]


def example_4_custom_callback():
    """
    Example 4: Custom callback for advanced logging.
    
    This allows host apps to implement custom logging logic,
    such as writing to databases, streaming to monitoring systems, etc.
    """
    print("\n=== Example 4: Custom Callback Logging ===")
    
    def custom_llm_logger(call_data):
        """Custom function to handle LLM call logging."""
        # Example: write to database, send to monitoring system, etc.
        print(f"📝 Custom logger: {call_data['agent_name']} call took {call_data['duration_ms']}ms")
        
        # Could also write to database:
        # db.insert_llm_call(call_data)
        
        # Or send to monitoring:
        # metrics.record_llm_call(call_data)
    
    config = AgentConfig(
        agent_name="custom_logged_agent",
        model_provider="anthropic",
        model_family="claude",
        model_version="claude-3-5-haiku-latest",
        prompt="assistant:v1",
        provider_auth={"api_key": "your-api-key-here"},
        custom_configs={
            "logging": {
                "enabled": True,
                "mode": "callback",
                "callback": custom_llm_logger,
                "level": "standard"
            }
        }
    )
    
    print("✓ Agent configured with custom callback logging")
    print("  - Each LLM call triggers custom_llm_logger function")
    print("  - Host app has complete control over log processing")
    print("  - Can integrate with databases, monitoring, etc.")


def example_5_disabled_logging():
    """
    Example 5: Completely disabled logging.
    
    For scenarios where logging is not needed or not allowed.
    """
    print("\n=== Example 5: Disabled Logging ===")
    
    config = AgentConfig(
        agent_name="no_logging_agent",
        model_provider="anthropic",
        model_family="claude",
        model_version="claude-3-5-haiku-latest",
        prompt="assistant:v1",
        provider_auth={"api_key": "your-api-key-here"},
        custom_configs={
            "logging": {
                "enabled": False
            }
        }
    )
    
    # Or simply omit the logging config entirely (disabled by default)
    config_default = AgentConfig(
        agent_name="default_agent",
        model_provider="anthropic",
        model_family="claude",
        model_version="claude-3-5-haiku-latest",
        prompt="assistant:v1",
        provider_auth={"api_key": "your-api-key-here"}
        # No custom_configs.logging = disabled by default
    )
    
    print("✓ Agents configured with disabled logging")
    print("  - Zero logging overhead")
    print("  - No files created")
    print("  - Suitable for privacy-sensitive applications")


def example_6_host_app_integration():
    """
    Example 6: Full integration with host app logging.
    
    Shows how a host app might integrate agent logging with its
    existing logging infrastructure.
    """
    print("\n=== Example 6: Full Host App Integration ===")
    
    # Host app's sophisticated logging setup
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
            },
            'simple': {
                'format': '%(name)s: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'level': 'INFO'
            },
            'file': {
                'class': 'logging.FileHandler', 
                'filename': 'host_app.log',
                'formatter': 'detailed',
                'level': 'DEBUG'
            },
            'agents': {
                'class': 'logging.FileHandler',
                'filename': 'agent_activity.log',
                'formatter': 'detailed',
                'level': 'INFO'
            }
        },
        'loggers': {
            'myapp': {
                'handlers': ['console', 'file'],
                'level': 'INFO'
            },
            'dsat.agents': {
                'handlers': ['agents', 'console'],
                'level': 'INFO',
                'propagate': False  # Don't send to root logger
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': ['console']
        }
    })
    
    # Now agent logs automatically go to the right places
    config = AgentConfig(
        agent_name="integrated_agent",
        model_provider="anthropic",
        model_family="claude",
        model_version="claude-3-5-haiku-latest",
        prompt="assistant:v1", 
        provider_auth={"api_key": "your-api-key-here"},
        custom_configs={
            "logging": {
                "enabled": True,
                "mode": "standard"  # Works with host app's logging config
            }
        }
    )
    
    print("✓ Agent fully integrated with host app logging")
    print("  - Agent logs go to 'agent_activity.log' and console")
    print("  - Host app controls all routing and formatting")
    print("  - Clean separation from other app logs")


if __name__ == "__main__":
    print("Agent Logging Configuration Examples")
    print("=" * 50)
    
    example_1_standard_python_logging()
    example_2_dedicated_jsonl_file()
    example_3_environment_variable_config()
    example_4_custom_callback()
    example_5_disabled_logging()
    example_6_host_app_integration()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo use these examples:")
    print("1. Replace 'your-api-key-here' with actual API keys")
    print("2. Adjust file paths for your application")
    print("3. Customize logging levels and formats as needed")