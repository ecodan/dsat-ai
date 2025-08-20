"""
Interactive chat CLI for DSAT agents.

Provides a terminal-based chat interface for testing prompts and having conversations
with different LLM providers through the DSAT agent system.
"""

import os
import sys
import json
import argparse
import logging
import requests
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add color support for terminal output
try:
    from colorama import init, Fore, Back, Style

    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback without colors
    class MockColorama:
        class Fore:
            RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""

        class Style:
            BRIGHT = DIM = NORMAL = RESET_ALL = ""

    Fore = MockColorama.Fore
    Style = MockColorama.Style
    COLORS_AVAILABLE = False

from ..agents.agent import Agent, AgentConfig


class ChatSession:
    """Manages a chat session with conversation history and state."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.history = []
        self.start_time = datetime.now()

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.history.append(
            {"timestamp": datetime.now().isoformat(), "role": role, "content": content}
        )

    def export_conversation(self, file_path: Path):
        """Export conversation history to a JSON file."""
        export_data = {
            "session_start": self.start_time.isoformat(),
            "agent_config": self.agent.config.to_dict(),
            "conversation": self.history,
        }

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)


class ChatInterface:
    """Interactive chat interface for DSAT agents."""

    def __init__(self):
        self.current_session: Optional[ChatSession] = None
        self.available_agents: Dict[str, AgentConfig] = {}
        self.logger = self._setup_logging()
        self.prompts_dir: Optional[Path] = None  # Will be set during initialization
        self.ollama_models_available: List[str] = []  # Store available Ollama models
        self.streaming_enabled: bool = False  # Track streaming state

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the chat interface."""
        logging.basicConfig(
            level=logging.WARNING,  # Keep quiet during chat
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)

    def _print_banner(self):
        """Print the chat interface banner."""
        print(f"{Fore.CYAN}{Style.BRIGHT}")
        print("=" * 60)
        print("🤖 DSAT Chat Interface")
        print("Interactive testing for LLM agents and prompts")
        print("=" * 60)
        print(f"{Style.RESET_ALL}")

    def _print_help(self):
        """Print available chat commands."""
        print(f"\n{Fore.YELLOW}Available Commands:{Style.RESET_ALL}")
        print(
            f"  {Fore.GREEN}/help{Style.RESET_ALL}                 - Show this help message"
        )
        print(
            f"  {Fore.GREEN}/agents{Style.RESET_ALL}               - List available agents"
        )
        print(
            f"  {Fore.GREEN}/providers{Style.RESET_ALL}            - List available LLM providers"
        )
        print(
            f"  {Fore.GREEN}/switch <agent>{Style.RESET_ALL}       - Switch to a different agent"
        )
        print(
            f"  {Fore.GREEN}/stream{Style.RESET_ALL}               - Toggle streaming mode (currently: {'ON' if self.streaming_enabled else 'OFF'})"
        )
        print(
            f"  {Fore.GREEN}/history{Style.RESET_ALL}              - Show conversation history"
        )
        print(
            f"  {Fore.GREEN}/clear{Style.RESET_ALL}                - Clear conversation history"
        )
        print(
            f"  {Fore.GREEN}/export <file>{Style.RESET_ALL}        - Export conversation to file"
        )
        print(
            f"  {Fore.GREEN}/quit{Style.RESET_ALL} or {Fore.GREEN}/exit{Style.RESET_ALL}        - Exit chat"
        )
        print()

    def _print_agents(self):
        """Print available agents."""
        if not self.available_agents:
            print(f"{Fore.YELLOW}No agents configured.{Style.RESET_ALL}")
            print(
                f"\nAvailable providers: {', '.join(Agent.get_available_providers().keys())}"
            )

            # Show Ollama models if available
            if self.ollama_models_available:
                print(f"\n{Fore.CYAN}Ollama models available:{Style.RESET_ALL}")
                for model in self.ollama_models_available:
                    family = self._infer_model_family(model)
                    print(f"  {Fore.GREEN}{model}{Style.RESET_ALL} ({family})")
                print(
                    f"\nUse: {Fore.GREEN}dsat chat --provider ollama{Style.RESET_ALL} to select a model"
                )
            return

        print(f"\n{Fore.YELLOW}Available Agents:{Style.RESET_ALL}")
        for name, config in self.available_agents.items():
            current_marker = (
                " (current)"
                if (
                    self.current_session
                    and self.current_session.agent.config.agent_name == name
                )
                else ""
            )
            print(
                f"  {Fore.GREEN}{name}{Style.RESET_ALL} - {config.model_provider}/{config.model_version}{current_marker}"
            )
        print()

    def _print_providers(self):
        """Print available LLM providers."""
        providers = Agent.get_available_providers()

        print(f"\n{Fore.YELLOW}Available LLM Providers:{Style.RESET_ALL}")

        # Group by source type
        built_in = [
            (name, details)
            for name, details in providers.items()
            if details == "built-in"
        ]
        registered = [
            (name, details)
            for name, details in providers.items()
            if details == "registered"
        ]
        plugins = [
            (name, details)
            for name, details in providers.items()
            if details == "plugin"
        ]

        if built_in:
            print(f"\n  {Fore.CYAN}Built-in Providers:{Style.RESET_ALL}")
            for name, _ in built_in:
                print(f"    {Fore.GREEN}{name}{Style.RESET_ALL}")

        if registered:
            print(f"\n  {Fore.CYAN}Registered Providers:{Style.RESET_ALL}")
            for name, _ in registered:
                print(f"    {Fore.GREEN}{name}{Style.RESET_ALL}")

        if plugins:
            print(f"\n  {Fore.CYAN}Plugin Providers:{Style.RESET_ALL}")
            for name, _ in plugins:
                print(f"    {Fore.GREEN}{name}{Style.RESET_ALL}")

        if not providers:
            print(f"  {Fore.RED}No providers available{Style.RESET_ALL}")

        print()

    def _handle_command(self, command: str) -> bool:
        """
        Handle special chat commands.

        :param command: Command string starting with /
        :return: True if should continue chat, False if should exit
        """
        parts = command[1:].split()
        cmd = parts[0].lower()

        if cmd in ["quit", "exit"]:
            return False
        elif cmd == "help":
            self._print_help()
        elif cmd == "agents":
            self._print_agents()
        elif cmd == "providers":
            self._print_providers()
        elif cmd == "switch":
            if len(parts) < 2:
                print(f"{Fore.RED}Usage: /switch <agent_name>{Style.RESET_ALL}")
            else:
                self._switch_agent(parts[1])
        elif cmd == "stream":
            self._toggle_streaming()
        elif cmd == "history":
            self._show_history()
        elif cmd == "clear":
            self._clear_history()
        elif cmd == "export":
            if len(parts) < 2:
                print(f"{Fore.RED}Usage: /export <filename>{Style.RESET_ALL}")
            else:
                self._export_conversation(parts[1])
        else:
            print(f"{Fore.RED}Unknown command: {command}{Style.RESET_ALL}")
            print(f"Type {Fore.GREEN}/help{Style.RESET_ALL} for available commands.")

        return True

    def _switch_agent(self, agent_name: str):
        """Switch to a different agent."""
        if agent_name not in self.available_agents:
            print(f"{Fore.RED}Agent '{agent_name}' not found.{Style.RESET_ALL}")
            print(f"Use {Fore.GREEN}/agents{Style.RESET_ALL} to see available agents.")
            return

        try:
            config = self.available_agents[agent_name]
            resolved_prompts_dir = self._resolve_prompts_directory(
                self.cli_prompts_dir, config, self.config_file
            )
            agent = Agent.create(
                config, logger=self.logger, prompts_dir=resolved_prompts_dir
            )
            self.current_session = ChatSession(agent)
            print(f"{Fore.GREEN}Switched to agent: {agent_name}{Style.RESET_ALL}")
        except Exception as e:
            print(
                f"{Fore.RED}Error switching to agent '{agent_name}': {e}{Style.RESET_ALL}"
            )

    def _toggle_streaming(self):
        """Toggle streaming mode on/off."""
        self.streaming_enabled = not self.streaming_enabled
        status = "enabled" if self.streaming_enabled else "disabled"
        print(f"{Fore.GREEN}Streaming mode {status}.{Style.RESET_ALL}")

    def _show_history(self):
        """Show conversation history."""
        if not self.current_session or not self.current_session.history:
            print(f"{Fore.YELLOW}No conversation history.{Style.RESET_ALL}")
            return

        print(f"\n{Fore.YELLOW}Conversation History:{Style.RESET_ALL}")
        for msg in self.current_session.history:
            role_color = Fore.BLUE if msg["role"] == "user" else Fore.MAGENTA
            print(
                f"{role_color}{msg['role'].upper()}:{Style.RESET_ALL} {msg['content'][:100]}..."
            )
        print()

    def _clear_history(self):
        """Clear conversation history."""
        if self.current_session:
            self.current_session.history.clear()
            print(f"{Fore.GREEN}Conversation history cleared.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No active session.{Style.RESET_ALL}")

    def _export_conversation(self, filename: str):
        """Export conversation to file."""
        if not self.current_session:
            print(f"{Fore.RED}No active session to export.{Style.RESET_ALL}")
            return

        try:
            file_path = Path(filename)
            self.current_session.export_conversation(file_path)
            print(f"{Fore.GREEN}Conversation exported to: {file_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error exporting conversation: {e}{Style.RESET_ALL}")

    def _load_agents_from_config(self, config_file: Path) -> Dict[str, AgentConfig]:
        """Load agents from a configuration file."""
        try:
            return AgentConfig.load_from_file(config_file)
        except Exception as e:
            self.logger.warning(f"Could not load agents from {config_file}: {e}")
            return {}

    def _discover_agent_configs(self) -> Dict[str, AgentConfig]:
        """Discover agent configurations from common locations."""
        configs = {}

        # Check common config file locations
        config_locations = [
            Path.cwd() / "agents.json",
            Path.cwd() / "config" / "agents.json",
            Path.cwd() / ".dsat" / "agents.json",
            Path.home() / ".dsat" / "agents.json",
        ]

        for config_file in config_locations:
            if config_file.exists():
                configs.update(self._load_agents_from_config(config_file))

        return configs

    def _check_ollama_health(self, base_url: str = "http://localhost:11434") -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            return False

    def _get_ollama_models(self, base_url: str = "http://localhost:11434") -> List[str]:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [
                    model["name"].split(":")[0] for model in data.get("models", [])
                ]
                return list(set(models))  # Remove duplicates
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            pass
        return []

    def _infer_model_family(self, model_name: str) -> str:
        """Infer model family from model name."""
        model_lower = model_name.lower()

        if "llama" in model_lower:
            return "llama"
        elif "qwen" in model_lower:
            return "qwen"
        elif "gemma" in model_lower:
            return "gemma"
        elif "mistral" in model_lower:
            return "mistral"
        elif "phi" in model_lower:
            return "phi"
        elif "codellama" in model_lower:
            return "llama"
        else:
            return "llm"  # Generic family

    def _prompt_user_for_ollama_model(
        self, available_models: List[str]
    ) -> tuple[str, str]:
        """Prompt user to select from available Ollama models."""
        if not available_models:
            return "llama3.2", "llama"  # Fallback

        if len(available_models) == 1:
            # Only one model available, use it automatically
            model = available_models[0]
            return model, self._infer_model_family(model)

        print(f"\n{Fore.YELLOW}Available Ollama models:{Style.RESET_ALL}")
        for i, model in enumerate(available_models, 1):
            family = self._infer_model_family(model)
            print(f"  {Fore.GREEN}{i}.{Style.RESET_ALL} {model} ({family})")

        while True:
            try:
                choice = input(
                    f"\n{Fore.CYAN}Select a model (1-{len(available_models)}): {Style.RESET_ALL}"
                ).strip()

                if not choice:
                    continue

                choice_num = int(choice)
                if 1 <= choice_num <= len(available_models):
                    selected_model = available_models[choice_num - 1]
                    family = self._infer_model_family(selected_model)
                    print(
                        f"{Fore.GREEN}Selected: {selected_model} ({family}){Style.RESET_ALL}\n"
                    )
                    return selected_model, family
                else:
                    print(
                        f"{Fore.RED}Please enter a number between 1 and {len(available_models)}{Style.RESET_ALL}"
                    )

            except ValueError:
                print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Selection cancelled{Style.RESET_ALL}")
                return available_models[0], self._infer_model_family(
                    available_models[0]
                )  # Use first as fallback

    def _create_default_agent(self, provider: str) -> Optional[AgentConfig]:
        """Create a default agent configuration for the given provider."""
        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            return AgentConfig(
                agent_name="default_claude",
                model_provider="anthropic",
                model_family="claude",
                model_version="claude-3-5-haiku-latest",
                prompt="assistant:latest",
                provider_auth={"api_key": api_key},
            )
        elif provider == "ollama":
            base_url = "http://localhost:11434"

            # Check if Ollama is running
            if not self._check_ollama_health(base_url):
                self.logger.debug("Ollama not running or not accessible")
                return None

            # Get available models
            available_models = self._get_ollama_models(base_url)
            if not available_models:
                self.logger.debug("No Ollama models found")
                return None

            # Prompt user to select model
            model_version, model_family = self._prompt_user_for_ollama_model(
                available_models
            )

            return AgentConfig(
                agent_name="default_ollama",
                model_provider="ollama",
                model_family=model_family,
                model_version=model_version,
                prompt="assistant:latest",
                provider_auth={"base_url": base_url},
            )
        elif provider == "google":
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                return None
            return AgentConfig(
                agent_name="default_gemini",
                model_provider="google",
                model_family="gemini",
                model_version="gemini-1.5-flash",
                prompt="assistant:latest",
                provider_auth={"project_id": project_id, "location": "us-central1"},
            )

        return None

    def _auto_detect_providers(self) -> Dict[str, AgentConfig]:
        """Auto-detect available providers and create default agents."""
        configs = {}

        # Check for Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            config = self._create_default_agent("anthropic")
            if config:
                configs["default_claude"] = config

        # Check for Google Cloud
        if os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        ):
            config = self._create_default_agent("google")
            if config:
                configs["default_gemini"] = config

        # For Ollama, just check if it's available but don't prompt yet
        # Prompting will happen when Ollama is explicitly requested
        if self._check_ollama_health():
            models = self._get_ollama_models()
            if models:
                # We'll create the actual config when needed
                self.ollama_models_available = models

        return configs

    def _resolve_prompts_directory(
        self,
        cli_prompts_dir: Optional[Path],
        agent_config: Optional[AgentConfig],
        config_file: Optional[Path],
    ) -> Path:
        """
        Resolve prompts directory using flexible search strategy.

        Search order:
        1. CLI argument (--prompts-dir)
        2. Agent config prompts_dir field
        3. Config file relative (config_file/prompts)
        4. Current directory (./prompts)
        5. User home directory (~/.dsat/prompts)

        :param cli_prompts_dir: Prompts directory from CLI argument
        :param agent_config: Agent configuration (may contain prompts_dir)
        :param config_file: Config file path (for relative lookup)
        :return: Resolved prompts directory path
        """
        # Priority 1: CLI argument
        if cli_prompts_dir:
            return cli_prompts_dir

        # Priority 2: Agent config prompts_dir field
        if agent_config and agent_config.prompts_dir:
            agent_prompts_path = Path(agent_config.prompts_dir)
            # If relative path and we have a config file, make it relative to config file
            if not agent_prompts_path.is_absolute() and config_file:
                agent_prompts_path = config_file.parent / agent_prompts_path
            return agent_prompts_path

        # Priority 3: Config file relative
        if config_file:
            config_relative = config_file.parent / "prompts"
            if config_relative.exists():
                return config_relative

        # Priority 4: Current directory
        current_dir = Path("prompts")
        if current_dir.exists():
            return current_dir

        # Priority 5: User home directory
        home_dir = Path.home() / ".dsat" / "prompts"
        if home_dir.exists():
            return home_dir

        # Fallback: Use config file relative or current directory
        if config_file:
            return config_file.parent / "prompts"
        else:
            return Path("prompts")

    def initialize_agents(
        self,
        config_file: Optional[Path] = None,
        agent_name: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
        stream: bool = False,
    ) -> bool:
        """
        Initialize agents for the chat session.

        :param config_file: Optional path to agent config file
        :param agent_name: Optional specific agent name to use
        :param provider: Optional provider for inline agent creation
        :param model: Optional model for inline agent creation
        :param prompts_dir: Optional prompts directory override
        :param stream: Enable streaming mode
        :return: True if agents were successfully initialized
        """
        # Store CLI prompts directory and streaming preference for later use
        self.cli_prompts_dir = prompts_dir
        self.config_file = config_file
        self.streaming_enabled = stream

        # Priority 1: Inline agent creation
        if provider and model:
            try:
                config = self._create_default_agent(provider)
                if config:
                    config.model_version = model
                    config.agent_name = f"{provider}_{model}"
                    resolved_prompts_dir = self._resolve_prompts_directory(
                        prompts_dir, config, config_file
                    )
                    agent = Agent.create(
                        config, logger=self.logger, prompts_dir=resolved_prompts_dir
                    )
                    self.current_session = ChatSession(agent)
                    self.available_agents[config.agent_name] = config
                    return True
            except Exception as e:
                print(f"{Fore.RED}Error creating inline agent: {e}{Style.RESET_ALL}")

        # Priority 1b: Provider without model (prompt for Ollama model selection)
        if provider and not model and provider == "ollama":
            try:
                config = self._create_default_agent("ollama")
                if config:
                    config.agent_name = f"ollama_{config.model_version}"
                    resolved_prompts_dir = self._resolve_prompts_directory(
                        prompts_dir, config, config_file
                    )
                    agent = Agent.create(
                        config, logger=self.logger, prompts_dir=resolved_prompts_dir
                    )
                    self.current_session = ChatSession(agent)
                    self.available_agents[config.agent_name] = config
                    return True
            except Exception as e:
                print(f"{Fore.RED}Error creating Ollama agent: {e}{Style.RESET_ALL}")

        # Priority 2: Specific config file
        if config_file:
            self.available_agents = self._load_agents_from_config(config_file)
        else:
            # Priority 3: Auto-discover configs
            self.available_agents = self._discover_agent_configs()

        # Priority 4: Auto-detect providers
        if not self.available_agents:
            self.available_agents = self._auto_detect_providers()

            # If we only found Ollama and no other providers, prompt for model selection
            if not self.available_agents and self.ollama_models_available:
                print(
                    f"{Fore.CYAN}Found Ollama with available models. Let's set it up!{Style.RESET_ALL}"
                )
                try:
                    config = self._create_default_agent("ollama")
                    if config:
                        config.agent_name = f"ollama_{config.model_version}"
                        self.available_agents[config.agent_name] = config
                except Exception as e:
                    print(f"{Fore.RED}Error setting up Ollama: {e}{Style.RESET_ALL}")

        if not self.available_agents:
            print(f"{Fore.RED}No agents available. Please:")
            print("  1. Set environment variables (ANTHROPIC_API_KEY, etc.)")
            print("  2. Create an agents.json config file")
            print(f"  3. Use --provider and --model flags{Style.RESET_ALL}")
            return False

        # Select initial agent
        if agent_name:
            if agent_name in self.available_agents:
                try:
                    config = self.available_agents[agent_name]
                    resolved_prompts_dir = self._resolve_prompts_directory(
                        prompts_dir, config, config_file
                    )
                    agent = Agent.create(
                        config, logger=self.logger, prompts_dir=resolved_prompts_dir
                    )
                    self.current_session = ChatSession(agent)
                except Exception as e:
                    print(
                        f"{Fore.RED}Error loading agent '{agent_name}': {e}{Style.RESET_ALL}"
                    )
                    return False
            else:
                print(
                    f"{Fore.RED}Agent '{agent_name}' not found in configuration.{Style.RESET_ALL}"
                )
                return False
        else:
            # Use first available agent
            first_agent_name = next(iter(self.available_agents))
            try:
                config = self.available_agents[first_agent_name]
                resolved_prompts_dir = self._resolve_prompts_directory(
                    prompts_dir, config, config_file
                )
                agent = Agent.create(
                    config, logger=self.logger, prompts_dir=resolved_prompts_dir
                )
                self.current_session = ChatSession(agent)
            except Exception as e:
                print(f"{Fore.RED}Error loading default agent: {e}{Style.RESET_ALL}")
                return False

        return True

    async def _handle_streaming_response(self, user_input: str):
        """Handle streaming response from agent."""
        print(
            f"{Fore.MAGENTA}🤖 {self.current_session.agent.config.agent_name}:{Style.RESET_ALL}"
        )

        # Collect the full response for history
        full_response = ""

        try:
            async for chunk in self.current_session.agent.invoke_async(user_input):
                print(chunk, end="", flush=True)
                full_response += chunk

            print()  # New line after streaming is complete
            print()  # Extra line for spacing

            # Add full response to history
            self.current_session.add_message("assistant", full_response)

        except Exception as e:
            print(f"\n{Fore.RED}Error during streaming: {e}{Style.RESET_ALL}")
            return None

        return full_response

    async def start_chat(self):
        """Start the interactive chat loop."""
        if not self.current_session:
            print(
                f"{Fore.RED}No active chat session. Please initialize an agent first.{Style.RESET_ALL}"
            )
            return

        self._print_banner()

        agent_name = self.current_session.agent.config.agent_name
        model_info = f"{self.current_session.agent.config.model_provider}/{self.current_session.agent.config.model_version}"
        stream_status = "ON" if self.streaming_enabled else "OFF"

        print(
            f"🤖 Active Agent: {Fore.GREEN}{agent_name}{Style.RESET_ALL} ({model_info})"
        )
        print(f"🌊 Streaming: {Fore.CYAN}{stream_status}{Style.RESET_ALL}")
        print(
            f"💡 Type {Fore.GREEN}/help{Style.RESET_ALL} for commands, {Fore.GREEN}/quit{Style.RESET_ALL} to exit"
        )
        print()

        try:
            while True:
                # Get user input
                user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break
                    continue

                # Add user message to history
                self.current_session.add_message("user", user_input)

                # Get agent response
                try:
                    if self.streaming_enabled:
                        # Use streaming response
                        print(f"{Fore.YELLOW}🤔 Thinking...{Style.RESET_ALL}")
                        await self._handle_streaming_response(user_input)
                    else:
                        # Use traditional response
                        print(f"{Fore.YELLOW}🤔 Thinking...{Style.RESET_ALL}")
                        response = self.current_session.agent.invoke(user_input)

                        # Print agent response
                        print(
                            f"{Fore.MAGENTA}🤖 {self.current_session.agent.config.agent_name}:{Style.RESET_ALL}"
                        )
                        print(response)
                        print()

                        # Add agent response to history
                        self.current_session.add_message("assistant", response)

                except KeyboardInterrupt:
                    print(f"\n{Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
                    continue
                except Exception as e:
                    print(f"{Fore.RED}Error getting response: {e}{Style.RESET_ALL}")
                    continue

        except KeyboardInterrupt:
            pass

        print(f"\n{Fore.CYAN}Thanks for chatting! 👋{Style.RESET_ALL}")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the chat command."""
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for DSAT agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dsat chat                                    # Auto-detect agents
  dsat chat --agent my_assistant               # Use specific agent
  dsat chat --config ./agents.json            # Use specific config file
  dsat chat --provider anthropic --model claude-3-5-haiku-latest
        """,
    )

    parser.add_argument(
        "--config", "-c", type=Path, help="Path to agent configuration file (JSON/TOML)"
    )

    parser.add_argument("--agent", "-a", help="Name of agent to use (from config file)")

    parser.add_argument(
        "--provider",
        "-p",
        choices=["anthropic", "google", "ollama"],
        help="LLM provider for inline agent creation",
    )

    parser.add_argument("--model", "-m", help="Model version for inline agent creation")

    parser.add_argument(
        "--no-colors", action="store_true", help="Disable colored output"
    )

    parser.add_argument(
        "--prompts-dir", "-d", type=Path, help="Directory containing prompt TOML files"
    )

    parser.add_argument(
        "--stream",
        "-s",
        action="store_true",
        help="Enable streaming mode for real-time token output",
    )

    return parser


def main(args: Optional[List[str]] = None):
    """Main entry point for the chat command."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Disable colors if requested or not available
    if parsed_args.no_colors or not COLORS_AVAILABLE:
        global Fore, Style
        Fore = MockColorama.Fore
        Style = MockColorama.Style

    # Create chat interface
    chat = ChatInterface()

    # Initialize agents
    success = chat.initialize_agents(
        config_file=parsed_args.config,
        agent_name=parsed_args.agent,
        provider=parsed_args.provider,
        model=parsed_args.model,
        prompts_dir=parsed_args.prompts_dir,
        stream=parsed_args.stream,
    )

    if not success:
        sys.exit(1)

    # Start chat (run async if streaming is enabled)
    if parsed_args.stream or chat.streaming_enabled:
        asyncio.run(chat.start_chat())
    else:
        # For backward compatibility, also support async even when not streaming
        asyncio.run(chat.start_chat())


if __name__ == "__main__":
    main()
