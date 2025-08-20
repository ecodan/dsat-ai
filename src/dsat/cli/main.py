"""
Main CLI entry point for DSAT.
"""

import sys
import argparse
from ..scryptorum.cli.commands import main as scryptorum_main


def main():
    """Main DSAT CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DSAT - Dan's Simple Agent Toolkit",
        epilog="Use 'dsat chat --help' for interactive chat or 'dsat scryptorum --help' for experiment management",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat subcommand
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive chat interface for LLM agents"
    )

    # Add chat-specific arguments directly
    chat_parser.add_argument(
        "--config", "-c", type=str, help="Path to agent configuration file (JSON/TOML)"
    )

    chat_parser.add_argument(
        "--agent", "-a", help="Name of agent to use (from config file)"
    )

    chat_parser.add_argument(
        "--provider",
        "-p",
        choices=["anthropic", "google", "ollama"],
        help="LLM provider for inline agent creation",
    )

    chat_parser.add_argument(
        "--model", "-m", help="Model version for inline agent creation"
    )

    chat_parser.add_argument(
        "--no-colors", action="store_true", help="Disable colored output"
    )

    chat_parser.add_argument(
        "--prompts-dir", "-d", type=str, help="Directory containing prompt TOML files"
    )

    chat_parser.add_argument(
        "--stream",
        "-s",
        action="store_true",
        help="Enable streaming mode for real-time token output",
    )

    # Scryptorum subcommand
    scryptorum_parser = subparsers.add_parser(
        "scryptorum", help="Scryptorum experiment management commands"
    )
    scryptorum_parser.add_argument(
        "scryptorum_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to scryptorum",
    )

    args = parser.parse_args()

    if args.command == "chat":
        # Import and run chat interface
        from .chat import ChatInterface
        from pathlib import Path

        # Convert config path to Path object if provided
        config_path = Path(args.config) if args.config else None
        prompts_path = Path(args.prompts_dir) if args.prompts_dir else None

        # Create and configure chat interface
        chat = ChatInterface()

        # Disable colors if requested
        if args.no_colors:
            global Fore, Style
            from .chat import MockColorama
            import dsat.cli.chat as chat_module

            chat_module.Fore = MockColorama.Fore
            chat_module.Style = MockColorama.Style

        # Initialize agents
        success = chat.initialize_agents(
            config_file=config_path,
            agent_name=args.agent,
            provider=args.provider,
            model=args.model,
            prompts_dir=prompts_path,
            stream=args.stream,
        )

        if not success:
            sys.exit(1)

        # Start chat (always run async since chat interface is now async)
        import asyncio

        asyncio.run(chat.start_chat())
    elif args.command == "scryptorum":
        # Replace sys.argv with scryptorum args and call scryptorum main
        original_argv = sys.argv[:]
        sys.argv = ["scryptorum"] + args.scryptorum_args
        try:
            scryptorum_main()
        finally:
            sys.argv = original_argv
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
