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
        epilog="Use 'dsat scryptorum --help' for experiment management commands"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Scryptorum subcommand
    scryptorum_parser = subparsers.add_parser(
        "scryptorum", 
        help="Scryptorum experiment management commands"
    )
    scryptorum_parser.add_argument(
        "scryptorum_args", 
        nargs=argparse.REMAINDER,
        help="Arguments to pass to scryptorum"
    )
    
    args = parser.parse_args()
    
    if args.command == "scryptorum":
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