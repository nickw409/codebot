"""
CLI entry point for the codebase Q&A chatbot.

Usage:
    # First, start the database:
    docker compose up -d

    # Ingest a codebase (run once, or again after code changes):
    python main.py --ingest /path/to/your/codebase

    # Start chatting:
    python main.py /path/to/your/codebase

    # Both at once (ingest then chat):
    python main.py --ingest /path/to/your/codebase

The codebase path is required so that the file tools (list_files, get_file_contents)
know which directory they're allowed to access.
"""

import argparse
import sys

from conversation.engine import chat
from conversation.history import create_conversation
from ingestion.ingest import ingest
from tools.registry import set_base_directory


def main():
    parser = argparse.ArgumentParser(
        description="Codebase Q&A Chatbot — ask questions about your code"
    )
    parser.add_argument(
        "codebase",
        help="Path to the codebase directory to analyze",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Re-ingest the codebase before starting the chat",
    )

    args = parser.parse_args()

    # Tell the tool registry which directory tools are allowed to access.
    set_base_directory(args.codebase)

    # Optionally re-ingest the codebase.
    if args.ingest:
        print(f"Ingesting codebase from {args.codebase}...\n")
        count = ingest(args.codebase)
        if count == 0:
            print("Warning: no chunks were ingested. Chat may not be useful.")
        print()

    # Create a new conversation session.
    conv_id = create_conversation()
    print("Codebot ready. Ask questions about your codebase.")
    print("Type 'quit' or Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            response = chat(conv_id, user_input)
            print(f"\nBot: {response}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()
