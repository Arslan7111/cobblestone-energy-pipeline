import argparse
from src.llm_note_gemini_rest import run_llm_note

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--show_prompt", action="store_true", help="Print the full prompt")
    args = p.parse_args()
    run_llm_note(show_prompt=args.show_prompt)