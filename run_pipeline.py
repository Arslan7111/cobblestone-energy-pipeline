from datetime import datetime
import argparse

from src.ingest_elexon import run_ingest
from src.qa import run_qa


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2026-01-29", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", type=str, default="2026-02-28", help="YYYY-MM-DD (inclusive)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ingest(start_date=args.start, end_date=args.end)

    qa = run_qa()
    print("\nQA summary (saved to logs/qa_report.json):")
    print(qa)