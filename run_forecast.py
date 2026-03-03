import argparse
from src.forecast import run_forecast, BacktestConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_days", type=int, default=7, help="How many last days to evaluate in walk-forward")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = BacktestConfig(test_days=args.test_days)
    run_forecast(cfg)