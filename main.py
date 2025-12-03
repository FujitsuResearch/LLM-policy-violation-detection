# Copyright © 2025 Fujitsu Research of Europe
# Licensed under the Apache License, Version 2.0
import argparse
from whitening_eval_dynabench import DynabenchWhiteningEval


def main():
    parser = argparse.ArgumentParser(
        description="Policy Violation Detection Demo CLI"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llama-3.1-8b-instruct", "Qwen2.5-7B-Instruct"],
        help="Choose which model's precomputed stats to use.",
    )

    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Optional: number of samples from the dataset to run.",
    )

    args = parser.parse_args()

    print(f"[INFO] Starting evaluation for model: {args.model}")

    evaluator = DynabenchWhiteningEval(
        model=args.model,
        dataset_slice=args.slice,
    )

    evaluator.run_evaluation()

    print("[INFO] Evaluation completed.")


if __name__ == "__main__":
    main()
