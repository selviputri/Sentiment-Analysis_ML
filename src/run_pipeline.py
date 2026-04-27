"""Master pipeline runner for the sentiment analysis project."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(step_name: str, command: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"RUNNING: {step_name}")
    print(f"{'='*60}")

    result = subprocess.run(command, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\nFAILED: {step_name}")
        sys.exit(result.returncode)

    print(f"\nCOMPLETED: {step_name}")


def main() -> None:
    python_executable = sys.executable

    # Step 1: Dataset validation / split
    run_step(
        "Dataset Pipeline",
        [python_executable, "src/dataset.py"]
    )

    # Step 2: Train model
    run_step(
        "Model Training",
        [python_executable, "src/train.py", "--config", "configs/base.yaml"]
    )

    # Step 3: Evaluate model
    run_step(
        "Model Evaluation",
        [
            python_executable,
            "src/evaluate.py",
            "--config",
            "configs/base.yaml",
            "--checkpoint",
            "results/saved_models/best.pth",
        ]
    )

    # Optional Step 4:
    # run_step("Prediction", [python_executable, "src/predict.py"])

    # Optional Step 5:
    # run_step("Visualization", [python_executable, "src/visualize_results.py"])

    print(f"\n{'='*60}")
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()