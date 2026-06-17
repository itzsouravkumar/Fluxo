#!/usr/bin/env python3
"""Train FLUXO TFT congestion forecaster."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("FLUXO TFT Forecaster Training")
    print("=" * 40)

    data_path = "data/btp_round1_dataset.csv"
    if not Path(data_path).exists():
        print(f"Training data not found: {data_path}")
        print("Place the BTP Round 1 dataset in data/ directory")
        return

    print("Training TFT model...")
    print("This will be implemented with pytorch-forecasting.")


if __name__ == "__main__":
    main()
