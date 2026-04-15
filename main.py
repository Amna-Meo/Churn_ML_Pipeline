import argparse
import logging
import sys
import os
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--input", type=str, help="Input CSV file for prediction")
    parser.add_argument(
        "--model", type=str, default="models/churn_pipeline.pkl", help="Model file path"
    )
    parser.add_argument(
        "--version", action="store_true", help="Save model with timestamp version"
    )

    args = parser.parse_args()

    if args.input:
        logger.info(f"Running inference on: {args.input}")
        from src.pipeline import load_and_predict

        load_and_predict(args.input, args.model)
    else:
        logger.info("Starting ML Pipeline...")
        from src.pipeline import run_pipeline

        model_path = run_pipeline(versioned=args.version)
        logger.info(f"Pipeline complete. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
