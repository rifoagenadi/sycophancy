"""Run uncertainty quantification on model responses.

Usage:
    python run_uqlm.py <model_name>

Example:
    python run_uqlm.py gemma-3-12b-it
    python run_uqlm.py Llama-3.1-8B-Instruct
"""

import argparse
import asyncio
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI

from uqlm import WhiteBoxUQ, BlackBoxUQ

RESPONSES_DIR = Path("responses")
OUTPUT_DIR = Path("uncertainty_result")


async def main():
    parser = argparse.ArgumentParser(description="Run UQ scoring on model responses")
    parser.add_argument("model", help="Full model name (e.g. google/gemma-3-12b-it)")
    args = parser.parse_args()

    # Parse short name from "org/model-name" format
    short_name = args.model.split("/")[-1]
    filepath = RESPONSES_DIR / f"{short_name}_opinion_responses.csv"
    if not filepath.exists():
        available = [f.stem.replace("_opinion_responses", "") for f in RESPONSES_DIR.glob("*_opinion_responses.csv")]
        raise FileNotFoundError(f"No response file for '{short_name}'. Available: {available}")

    # Local vLLM server
    llm = ChatOpenAI(
        model=args.model,
        base_url="http://localhost:8000/v1",
        api_key="unused",
        temperature=0.7,
    )

    wbuq = WhiteBoxUQ(llm=llm, scorers=["mean_token_negentropy"])
    bbuq = BlackBoxUQ(
        llm=llm,
        scorers=["semantic_negentropy"],
        nli_model_name="microsoft/deberta-large-mnli",
        device="cuda:1",
    )

    df = pd.read_csv(filepath)
    prompts = df["question"].tolist()

    print(f"Processing {filepath.name} ({len(prompts)} prompts)...")

    results = await bbuq.generate_and_score(prompts=prompts, num_responses=5)
    df["semantic_negentropy"] = results.to_df()["semantic_negentropy"].values

    results = await wbuq.generate_and_score(prompts=prompts)
    result_df = results.to_df()

    score_cols = [c for c in result_df.columns if c not in ("prompt", "response", "logprob")]
    for col in score_cols:
        df[col] = result_df[col].values

    COLUMNS_TO_DROP = ["first_answer", "disagreement", "second_answer"]
    for col in COLUMNS_TO_DROP:
        df.drop(columns=[col], inplace=True)

    out_path = OUTPUT_DIR / filepath.name
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
