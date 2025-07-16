#!/usr/bin/env python3

import json
import asyncio
import aiohttp
import argparse
from typing import List, Dict, Any
import os
import numpy as np


async def call_openrouter(
    session: aiohttp.ClientSession, prompt: str, api_key: str
) -> str:
    """Call OpenRouter API with a single prompt"""
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {
        "model": "meta-llama/llama-4-scout",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0,
    }

    async with session.post(url, headers=headers, json=data) as response:
        result = await response.json()

        # Debug: print response if error
        if "choices" not in result:
            print(f"API Error Response: {result}")
            raise Exception(
                f"API Error: {result.get('error', {}).get('message', 'Unknown error')}"
            )

        return result["choices"][0]["message"]["content"].strip().lower()


async def process_prompt(
    session: aiohttp.ClientSession,
    prompt_data: Dict[str, str],
    text_segment: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process a single prompt with rate limiting"""
    async with semaphore:
        formatted_prompt = prompt_data["prompt"].format(text=text_segment)
        try:
            response = await call_openrouter(session, formatted_prompt, api_key)
            result = response == "true"
            return {
                "prompt_id": prompt_data["id"],
                "result": result,
                "raw_response": response,
                "success": True,
            }
        except Exception as e:
            return {
                "prompt_id": prompt_data["id"],
                "result": None,
                "error": str(e),
                "success": False,
            }


async def run_all_prompts(
    prompts_file: str, text_segment: str, max_parallel: int, api_key: str
) -> List[Dict[str, Any]]:
    """Run all prompts in parallel with specified concurrency limit"""

    with open(prompts_file, "r") as f:
        prompts = json.load(f)

    semaphore = asyncio.Semaphore(max_parallel)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_prompt(session, prompt, text_segment, api_key, semaphore)
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks)

    return results


def create_feature_vector(results: List[Dict[str, Any]]) -> np.ndarray:
    """Convert results to numpy feature vector of 0s and 1s"""
    # Sort results by prompt_id to ensure consistent ordering
    sorted_results = sorted(results, key=lambda x: x["prompt_id"])
    
    # Create feature vector
    feature_vector = []
    for result in sorted_results:
        if result["success"]:
            # Convert boolean result to int (True -> 1, False -> 0)
            feature_vector.append(int(result["result"]))
        else:
            # Use 0 for failed requests (could also use -1 or NaN depending on requirements)
            feature_vector.append(0)
    
    return np.array(feature_vector, dtype=int)


def main():
    parser = argparse.ArgumentParser(
        description="Run prompts in parallel via OpenRouter and output feature vector"
    )
    parser.add_argument(
        "--prompts", default="prompts.json", help="Path to prompts JSON file"
    )
    parser.add_argument("--text", required=True, help="Text segment to analyze")
    parser.add_argument(
        "--max-parallel", type=int, default=5, help="Max parallel requests"
    )
    parser.add_argument(
        "--output", choices=["vector", "detailed"], default="vector",
        help="Output format: 'vector' for numpy array, 'detailed' for full results"
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return

    results = asyncio.run(
        run_all_prompts(args.prompts, args.text, args.max_parallel, api_key)
    )

    if args.output == "vector":
        # Output feature vector
        feature_vector = create_feature_vector(results)
        print(f"Feature vector shape: {feature_vector.shape}")
        print(f"Feature vector: {feature_vector}")
    else:
        # Output detailed results (original format)
        print(f"Results for text: '{args.text[:50]}...'")
        print("-" * 60)
        for result in results:
            if result["success"]:
                print(f"{result['prompt_id']}: {result['result']}")
            else:
                print(f"{result['prompt_id']}: ERROR - {result['error']}")


if __name__ == "__main__":
    main()
