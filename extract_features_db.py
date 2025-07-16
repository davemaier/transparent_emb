#!/usr/bin/env python3

import json
import asyncio
import aiohttp
import argparse
from typing import List, Dict, Any, Optional
import os
import numpy as np
import asyncpg
from datetime import datetime


async def call_openrouter(
    session: aiohttp.ClientSession, prompt: str, api_key: str
) -> Dict[str, Any]:
    """Call OpenRouter API with a single prompt and return full response"""
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

        return {
            "content": result["choices"][0]["message"]["content"].strip().lower(),
            "generation_id": result.get("id"),
            "usage": result.get("usage", {}),
        }


def estimate_cost_from_usage(
    usage: Dict[str, Any], model: str = "meta-llama/llama-4-scout"
) -> Optional[float]:
    """Estimate cost from token usage based on OpenRouter pricing"""
    if not usage:
        return None

    # Llama 4 Scout pricing (as of 2025)
    if model == "meta-llama/llama-4-scout":
        input_cost_per_million = 0.08
        output_cost_per_million = 0.30
    else:
        # Default pricing (you can extend this for other models)
        input_cost_per_million = 0.08
        output_cost_per_million = 0.30

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million
    output_cost = (completion_tokens / 1_000_000) * output_cost_per_million

    return input_cost + output_cost


async def get_generation_cost(
    session: aiohttp.ClientSession,
    generation_id: str,
    api_key: str,
    usage: Dict[str, Any],
) -> Optional[float]:
    """Get the actual cost of a generation from OpenRouter, fallback to usage estimation"""
    if not generation_id:
        return estimate_cost_from_usage(usage)

    url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Wait 1 second for generation data to be available
    await asyncio.sleep(1.0)

    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                total_cost = result.get("total_cost")
                if total_cost is not None and total_cost > 0:
                    return total_cost
                else:
                    # Even if we get a 200, if total_cost is None/0, fallback to usage estimation
                    return estimate_cost_from_usage(usage)
            else:
                # Fallback to usage-based estimation
                return estimate_cost_from_usage(usage)
    except Exception as e:
        # Fallback to usage-based estimation
        return estimate_cost_from_usage(usage)


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
            response_data = await call_openrouter(session, formatted_prompt, api_key)
            result = response_data["content"] == "true"

            # Get the actual cost from the generation endpoint
            cost = await get_generation_cost(
                session, response_data["generation_id"], api_key, response_data["usage"]
            )

            return {
                "prompt_id": prompt_data["id"],
                "result": result,
                "raw_response": response_data["content"],
                "generation_id": response_data["generation_id"],
                "cost": cost,
                "usage": response_data["usage"],
                "success": True,
            }
        except Exception as e:
            return {
                "prompt_id": prompt_data["id"],
                "result": None,
                "error": str(e),
                "success": False,
                "cost": None,
            }


async def run_all_prompts(
    prompts: List[Dict[str, str]], text_segment: str, max_parallel: int, api_key: str
) -> List[Dict[str, Any]]:
    """Run all prompts in parallel with specified concurrency limit"""
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


async def get_db_connection(db_url: str) -> asyncpg.Connection:
    """Create database connection"""
    return await asyncpg.connect(db_url)


async def fetch_text_segments(
    conn: asyncpg.Connection, model_run_id: int, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetch text segments from generated_segments table for specific model run"""
    query = "SELECT id, content FROM generated_segments WHERE seg_model_run_id = $1"

    if limit:
        query += f" LIMIT {limit}"

    rows = await conn.fetch(query, model_run_id)
    return [
        {"id": row["id"], "text": row["content"], "ground_truth": None} for row in rows
    ]


async def fetch_segments_with_ground_truth(
    conn: asyncpg.Connection, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetch segments from segments table with ground truth labels"""
    query = (
        "SELECT id, content, ground_truth FROM segments WHERE ground_truth IS NOT NULL"
    )

    if limit:
        query += f" LIMIT {limit}"

    rows = await conn.fetch(query)
    return [
        {"id": row["id"], "text": row["content"], "ground_truth": row["ground_truth"]}
        for row in rows
    ]


async def create_feature_extraction_model_run(
    conn: asyncpg.Connection,
    seg_model_run_id: Optional[int],
    prompts_file: str,
    max_parallel: int,
    limit: Optional[int] = None,
    description_suffix: str = "",
) -> int:
    """Create a new model run for feature extraction"""
    name = f"Feature Extraction {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if description_suffix:
        name += f" - {description_suffix}"

    if seg_model_run_id is not None:
        description = f"Feature extraction using LLM prompts from segmentation run {seg_model_run_id}"
    else:
        description = (
            "Feature extraction using LLM prompts from segments table (ground truth)"
        )

    metadata = {
        "source_seg_model_run_id": seg_model_run_id,
        "prompts_file": prompts_file,
        "max_parallel": max_parallel,
        "limit": limit,
        "timestamp": datetime.now().isoformat(),
        "extraction_type": "feature_extraction_llm",
        "api_provider": "OpenRouter",
        "llm_model": "meta-llama/llama-4-scout",
    }

    result = await conn.fetchrow(
        "INSERT INTO model_runs (name, description, metadata, type, created_at) VALUES ($1, $2, $3, $4, $5) RETURNING id",
        name,
        description,
        json.dumps(metadata),
        "featureExtraction",
        datetime.now(),
    )

    return result["id"]


async def save_feature_vector(
    conn: asyncpg.Connection,
    feature_extraction_model_run_id: int,
    seg_model_run_id: Optional[int],
    segment_id: int,
    feature_vector: np.ndarray,
    prompts: List[Dict[str, str]],
    ground_truth: Optional[str] = None,
    is_generated_segment: bool = True,
    total_cost: Optional[float] = None,
) -> None:
    """Save feature vector to database"""
    # Convert numpy array to list for JSON serialization
    feature_list = feature_vector.tolist()

    # Create mapping of prompt IDs to their positions in the feature vector
    prompt_ids = [prompt["id"] for prompt in sorted(prompts, key=lambda x: x["id"])]

    if is_generated_segment:
        # For generated segments
        await conn.execute(
            """
            INSERT INTO feature_vectors (feature_extraction_model_run_id, seg_model_run_id, generated_segment_id, feature_vector, prompt_ids, ground_truth, total_cost, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (feature_extraction_model_run_id, generated_segment_id) 
            DO UPDATE SET 
                feature_vector = EXCLUDED.feature_vector,
                prompt_ids = EXCLUDED.prompt_ids,
                ground_truth = EXCLUDED.ground_truth,
                total_cost = EXCLUDED.total_cost,
                created_at = EXCLUDED.created_at
            """,
            feature_extraction_model_run_id,
            seg_model_run_id,
            segment_id,
            json.dumps(feature_list),
            json.dumps(prompt_ids),
            ground_truth,
            total_cost,
            datetime.now(),
        )
    else:
        # For ground truth segments
        await conn.execute(
            """
            INSERT INTO feature_vectors (feature_extraction_model_run_id, seg_model_run_id, ground_truth_segment_id, feature_vector, prompt_ids, ground_truth, total_cost, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (feature_extraction_model_run_id, ground_truth_segment_id) 
            DO UPDATE SET 
                feature_vector = EXCLUDED.feature_vector,
                prompt_ids = EXCLUDED.prompt_ids,
                ground_truth = EXCLUDED.ground_truth,
                total_cost = EXCLUDED.total_cost,
                created_at = EXCLUDED.created_at
            """,
            feature_extraction_model_run_id,
            seg_model_run_id,
            segment_id,
            json.dumps(feature_list),
            json.dumps(prompt_ids),
            ground_truth,
            total_cost,
            datetime.now(),
        )


async def process_segments(
    seg_model_run_id: Optional[int],
    prompts_file: str,
    max_parallel: int,
    api_key: str,
    db_url: str,
    limit: Optional[int] = None,
    start_id: Optional[int] = None,
    description_suffix: str = "",
    use_segments: bool = False,
) -> None:
    """Process text segments from database and save feature vectors"""

    # Load prompts
    with open(prompts_file, "r") as f:
        prompts = json.load(f)

    # Connect to database
    conn = await get_db_connection(db_url)

    try:
        # Create new model run for this feature extraction task
        feature_extraction_model_run_id = await create_feature_extraction_model_run(
            conn,
            seg_model_run_id,
            prompts_file,
            max_parallel,
            limit,
            description_suffix,
        )

        print(f"Created feature extraction model run {feature_extraction_model_run_id}")

        # Fetch text segments based on source
        if use_segments:
            segments = await fetch_segments_with_ground_truth(conn, limit)
            print(
                f"Processing {len(segments)} segments from segments table (ground truth data)"
            )
        else:
            if seg_model_run_id is None:
                raise ValueError(
                    "seg_model_run_id is required when not using segments table"
                )
            segments = await fetch_text_segments(conn, seg_model_run_id, limit)
            print(
                f"Processing {len(segments)} segments from generated_segments table (segmentation model run {seg_model_run_id})"
            )

        # Filter by start_id if provided
        if start_id:
            segments = [seg for seg in segments if seg["id"] >= start_id]

        # Process each segment
        for i, segment in enumerate(segments):
            print(f"Processing segment {i + 1}/{len(segments)}: ID {segment['id']}")

            # Skip if text is empty
            if not segment["text"] or not segment["text"].strip():
                print(f"  Skipping empty segment {segment['id']}")
                continue

            try:
                # Run feature extraction
                results = await run_all_prompts(
                    prompts, segment["text"], max_parallel, api_key
                )

                # Create feature vector
                feature_vector = create_feature_vector(results)

                # Calculate total cost for this segment
                total_cost = sum(
                    result.get("cost", 0) or 0
                    for result in results
                    if result["success"]
                )

                # Save to database
                await save_feature_vector(
                    conn,
                    feature_extraction_model_run_id,
                    seg_model_run_id,
                    segment["id"],
                    feature_vector,
                    prompts,
                    segment.get("ground_truth"),
                    is_generated_segment=not use_segments,
                    total_cost=total_cost
                    if total_cost is not None and total_cost >= 0
                    else None,
                )

                # Debug: Show cost calculation details
                successful_results = [r for r in results if r["success"]]
                costs = [r.get("cost", 0) or 0 for r in successful_results]
                print(
                    f"  DEBUG: {len(successful_results)} successful prompts, individual costs: {costs[:5]}... (showing first 5)"
                )
                print(f"  DEBUG: Total cost calculated: {total_cost}")

                ground_truth_info = (
                    f" (ground truth: {segment.get('ground_truth')})"
                    if segment.get("ground_truth")
                    else ""
                )
                cost_info = (
                    f" (cost: ${total_cost:.6f})"
                    if total_cost is not None and total_cost >= 0
                    else " (cost: unknown)"
                )
                print(
                    f"  Saved feature vector for segment {segment['id']} (shape: {feature_vector.shape}){ground_truth_info}{cost_info}"
                )

            except Exception as e:
                print(f"  Error processing segment {segment['id']}: {e}")
                continue

        # Calculate and display total cost summary
        total_cost_query = f"""
        SELECT 
            COUNT(*) as total_segments,
            SUM(total_cost) as total_cost,
            AVG(total_cost) as avg_cost_per_segment
        FROM feature_vectors 
        WHERE feature_extraction_model_run_id = {feature_extraction_model_run_id}
        """

        cost_summary = await conn.fetchrow(total_cost_query)

        print(
            f"\nProcessing complete! Feature extraction model run ID: {feature_extraction_model_run_id}"
        )
        print(f"Processed {cost_summary['total_segments']} segments")
        if cost_summary["total_cost"]:
            print(f"Total cost: ${cost_summary['total_cost']:.6f}")
            print(
                f"Average cost per segment: ${cost_summary['avg_cost_per_segment']:.6f}"
            )
        else:
            print("Cost information not available")

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from database text segments and store results"
    )
    parser.add_argument(
        "--seg-model-run-id",
        type=int,
        help="Segmentation model run ID to process (required unless --use-segments is specified)",
    )
    parser.add_argument(
        "--prompts", default="prompts.json", help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--max-parallel", type=int, default=5, help="Max parallel requests"
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://traindata:traindata@localhost:5433/traindata",
        help="Database connection URL",
    )
    parser.add_argument("--limit", type=int, help="Limit number of segments to process")
    parser.add_argument(
        "--start-id", type=int, help="Start processing from specific segment ID"
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Additional description suffix for the model run",
    )
    parser.add_argument(
        "--use-segments",
        action="store_true",
        help="Use segments table instead of generated_segments (includes ground truth labels)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.use_segments and args.seg_model_run_id is None:
        print(
            "Error: --seg-model-run-id is required unless --use-segments is specified"
        )
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return

    asyncio.run(
        process_segments(
            args.seg_model_run_id,
            args.prompts,
            args.max_parallel,
            api_key,
            args.db_url,
            args.limit,
            args.start_id,
            args.description or "",
            args.use_segments,
        )
    )


if __name__ == "__main__":
    main()
