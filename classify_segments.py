#!/usr/bin/env python3

import asyncio
import asyncpg
import argparse
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


async def get_db_connection(db_url: str) -> asyncpg.Connection:
    """Create database connection"""
    return await asyncpg.connect(db_url)


def load_decision_tree_model(model_path: str) -> Dict[str, Any]:
    """Load the pretrained decision tree model"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"Loaded model trained on feature extraction run: {model_data['feature_extraction_model_run_id']}")
    print(f"Model labels: {model_data['labels']}")
    print(f"Model type: {model_data['model_type']}")

    return model_data


async def fetch_feature_vectors(
    conn: asyncpg.Connection,
    seg_model_run_id: int
) -> List[Dict[str, Any]]:
    """Fetch feature vectors for segments from the specified segmentation model run"""
    query = """
    SELECT
        fv.id as feature_vector_id,
        fv.generated_segment_id,
        fv.vector,
        gs.content as segment_content
    FROM feature_vectors fv
    JOIN generated_segments gs ON fv.generated_segment_id = gs.id
    WHERE gs.seg_model_run_id = $1
    AND fv.generated_segment_id IS NOT NULL
    ORDER BY fv.generated_segment_id
    """

    rows = await conn.fetch(query, seg_model_run_id)

    if not rows:
        raise ValueError(f"No feature vectors found for segmentation model run {seg_model_run_id}")

    print(f"Found {len(rows)} feature vectors for segmentation model run {seg_model_run_id}")

    return [
        {
            'feature_vector_id': row['feature_vector_id'],
            'generated_segment_id': row['generated_segment_id'],
            'vector': row['vector'],
            'content': row['segment_content']
        }
        for row in rows
    ]


async def create_classification_model_run(
    conn: asyncpg.Connection,
    seg_model_run_id: int,
    model_path: str,
    model_info: Dict[str, Any]
) -> int:
    """Create a new model run for the classification task"""
    name = f"Decision Tree Classification {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    description = f"Decision tree classification of segments from segmentation run {seg_model_run_id}"

    metadata = {
        "seg_model_run_id": seg_model_run_id,
        "model_path": model_path,
        "model_type": "decision_tree_classification",
        "feature_extraction_model_run_id": model_info.get('feature_extraction_model_run_id'),
        "model_labels": model_info.get('labels'),
        "timestamp": datetime.now().isoformat()
    }

    result = await conn.fetchrow(
        "INSERT INTO model_runs (name, description, metadata, type, seg_model_run_id, created_at) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        name,
        description,
        json.dumps(metadata),
        "classification",
        seg_model_run_id,
        datetime.now()
    )

    return result["id"]


async def save_predictions(
    conn: asyncpg.Connection,
    model_run_id: int,
    predictions: List[Dict[str, Any]]
) -> None:
    """Save predictions to the segment_predictions table"""
    for pred in predictions:
        await conn.execute(
            """
            INSERT INTO segment_predictions (generated_segment_id, model_run_id, classification, confidence, metadata, reason, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (generated_segment_id, model_run_id)
            DO UPDATE SET
                classification = EXCLUDED.classification,
                confidence = EXCLUDED.confidence,
                metadata = EXCLUDED.metadata,
                reason = EXCLUDED.reason,
                created_at = EXCLUDED.created_at
            """,
            pred['generated_segment_id'],
            model_run_id,
            pred['classification'],
            pred['confidence'],
            json.dumps(pred['metadata']),
            pred['reason'],
            datetime.now()
        )


async def classify_segments(
    seg_model_run_id: int,
    model_path: str,
    db_url: str
) -> None:
    """Main function to classify segments using the pretrained decision tree"""

    # Load the pretrained model
    model_data = load_decision_tree_model(model_path)
    classifier = model_data['model']
    labels = model_data['labels']

    # Connect to database
    conn = await get_db_connection(db_url)

    try:
        # Create new model run for this classification task
        classification_model_run_id = await create_classification_model_run(
            conn, seg_model_run_id, model_path, model_data
        )

        print(f"Created classification model run {classification_model_run_id}")

        # Fetch feature vectors for the specified segmentation model run
        feature_data = await fetch_feature_vectors(conn, seg_model_run_id)

        print(f"Classifying {len(feature_data)} segments...")

        # Prepare feature matrix
        X = np.array([item['vector'] for item in feature_data])

        # Make predictions
        predictions = classifier.predict(X)
        prediction_probabilities = classifier.predict_proba(X)

        # Prepare prediction data for database
        prediction_records = []
        for i, item in enumerate(feature_data):
            predicted_label = predictions[i]
            confidence = float(np.max(prediction_probabilities[i]))

            # Create prediction record
            prediction_records.append({
                'generated_segment_id': item['generated_segment_id'],
                'classification': predicted_label,
                'confidence': confidence,
                'metadata': {
                    'feature_vector_id': item['feature_vector_id'],
                    'prediction_probabilities': {
                        label: float(prob) for label, prob in zip(labels, prediction_probabilities[i])
                    }
                },
                'reason': f"Decision tree prediction with {confidence:.3f} confidence"
            })

        # Save predictions to database
        await save_predictions(conn, classification_model_run_id, prediction_records)

        # Print classification summary
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        print(f"\nClassification Summary:")
        for label, count in zip(unique_predictions, counts):
            percentage = (count / len(predictions)) * 100
            print(f"  {label}: {count} segments ({percentage:.1f}%)")

        print(f"\nClassification completed! Model run ID: {classification_model_run_id}")
        print(f"Processed {len(feature_data)} segments")

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Classify segments using pretrained decision tree model"
    )
    parser.add_argument(
        "--seg-model-run-id",
        type=int,
        required=True,
        help="Segmentation model run ID to classify"
    )
    parser.add_argument(
        "--model",
        default="decision_tree_model.pkl",
        help="Path to the pretrained decision tree model file"
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://traindata:traindata@localhost:5433/traindata",
        help="Database connection URL"
    )

    args = parser.parse_args()

    asyncio.run(
        classify_segments(
            args.seg_model_run_id,
            args.model,
            args.db_url
        )
    )


if __name__ == "__main__":
    main()