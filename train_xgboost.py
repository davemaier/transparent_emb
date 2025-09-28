#!/usr/bin/env python3

import json
import asyncio
import asyncpg
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
from datetime import datetime


async def get_db_connection(db_url: str) -> asyncpg.Connection:
    """Create database connection"""
    return await asyncpg.connect(db_url)


async def fetch_training_data(
    conn: asyncpg.Connection,
    feature_extraction_model_run_id: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Fetch feature vectors and ground truth labels for training

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Ground truth labels (n_samples,)
        labels: List of unique labels
    """
    query = """
    SELECT
        vector,
        ground_truth
    FROM feature_vectors
    WHERE feature_extraction_model_run_id = $1
    AND ground_truth IS NOT NULL
    ORDER BY id
    """

    rows = await conn.fetch(query, feature_extraction_model_run_id)

    if not rows:
        raise ValueError(f"No training data found for feature extraction model run {feature_extraction_model_run_id}")

    print(f"Found {len(rows)} samples with ground truth labels")

    # Extract feature vectors and labels
    X = []
    y = []

    for row in rows:
        vector = row['vector']
        ground_truth = row['ground_truth']

        X.append(vector)
        y.append(ground_truth)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Get unique labels
    labels = list(set(y))
    labels.sort()  # Sort for consistency

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: {labels}")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, labels


async def get_model_run_info(conn: asyncpg.Connection, model_run_id: int) -> Dict[str, Any]:
    """Get information about the model run"""
    query = "SELECT name, description, metadata FROM model_runs WHERE id = $1"
    row = await conn.fetchrow(query, model_run_id)

    if not row:
        raise ValueError(f"Model run {model_run_id} not found")

    return {
        'name': row['name'],
        'description': row['description'],
        'metadata': json.loads(row['metadata']) if row['metadata'] else {}
    }


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    random_state: int = 42
) -> Any:
    """
    Train an XGBoost classifier

    Args:
        X: Feature matrix
        y: Ground truth labels
        n_estimators: Number of boosting rounds
        max_depth: Maximum depth of trees
        learning_rate: Boosting learning rate
        subsample: Subsample ratio of training instances
        colsample_bytree: Subsample ratio of columns when constructing each tree
        random_state: Random state for reproducibility

    Returns:
        Trained XGBoost classifier
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        raise ImportError("xgboost and scikit-learn are required for XGBoost training. Please install them with: pip install xgboost scikit-learn")

    # Encode labels for XGBoost (it requires numeric labels)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Create and train XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    print(f"Training XGBoost classifier (n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate})...")
    xgb_classifier.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_encoded = xgb_classifier.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test_original = label_encoder.inverse_transform(y_test)

    accuracy = accuracy_score(y_test_original, y_pred)
    total_f1 = f1_score(y_test_original, y_pred, average='macro')

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Total F1 Score (macro): {total_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_original, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_original, y_pred)
    labels = sorted(list(set(y)))
    print(f"Labels: {labels}")
    print(cm)

    # Feature importance
    feature_importance = xgb_classifier.feature_importances_
    print(f"\nTop 10 most important features:")
    feature_indices = np.argsort(feature_importance)[::-1][:10]
    for i, idx in enumerate(feature_indices):
        print(f"{i+1}. Feature {idx}: {feature_importance[idx]:.4f}")

    # Return both the classifier and the label encoder
    return {'classifier': xgb_classifier, 'label_encoder': label_encoder}


def save_model(
    model_data: Dict[str, Any],
    model_info: Dict[str, Any],
    feature_extraction_model_run_id: int,
    labels: List[str],
    output_path: str
) -> None:
    """Save the trained model and metadata"""
    full_model_data = {
        'classifier': model_data['classifier'],
        'label_encoder': model_data['label_encoder'],
        'feature_extraction_model_run_id': feature_extraction_model_run_id,
        'labels': labels,
        'model_info': model_info,
        'training_timestamp': datetime.now().isoformat(),
        'model_type': 'xgboost'
    }

    with open(output_path, 'wb') as f:
        pickle.dump(full_model_data, f)

    print(f"Model saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier on ground truth data from feature vectors"
    )
    parser.add_argument(
        "--feature-extraction-model-run-id",
        type=int,
        required=True,
        help="Feature extraction model run ID to train on"
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://traindata:traindata@localhost:5433/traindata",
        help="Database connection URL"
    )
    parser.add_argument(
        "--output",
        default="xgboost_model.pkl",
        help="Output path for the trained model"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting rounds (default: 100)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum depth of trees (default: 6)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Boosting learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Subsample ratio of training instances (default: 1.0)"
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="Subsample ratio of columns when constructing each tree (default: 1.0)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )

    args = parser.parse_args()

    # Connect to database
    conn = await get_db_connection(args.db_url)

    try:
        # Get model run information
        model_info = await get_model_run_info(conn, args.feature_extraction_model_run_id)
        print(f"Training on model run: {model_info['name']}")
        print(f"Description: {model_info['description']}")

        # Fetch training data
        X, y, labels = await fetch_training_data(conn, args.feature_extraction_model_run_id)

        # Train XGBoost
        model_data = train_xgboost(
            X, y,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=args.random_state
        )

        # Save model
        save_model(
            model_data,
            model_info,
            args.feature_extraction_model_run_id,
            labels,
            args.output
        )

        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    finally:
        await conn.close()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)