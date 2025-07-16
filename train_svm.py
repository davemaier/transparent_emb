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
        feature_vector,
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
        feature_vector = json.loads(row['feature_vector'])
        ground_truth = row['ground_truth']
        
        X.append(feature_vector)
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


def train_svm(
    X: np.ndarray, 
    y: np.ndarray, 
    C: float = 1.0,
    kernel: str = 'rbf',
    gamma: str = 'scale',
    random_state: int = 42
) -> Any:
    """
    Train an SVM classifier
    
    Args:
        X: Feature matrix
        y: Ground truth labels
        C: Regularization parameter
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        gamma: Kernel coefficient ('scale', 'auto', or float)
        random_state: Random state for reproducibility
    
    Returns:
        Trained SVM classifier
    """
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn is required for SVM training. Please install it with: pip install scikit-learn")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features for SVM (important for SVM performance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train SVM
    svm_classifier = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=random_state,
        probability=True  # Enable probability estimates
    )
    
    print(f"Training SVM classifier (C={C}, kernel={kernel}, gamma={gamma})...")
    svm_classifier.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = svm_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(list(set(y)))
    print(f"Labels: {labels}")
    print(cm)
    
    # SVM-specific information
    print(f"\nSVM Information:")
    print(f"Number of support vectors: {svm_classifier.n_support_}")
    print(f"Support vectors per class: {dict(zip(labels, svm_classifier.n_support_))}")
    
    # Return both the classifier and the scaler (needed for predictions)
    return {'classifier': svm_classifier, 'scaler': scaler}


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
        'scaler': model_data['scaler'],
        'feature_extraction_model_run_id': feature_extraction_model_run_id,
        'labels': labels,
        'model_info': model_info,
        'training_timestamp': datetime.now().isoformat(),
        'model_type': 'svm'
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(full_model_data, f)
    
    print(f"Model saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Train an SVM classifier on ground truth data from feature vectors"
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
        default="svm_model.pkl",
        help="Output path for the trained model"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter (default: 1.0)"
    )
    parser.add_argument(
        "--kernel",
        default="rbf",
        choices=["linear", "poly", "rbf", "sigmoid"],
        help="Kernel type (default: rbf)"
    )
    parser.add_argument(
        "--gamma",
        default="scale",
        help="Kernel coefficient ('scale', 'auto', or float value)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Handle gamma parameter
    if args.gamma not in ['scale', 'auto']:
        try:
            gamma = float(args.gamma)
        except ValueError:
            print(f"Error: gamma must be 'scale', 'auto', or a float value, got: {args.gamma}")
            return 1
    else:
        gamma = args.gamma
    
    # Connect to database
    conn = await get_db_connection(args.db_url)
    
    try:
        # Get model run information
        model_info = await get_model_run_info(conn, args.feature_extraction_model_run_id)
        print(f"Training on model run: {model_info['name']}")
        print(f"Description: {model_info['description']}")
        
        # Fetch training data
        X, y, labels = await fetch_training_data(conn, args.feature_extraction_model_run_id)
        
        # Train SVM
        model_data = train_svm(
            X, y,
            C=args.C,
            kernel=args.kernel,
            gamma=gamma,
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