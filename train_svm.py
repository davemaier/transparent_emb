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
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale',
    random_state: int = 42
) -> Tuple[Any, Any]:
    """
    Train an SVM classifier with feature scaling
    
    Args:
        X: Feature matrix
        y: Ground truth labels
        kernel: Kernel type for SVM ('linear', 'poly', 'rbf', 'sigmoid')
        C: Regularization parameter
        gamma: Kernel coefficient ('scale', 'auto', or float)
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (trained SVM classifier, fitted scaler)
    """
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
    except ImportError:
        raise ImportError("scikit-learn is required for SVM training. Please install it with: pip install scikit-learn")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Feature scaling (important for SVM)
    print("Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train SVM classifier
    svm_classifier = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=random_state
    )
    
    print(f"Training SVM classifier with kernel={kernel}, C={C}, gamma={gamma}...")
    svm_classifier.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = svm_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Calculate F1 scores with different averaging methods
    print("\nF1 Scores:")
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(list(set(y)))
    print(f"Labels: {labels}")
    print(cm)
    
    # SVM-specific information
    print(f"\nSVM Model Information:")
    print(f"Number of support vectors: {svm_classifier.n_support_}")
    print(f"Total support vectors: {svm_classifier.support_vectors_.shape[0]}")
    
    return svm_classifier, scaler


def save_model(
    model: Any,
    scaler: Any,
    model_info: Dict[str, Any],
    feature_extraction_model_run_id: int,
    labels: List[str],
    output_path: str
) -> None:
    """Save the trained model, scaler, and metadata"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_extraction_model_run_id': feature_extraction_model_run_id,
        'labels': labels,
        'model_info': model_info,
        'training_timestamp': datetime.now().isoformat(),
        'model_type': 'svm'
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model and scaler saved to: {output_path}")


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
        "--kernel",
        default="rbf",
        choices=["linear", "poly", "rbf", "sigmoid"],
        help="Kernel type for SVM"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter"
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
            args.gamma = float(args.gamma)
        except ValueError:
            raise ValueError("Gamma must be 'scale', 'auto', or a float value")
    
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
        svm_classifier, scaler = train_svm(
            X, y,
            kernel=args.kernel,
            C=args.C,
            gamma=args.gamma,
            random_state=args.random_state
        )
        
        # Save model
        save_model(
            svm_classifier,
            scaler,
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