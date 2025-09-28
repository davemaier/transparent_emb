#!/usr/bin/env python3

import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import psycopg2
from pathlib import Path
from typing import Optional, List, Dict, Any


def load_model(model_path: str) -> Dict[str, Any]:
    """Load the saved decision tree model"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    if model_data.get('model_type') != 'decision_tree':
        raise ValueError(f"Expected decision tree model, got: {model_data.get('model_type')}")

    return model_data


def load_feature_labels_from_db(feature_extraction_model_run_id: int, connection_string: Optional[str] = None) -> List[str]:
    """Load feature labels from database for the given feature extraction model run ID"""
    if not connection_string:
        connection_string = "postgresql://traindata:traindata@localhost:5433/traindata"

    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()

        # Get feature labels from any feature vector with this feature_extraction_model_run_id
        cursor.execute(
            "SELECT prompts FROM feature_vectors WHERE feature_extraction_model_run_id = %s LIMIT 1",
            (feature_extraction_model_run_id,)
        )

        result = cursor.fetchone()
        if not result:
            raise ValueError(f"No feature vectors found for feature_extraction_model_run_id: {feature_extraction_model_run_id}")

        feature_labels = result[0]
        cursor.close()
        conn.close()

        return feature_labels

    except Exception as e:
        raise ValueError(f"Error loading feature labels from database: {e}")


def visualize_tree_structure(model_data: Dict[str, Any], feature_labels: Optional[List[str]] = None, max_depth: Optional[int] = None, output_path: Optional[str] = None) -> None:
    """Create a visual representation of the decision tree structure"""
    try:
        from sklearn.tree import plot_tree
    except ImportError:
        raise ImportError("scikit-learn is required for tree visualization")

    dt_classifier = model_data['model']
    labels = model_data['labels']

    # Use feature labels if provided, otherwise fall back to generic names
    if feature_labels and len(feature_labels) == dt_classifier.n_features_in_:
        feature_names = feature_labels
    else:
        feature_names = [f"Feature_{i}" for i in range(dt_classifier.n_features_in_)]

    # Adjust figure size and font based on depth limitation
    if max_depth and max_depth <= 5:
        figsize = (15, 10)
        fontsize = 10
    elif max_depth and max_depth <= 10:
        figsize = (20, 12)
        fontsize = 9
    else:
        figsize = (25, 15)
        fontsize = 8

    # Create figure with appropriate size
    plt.figure(figsize=figsize)

    # Plot the tree with depth limitation
    plot_tree(
        dt_classifier,
        filled=True,
        feature_names=feature_names,
        class_names=[str(label) for label in labels],
        rounded=True,
        fontsize=fontsize,
        max_depth=max_depth
    )

    depth_info = f" (max depth: {max_depth})" if max_depth else " (full tree)"
    plt.title(f"Decision Tree Structure{depth_info}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Tree structure saved to: {output_path}")
    else:
        plt.show()


def visualize_feature_importance(model_data: Dict[str, Any], feature_labels: Optional[List[str]] = None, top_n: int = 10, output_path: Optional[str] = None) -> None:
    """Create a bar chart of feature importance"""
    dt_classifier = model_data['model']
    feature_importance = dt_classifier.feature_importances_

    # Get top N features
    feature_indices = np.argsort(feature_importance)[::-1][:top_n]
    top_importance = feature_importance[feature_indices]

    # Use feature labels if provided, otherwise fall back to generic names
    if feature_labels and len(feature_labels) == len(feature_importance):
        feature_names = [feature_labels[idx] for idx in feature_indices]
    else:
        feature_names = [f"F{idx}" for idx in feature_indices]

    # Adjust figure size based on number of features
    if top_n <= 10:
        figsize = (12, 8)
        fontsize = 10
    elif top_n <= 15:
        figsize = (14, 9)
        fontsize = 9
    else:
        figsize = (15, 10)
        fontsize = 9

    # Create bar chart
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(top_importance)), top_importance)

    # Color bars by importance
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_importance)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.xticks(range(len(top_importance)), feature_names, rotation=45, ha='right')

    # Add value labels on bars
    for i, importance in enumerate(top_importance):
        plt.text(i, importance + 0.001, f'{importance:.3f}',
                ha='center', va='bottom', fontsize=fontsize)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {output_path}")
    else:
        plt.show()


def visualize_tree_depth_stats(model_data: Dict[str, Any], output_path: Optional[str] = None) -> None:
    """Create visualizations showing tree depth and node statistics"""
    dt_classifier = model_data['model']

    # Get tree structure information
    tree = dt_classifier.tree_

    # Calculate depth of each leaf
    def get_leaf_depths(node_id=0, depth=0):
        depths = []
        if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf node
            depths.append(depth)
        else:
            depths.extend(get_leaf_depths(tree.children_left[node_id], depth + 1))
            depths.extend(get_leaf_depths(tree.children_right[node_id], depth + 1))
        return depths

    leaf_depths = get_leaf_depths()

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Histogram of leaf depths
    axes[0, 0].hist(leaf_depths, bins=range(max(leaf_depths) + 2), alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Leaf Depths')
    axes[0, 0].set_xlabel('Depth')
    axes[0, 0].set_ylabel('Number of Leaves')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. Tree statistics
    stats_text = f"""Tree Statistics:

Max Depth: {dt_classifier.get_depth()}
Number of Leaves: {dt_classifier.get_n_leaves()}
Total Nodes: {tree.node_count}
Number of Features: {dt_classifier.n_features_in_}
Number of Classes: {dt_classifier.n_classes_}

Average Leaf Depth: {np.mean(leaf_depths):.2f}
Min Leaf Depth: {min(leaf_depths)}
Max Leaf Depth: {max(leaf_depths)}"""

    axes[0, 1].text(0.1, 0.9, stats_text, transform=axes[0, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Tree Statistics')

    # 3. Node impurity distribution
    node_impurities = tree.impurity[tree.impurity >= 0]  # Filter out invalid values
    axes[1, 0].hist(node_impurities, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('Distribution of Node Impurities')
    axes[1, 0].set_xlabel('Impurity (Gini)')
    axes[1, 0].set_ylabel('Number of Nodes')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Samples per node distribution
    node_samples = tree.n_node_samples
    axes[1, 1].hist(node_samples, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title('Distribution of Samples per Node')
    axes[1, 1].set_xlabel('Number of Samples')
    axes[1, 1].set_ylabel('Number of Nodes')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Tree statistics plot saved to: {output_path}")
    else:
        plt.show()


def print_model_info(model_data: Dict[str, Any]) -> None:
    """Print information about the loaded model"""
    dt_classifier = model_data['model']
    model_info = model_data.get('model_info', {})

    print("="*50)
    print("DECISION TREE MODEL INFORMATION")
    print("="*50)

    if model_info:
        print(f"Model Name: {model_info.get('name', 'Unknown')}")
        print(f"Description: {model_info.get('description', 'None')}")

    print(f"Training Timestamp: {model_data.get('training_timestamp', 'Unknown')}")
    print(f"Feature Extraction Model Run ID: {model_data.get('feature_extraction_model_run_id', 'Unknown')}")
    print(f"Labels: {model_data.get('labels', [])}")

    print(f"\nModel Parameters:")
    print(f"  Max Depth: {dt_classifier.get_depth()}")
    print(f"  Number of Leaves: {dt_classifier.get_n_leaves()}")
    print(f"  Number of Features: {dt_classifier.n_features_in_}")
    print(f"  Number of Classes: {dt_classifier.n_classes_}")

    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained decision tree classifier"
    )
    parser.add_argument(
        "--model",
        default="decision_tree_model.pkl",
        help="Path to the saved decision tree model"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save visualization plots (if not specified, plots will be displayed)"
    )
    parser.add_argument(
        "--tree-structure",
        action="store_true",
        help="Generate tree structure visualization"
    )
    parser.add_argument(
        "--feature-importance",
        action="store_true",
        help="Generate feature importance plot"
    )
    parser.add_argument(
        "--tree-stats",
        action="store_true",
        help="Generate tree statistics plots"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all visualizations"
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=10,
        help="Number of top features to show in importance plot"
    )
    parser.add_argument(
        "--db-connection",
        help="Database connection string (uses DATABASE_URL env var if not provided)"
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        help="Maximum depth to display in tree structure (default: show full tree)"
    )

    args = parser.parse_args()

    # If no specific visualization is requested, show all
    if not any([args.tree_structure, args.feature_importance, args.tree_stats, args.all]):
        args.all = True

    if args.all:
        args.tree_structure = True
        args.feature_importance = True
        args.tree_stats = True

    # Load model
    try:
        model_data = load_model(args.model)
        print_model_info(model_data)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Try to load feature labels from database
    feature_labels = None
    feature_extraction_model_run_id = model_data.get('feature_extraction_model_run_id')

    if feature_extraction_model_run_id:
        try:
            feature_labels = load_feature_labels_from_db(feature_extraction_model_run_id, args.db_connection)
            print(f"Loaded {len(feature_labels)} feature labels from database")
        except Exception as e:
            print(f"Warning: Could not load feature labels from database: {e}")
            print("Falling back to generic feature names")
    else:
        print("Warning: No feature_extraction_model_run_id found in model data")

    # Create output directory if specified
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

    try:
        # Generate visualizations
        if args.tree_structure:
            output_path = None
            if output_dir:
                depth_suffix = f"_depth{args.tree_max_depth}" if args.tree_max_depth else "_full"
                output_path = output_dir / f"decision_tree_structure{depth_suffix}.png"
            visualize_tree_structure(model_data, feature_labels, args.tree_max_depth, output_path)

        if args.feature_importance:
            output_path = None
            if output_dir:
                output_path = output_dir / "feature_importance.png"
            visualize_feature_importance(model_data, feature_labels, args.top_features, output_path)

        if args.tree_stats:
            output_path = None
            if output_dir:
                output_path = output_dir / "tree_statistics.png"
            visualize_tree_depth_stats(model_data, output_path)

        print("\nVisualization completed successfully!")

    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)