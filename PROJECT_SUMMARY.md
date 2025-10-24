# Project Summary: Transparent Climate Communication Analysis

## Overview

This project is a comprehensive Python-based system for analyzing climate-related text using AI-powered prompt evaluation. The system processes text segments against 80+ climate communication criteria to classify different aspects of climate discourse, with support for machine learning model training and database integration.

## Architecture Overview

The project follows a pipeline architecture:
1. **Text Input** → **Feature Extraction** → **Model Training** → **Classification**
2. **Database Integration** for large-scale processing and result storage
3. **Multiple ML Models** (Decision Tree, SVM) for classification tasks

## Core Scripts

### 1. `run_prompts.py` - Standalone Feature Extraction

**Purpose**: Main application for processing individual text segments with LLM prompts.

**Key Features**:
- Asynchronous processing using OpenRouter API with Llama 4 Scout model
- Parallel prompt evaluation with configurable concurrency (default: 5 parallel requests)
- Converts LLM responses to boolean feature vectors
- Comprehensive error handling and rate limiting

**Usage**:
```bash
# Basic usage
python run_prompts.py --text "Your German text segment here"

# With custom parameters
python run_prompts.py --text "Your text" --max-parallel 10 --output vector

# Output formats: 'vector' (numpy array) or 'detailed' (full results)
python run_prompts.py --text "Your text" --output detailed
```

**Input/Output**:
- **Input**: German text segment
- **Output**: Feature vector (80+ boolean values) or detailed prompt results
- **API**: OpenRouter with meta-llama/llama-4-scout model

### 2. `extract_features_db.py` - Database-Integrated Feature Extraction

**Purpose**: Large-scale feature extraction from PostgreSQL database with cost tracking.

**Key Features**:
- Processes text segments from database tables (`generated_segments` or `segments`)
- Saves feature vectors to `feature_vectors` table
- Tracks API costs and generation IDs
- Creates model run records for experiment tracking
- Supports both generated segments and ground truth data

**Usage**:
```bash
# Process from segmentation model run
python extract_features_db.py --seg-model-run-id 123 --max-parallel 5

# Process ground truth segments
python extract_features_db.py --use-segments --limit 1000

# With custom database and description
python extract_features_db.py --seg-model-run-id 123 --db-url postgresql://... --description "Test run"
```

**Database Schema Integration**:
- Reads from: `generated_segments`, `segments`, `model_runs`
- Writes to: `feature_vectors`, `model_runs`
- Tracks: costs, generation IDs, ground truth labels

### 3. `train_decision_tree.py` - Decision Tree Classifier Training

**Purpose**: Train decision tree models on extracted feature vectors.

**Key Features**:
- Fetches training data from `feature_vectors` table
- Implements train/test split with stratification
- Comprehensive evaluation metrics (accuracy, F1-score, confusion matrix)
- Feature importance analysis
- Model persistence with pickle

**Usage**:
```bash
# Basic training
python train_decision_tree.py --feature-extraction-model-run-id 456

# With hyperparameter tuning
python train_decision_tree.py --feature-extraction-model-run-id 456 \
  --max-depth 10 --min-samples-split 5 --output my_model.pkl
```

**Model Configuration**:
- Algorithm: scikit-learn DecisionTreeClassifier
- Split: 80% train, 20% test
- Evaluation: Classification report, confusion matrix, feature importance
- Output: Pickle file with model and metadata

### 4. `train_svm.py` - Support Vector Machine Training

**Purpose**: Train SVM models with advanced hyperparameter support.

**Key Features**:
- Feature scaling with StandardScaler (crucial for SVM performance)
- Multiple kernel support (linear, poly, rbf, sigmoid)
- Hyperparameter tuning (C, gamma, kernel)
- Probability estimation enabled
- Model persistence with both classifier and scaler

**Usage**:
```bash
# Basic SVM training
python train_svm.py --feature-extraction-model-run-id 456

# With hyperparameters
python train_svm.py --feature-extraction-model-run-id 456 \
  --C 10.0 --kernel rbf --gamma 0.001 --output svm_model.pkl
```

**SVM-Specific Features**:
- **Feature Scaling**: Automatic StandardScaler application
- **Kernel Types**: linear, polynomial, RBF, sigmoid
- **Hyperparameters**: C (regularization), gamma (kernel coefficient)
- **Output**: Both trained classifier and fitted scaler

### 5. `test_decision_tree.py` - Decision Tree Testing

**Purpose**: Validation script for decision tree implementation.

**Key Features**:
- Syntax validation using AST parsing
- Import availability testing
- Argument parsing validation
- Comprehensive test reporting

**Test Coverage**:
- Script syntax correctness
- Required dependencies availability
- CLI argument structure
- Error handling for missing scikit-learn

### 6. `test_svm.py` - SVM Testing

**Purpose**: Validation script for SVM implementation.

**Key Features**:
- All decision tree tests plus SVM-specific validations
- Gamma parameter validation logic testing
- Kernel choice validation
- Feature scaling verification

**SVM-Specific Tests**:
- Gamma parameter handling ('scale', 'auto', float values)
- Kernel type validation
- Feature scaling implementation check

## Configuration Files

### `prompts.json` - Climate Communication Evaluation Prompts

**Purpose**: Contains 80+ German-language prompts for climate discourse analysis.

**Structure**: Array of prompt objects with `id` and `prompt` fields.

**Categories**:
- **Threat Assessment**: humans, nature, Austria/Europe, species
- **Temporal Dimensions**: past damages, current dangers, future threats
- **Spatial Scope**: local (Austria), European, global
- **Causality**: climate change, human activities, causal links
- **Impact Analysis**: negative consequences, intensity, quality of life
- **Climate Manifestations**: extreme weather, temperature, water scarcity, floods, sea level
- **Speech Acts**: warnings, threats, admonitions
- **Solutions**: actors, timeframes, spatial scope, reasoning, actions
- **Emotional Dimensions**: fear, hope, optimism, urgency
- **Action Levels**: individual, collective, systemic, technological
- **Communication Styles**: scientific, persuasive, imperative
- **Target Audiences**: specific groups, general public, decision makers
- **Sectoral Focus**: mobility, energy, agriculture, buildings, consumption, nature

**Example Prompt**:
```json
{
  "id": "threat_humans",
  "prompt": "Werden in dem gegebenen Textsegment Menschen als bedroht oder gefährdet dargestellt? Antworte nur mit 'true' oder 'false'.\n\nText: {text}"
}
```

### `pyproject.toml` - Python Project Configuration

**Purpose**: Standard Python project configuration using uv/pip.

**Dependencies**:
- `numpy`: Feature vector operations
- `aiohttp`: Async HTTP client for API calls
- `asyncpg`: PostgreSQL async database driver
- `scikit-learn`: Machine learning algorithms

**Requirements**: Python >=3.12

### `drizzle_schema.ts` - Database Schema

**Purpose**: TypeScript schema definition for PostgreSQL database.

**Key Tables**:
- `sentences`: Original text data with labels
- `model_runs`: Experiment tracking and metadata
- `model_predictions`: Model output storage
- `feature_vectors`: Extracted feature data (inferred from usage)

## Data Flow

### 1. Feature Extraction Pipeline
```
Text Input → OpenRouter API → LLM Processing → Boolean Responses → Feature Vector
```

### 2. Database Integration Pipeline
```
Database Segments → Batch Processing → Feature Vectors → Database Storage
```

### 3. Machine Learning Pipeline
```
Feature Vectors → Train/Test Split → Model Training → Evaluation → Model Persistence
```

## API Integration

**Provider**: OpenRouter (https://openrouter.ai)
**Model**: meta-llama/llama-4-scout
**Configuration**:
- Max tokens: 10 (optimized for true/false responses)
- Temperature: 0 (deterministic responses)
- Cost tracking: Both usage-based estimation and actual generation costs

**Authentication**: Requires `OPENROUTER_API_KEY` environment variable

## Database Schema

**Core Tables**:
- `model_runs`: Experiment metadata and tracking
- `feature_vectors`: Extracted features with prompt mappings
- `generated_segments` / `segments`: Source text data
- `sentences`: Labeled sentence data
- `model_predictions`: Classification results

**Relationships**:
- Feature vectors link to model runs and segments
- Model predictions reference sentences and model runs
- Foreign key constraints ensure data integrity

## Environment Setup

### Required Environment Variables
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### Installation
```bash
# Setup environment
uv sync

# Alternative with pip
pip install numpy aiohttp asyncpg scikit-learn
```

### Database Setup
Default connection: `postgresql://traindata:traindata@localhost:5433/traindata`

## Usage Patterns

### 1. Single Text Analysis
```bash
python run_prompts.py --text "German climate text here"
```

### 2. Batch Processing from Database
```bash
python extract_features_db.py --seg-model-run-id 123 --max-parallel 10
```

### 3. Model Training
```bash
# Decision Tree
python train_decision_tree.py --feature-extraction-model-run-id 456

# SVM
python train_svm.py --feature-extraction-model-run-id 456 --kernel rbf --C 10.0
```

### 4. Testing and Validation
```bash
python test_decision_tree.py
python test_svm.py
```

## Output Formats

### Feature Vectors
- **Format**: NumPy arrays of 0s and 1s
- **Length**: 80+ elements (one per prompt in prompts.json)
- **Ordering**: Alphabetically sorted by prompt_id
- **Values**: 1 for true responses, 0 for false/failed responses

### Model Files
- **Format**: Pickle files (.pkl)
- **Contents**: Trained model, metadata, labels, timestamps
- **SVM Special**: Includes both classifier and scaler objects

### Database Records
- **Feature vectors**: JSON arrays with prompt ID mappings
- **Model runs**: Metadata, costs, timestamps, experiment tracking
- **Predictions**: Classification results with confidence scores

## Error Handling

### API Resilience
- Rate limiting with semaphore-based concurrency control
- Comprehensive error handling for API failures
- Fallback cost estimation when generation data unavailable

### Database Robustness
- Connection management with proper cleanup
- Transaction handling for batch operations
- Conflict resolution with ON CONFLICT clauses

### Dependency Management
- Graceful handling of missing scikit-learn
- Clear error messages for missing API keys
- Import error handling with helpful suggestions

## Performance Characteristics

### Concurrency
- Default: 5 parallel API requests
- Configurable via `--max-parallel` parameter
- Semaphore-based rate limiting prevents API overload

### Cost Optimization
- 10 token limit for binary responses
- Temperature 0 for deterministic results
- Usage tracking for budget management

### Scalability
- Async/await for non-blocking operations
- Database batch processing
- Resumable processing with start-id support

## Project Philosophy

This is a **scientific research project** focused on:
- **Simplicity and clarity** over production complexity
- **Transparent feature extraction** using interpretable prompts
- **Reproducible experiments** through comprehensive metadata tracking
- **Flexible model comparison** with multiple ML algorithms

The system prioritizes **research utility** and **interpretability** over deployment optimization, making it ideal for academic climate communication research and analysis.