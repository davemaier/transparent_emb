# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for analyzing climate-related text using AI-powered prompt evaluation. The project processes text segments against a comprehensive set of climate communication criteria to classify different aspects of climate discourse.

## Core Architecture

- **run_prompts.py**: Main application logic for parallel prompt processing
- **prompts.json**: Configuration file containing 80+ climate-related evaluation prompts in German
- **pyproject.toml**: Standard Python project configuration

## Key Components

### Prompt Processing System
The system uses asynchronous processing to evaluate text segments against multiple climate communication criteria:

1. **API Integration**: Uses OpenRouter API with Llama 4 Scout model
2. **Concurrency Control**: Implements semaphore-based rate limiting for parallel requests
3. **Error Handling**: Comprehensive error handling with detailed logging
4. **Response Processing**: Converts API responses to boolean results

### Prompt Categories
The prompts.json file contains evaluation criteria organized around:
- **Threat Assessment**: Humans, nature, Austria/Europe, species
- **Temporal Dimensions**: Past damages, current dangers, future threats
- **Spatial Scope**: Local (Austria), European, global
- **Causality**: Climate change, human activities, causal links
- **Impact Analysis**: Negative consequences, intensity, quality of life
- **Manifestations**: Extreme weather, temperature, water scarcity, floods, sea level
- **Speech Acts**: Warnings, threats, admonitions
- **Solutions**: Actors, timeframes, spatial scope, reasoning, actions
- **Emotional Dimensions**: Fear, hope, optimism, urgency
- **Action Levels**: Individual, collective, systemic, technological
- **Communication Styles**: Scientific, persuasive, imperative
- **Target Audiences**: Specific groups, general public, decision makers
- **Sectoral Focus**: Mobility, energy, agriculture, buildings, consumption, nature

## Development Commands

### Environment Setup
```bash
# Set up environment with uv
uv sync

# Set required environment variable
export OPENROUTER_API_KEY="your_api_key_here"
```

### Running the Application
```bash
# Basic usage
python run_prompts.py --text "Your text segment here"

# With custom prompts file
python run_prompts.py --prompts custom_prompts.json --text "Your text"

# Control concurrency
python run_prompts.py --text "Your text" --max-parallel 10

# Output feature vector (default)
python run_prompts.py --text "Your text" --output vector

# Output detailed results (original format)
python run_prompts.py --text "Your text" --output detailed
```

## Configuration

### Required Environment Variables
- `OPENROUTER_API_KEY`: API key for OpenRouter service

### API Configuration
- Model: `meta-llama/llama-4-scout`
- Max tokens: 10 (optimized for true/false responses)
- Temperature: 0 (deterministic responses)
- Default concurrency: 5 parallel requests

## Usage Notes

- All prompts expect German text input
- Responses are automatically normalized to boolean values
- The system is designed for climate communication analysis
- Rate limiting prevents API overload
- Error handling provides detailed feedback for debugging

## Output Format

The system outputs feature vectors (numpy arrays) suitable for machine learning:
- **Feature Vector**: 1D numpy array of 0s and 1s
- **Length**: 80+ elements (one per prompt in prompts.json)
- **Ordering**: Alphabetically sorted by prompt_id for consistency
- **Values**: 1 for true responses, 0 for false responses or failed requests

## Dependencies

- `numpy`: For feature vector creation
- `aiohttp`: Async HTTP client for API calls
- `asyncio`: Async programming support
- Standard library: `json`, `argparse`, `os`, `typing`

# Instructions for Developers

- This is a scientific project and focus is on simplicity and clarity. The code does not need to be production-ready and tests are not required.
- Do not create extensive documentation or detailed comments.
- Use the available postgres mcp to get information on datastructure and data.
- If changes to the database schema are required, these need to be done in the drizzle_schema.ts file
