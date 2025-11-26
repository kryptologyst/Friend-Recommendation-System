# Friend Recommendation System

A comprehensive friend recommendation system that combines multiple approaches including content-based filtering, collaborative filtering, and graph-based methods.

## Features

- **Multiple Recommendation Models**: Content-based, collaborative filtering, graph-based, and hybrid approaches
- **Comprehensive Evaluation**: Precision@K, Recall@K, MAP@K, NDCG@K, HitRate, Coverage, Novelty, Diversity metrics
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations
- **Production Ready**: Clean code structure, type hints, comprehensive testing, and CI/CD

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Friend-Recommendation-System.git
cd Friend-Recommendation-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Sample Data

```bash
python scripts/generate_data.py
```

### Train Models

```bash
python scripts/train_models.py
```

### Run Evaluation

```bash
python scripts/evaluate_models.py
```

### Launch Demo

```bash
streamlit run scripts/demo.py
```

## Project Structure

```
├── src/                    # Source code modules
│   ├── models/            # Recommendation models
│   ├── data/              # Data processing utilities
│   └── utils/              # General utilities
├── data/                   # Data directory
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── configs/               # Configuration files
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/               # Executable scripts
├── tests/                 # Unit tests
├── assets/                # Static assets (images, etc.)
└── requirements.txt       # Python dependencies
```

## Dataset Schema

### users.csv
- `user_id`: Unique user identifier
- `name`: User name
- `interests`: Comma-separated list of interests/hobbies
- `age`: User age
- `location`: User location
- `occupation`: User occupation

### interactions.csv
- `user_id`: User identifier
- `friend_id`: Friend identifier
- `timestamp`: Interaction timestamp
- `interaction_type`: Type of interaction (friend_request, message, etc.)
- `weight`: Interaction weight/strength

### items.csv (for content-based features)
- `item_id`: Item identifier (interests, activities, etc.)
- `name`: Item name
- `category`: Item category
- `description`: Item description

## Models

### Content-Based Filtering
- TF-IDF vectorization of user interests
- Cosine similarity for friend recommendations
- Sentence-BERT embeddings for semantic similarity

### Collaborative Filtering
- Matrix factorization using ALS
- BPR (Bayesian Personalized Ranking)
- User-based and item-based collaborative filtering

### Graph-Based Methods
- Social network analysis using NetworkX
- Graph neural networks for friend recommendations
- Community detection for group recommendations

### Hybrid Approaches
- Weighted combination of multiple models
- Stacking and ensemble methods
- Context-aware recommendations

## Evaluation Metrics

- **Ranking Metrics**: Precision@K, Recall@K, MAP@K, NDCG@K, HitRate
- **Coverage**: Percentage of users/items that can be recommended
- **Novelty**: Average popularity of recommended items
- **Diversity**: Intra-list diversity of recommendations
- **Popularity Bias**: Distribution of recommendation popularity

## Configuration

Models and experiments can be configured using YAML files in the `configs/` directory. See `configs/default.yaml` for available options.

## Development

### Code Quality

```bash
# Format code
black src/ scripts/ tests/

# Lint code
ruff check src/ scripts/ tests/

# Type checking
mypy src/
```

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# Friend-Recommendation-System
