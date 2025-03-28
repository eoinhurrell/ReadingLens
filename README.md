# ReadingLens

ReadingLens is a personal article recommender system that learns from your highlighted content to predict how well you'd like new articles. The system analyzes markdown files containing article highlights, trains a model, and rates new articles from their URLs.

## Features

- **Data Processing**: Extracts highlighted sections from markdown files as positive signals
- **Hybrid Model**: Combines BERT for text encoding with DLRM-inspired feature interactions
- **Trained on Your Preferences**: Learns what content you find interesting from your highlights
- **Article Rating**: Gives an overall score to new articles based on the model's predictions
- **Highlight Suggestions**: Identifies sections of new articles you're likely to find valuable

## System Architecture

The system consists of four main components:

1. **Data Processor** (`data_processor.py`): Processes markdown files containing highlights to prepare training data.
2. **Model** (`model.py`): Implements a hybrid BERT+DLRM architecture for article recommendation.
3. **Training** (`train.py`): Trains the model and evaluates performance.
4. **Prediction** (`predict.py`): Rates new articles and suggests highlights.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/readinglens.git
cd readinglens

# Create and activate virtual environment
python -m pip install uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package and dependencies
uv pip install -e ".[dev]"
```

## Usage

### Data Format

Place your markdown files (one per article) in a directory. Each file should have a title and highlighted sections marked with `##` headings.

### Training the Model

Train the model directly on your markdown files:

```bash
python train.py --input_dir ./markdown_files --output_dir ./model --epochs 3
```

This processes the markdown files and trains the model in a single step, avoiding serialization issues.

### Rating New Articles

Rate a new article and get suggested highlights:

```bash
python predict.py --url https://example.com/article --model_path ./model/best_model.pt
```

### Optional Arguments

- For training: `--learning_rate`, `--batch_size`, `--use_cuda`, etc.
- For prediction: `--top_k` (number of highlights to suggest), `--output_file` (save results)

## Example Data Format

```markdown
# Article Title: Some Interesting Topic

Regular article text that wasn't highlighted...

## This is a highlighted section
Content that I found interesting enough to highlight.

More regular text...

## Another highlighted section
More content I found valuable.
```

## License

MIT
