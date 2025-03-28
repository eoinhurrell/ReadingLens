#!/usr/bin/env python
import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model import ArticleRecommender


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    output_dir: str,
    early_stopping_patience: int = 3,
) -> Tuple[List[float], List[float]]:
    """
    Train the model and evaluate on validation set

    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on
        num_epochs: Number of epochs to train for
        output_dir: Directory to save model checkpoints
        early_stopping_patience: Number of epochs to wait for improvement before stopping

    Returns:
        Tuple of (train_losses, val_losses)
    """
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_progress = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
        )
        for batch in train_progress:
            # Get inputs from TensorDataset (which uses integer indexing)
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].float().to(device).view(-1, 1)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item() * input_ids.size(0)

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            # Update progress bar
            train_progress.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": train_correct / train_total if train_total > 0 else 0,
                }
            )

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader.dataset)
        train_losses.append(avg_train_loss)
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_progress = tqdm(
                val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"
            )
            for batch in val_progress:
                # Get inputs from TensorDataset (which uses integer indexing)
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].float().to(device).view(-1, 1)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Update metrics
                val_loss += loss.item() * input_ids.size(0)

                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # Update progress bar
                val_progress.set_postfix(
                    {
                        "loss": loss.item(),
                        "acc": val_correct / val_total if val_total > 0 else 0,
                    }
                )

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader.dataset)
        val_losses.append(avg_val_loss)
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(
            f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}"
            )
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save the model
            model.save(best_model_path)
            print(f"Model saved to {best_model_path}")
        else:
            patience_counter += 1
            patience_msg = "Validation loss did not improve. "
            patience_msg += f"Patience: {patience_counter}/{early_stopping_patience}"
            print(patience_msg)

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses


def plot_training_history(
    train_losses: List[float], val_losses: List[float], output_dir: str
):
    """
    Plot training and validation losses

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")


class MarkdownProcessor:
    def __init__(
        self, tokenizer_name: str = "bert-base-uncased", max_length: int = 512
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def parse_markdown_file(self, file_path: str) -> Dict:
        """
        Parse a markdown file containing article content with highlighted sections.

        Args:
            file_path: Path to the markdown file

        Returns:
            Dictionary with article metadata, full content, and highlighted sections
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract article title (assuming first line is title)
        title_match = re.search(r"# (.+)", content)
        title = title_match.group(1) if title_match else "Untitled"

        # Extract highlighted sections (## headings)
        highlights = re.findall(
            r"## (.+?)(?=\n\n|$)(.*?)(?=\n## |\n# |$)", content, re.DOTALL
        )
        highlighted_sections = []

        for heading, content in highlights:
            highlighted_sections.append(
                {"heading": heading.strip(), "content": content.strip()}
            )

        # Get full text without markdown formatting
        full_content = re.sub(r"#+ .*?\n", "", content)
        full_content = re.sub(r"\n+", " ", full_content).strip()

        # Get domain from filename if available
        domain = (
            os.path.basename(file_path).split("_")[0]
            if "_" in os.path.basename(file_path)
            else None
        )

        return {
            "title": title,
            "domain": domain,
            "full_content": full_content,
            "highlights": highlighted_sections,
            "file_path": file_path,
        }

    def create_training_examples(
        self, article_data: List[Dict], neg_to_pos_ratio: float = 2.0
    ) -> List[Dict]:
        """
        Create training examples from parsed article data.
        Positive examples come from highlighted sections, negative from non-highlighted.

        Args:
            article_data: List of dictionaries containing parsed article data
            neg_to_pos_ratio: Ratio of negative to positive examples

        Returns:
            List of training examples with features and labels
        """
        examples = []

        for article in tqdm(article_data, desc="Creating examples"):
            # Create positive examples from highlights
            for highlight in article["highlights"]:
                text = highlight["content"]
                if not text.strip():
                    continue

                # Create chunks if text is too long
                chunks = self._chunk_text(text)
                for chunk in chunks:
                    examples.append(
                        {
                            "text": chunk,
                            "label": 1,
                            "domain": article["domain"],
                            "article_title": article["title"],
                        }
                    )

            # Create negative examples from non-highlighted sections
            # Extract paragraphs that aren't in highlights
            full_text = article["full_content"]
            highlighted_text = " ".join([h["content"] for h in article["highlights"]])

            # Simple approach - split by paragraphs and filter out those in highlights
            paragraphs = re.split(r"\n\n+", full_text)
            negative_paragraphs = []

            for para in paragraphs:
                para = para.strip()
                if para and para not in highlighted_text and len(para.split()) > 10:
                    negative_paragraphs.append(para)

            # Sample negative examples based on ratio
            num_positive = len(article["highlights"])
            num_negative = min(
                int(num_positive * neg_to_pos_ratio), len(negative_paragraphs)
            )

            if negative_paragraphs and num_negative > 0:
                sampled_negatives = random.sample(negative_paragraphs, num_negative)

                for text in sampled_negatives:
                    chunks = self._chunk_text(text)
                    for chunk in chunks:
                        examples.append(
                            {
                                "text": chunk,
                                "label": 0,
                                "domain": article["domain"],
                                "article_title": article["title"],
                            }
                        )

        return examples

    def _chunk_text(self, text: str, overlap: int = 50) -> List[str]:
        """Split text into chunks that fit within max_length tokens"""
        if not text.strip():
            return []

        # Simple method - split by tokens directly
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= self.max_length - 2:  # Account for [CLS] and [SEP]
            return [text]

        chunks = []
        max_tokens = self.max_length - 2

        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)

            if i + max_tokens >= len(tokens):
                break

        return chunks

    def preprocess_for_bert(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Convert text examples to BERT input format

        Args:
            examples: List of dictionaries containing text and labels

        Returns:
            Dictionary of tensors for model input
        """
        texts = [example["text"] for example in examples]
        labels = [example["label"] for example in examples]

        # Tokenize all examples
        encoded_batch = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Add labels
        encoded_batch["labels"] = torch.tensor(labels)

        return encoded_batch

    def process_data(
        self, input_dir: str, train_ratio: float = 0.8, seed: int = 42
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[Dict], List[Dict]
    ]:
        """
        Process all markdown files in input_dir and return processed data directly
        without serialization/deserialization

        Args:
            input_dir: Directory containing markdown files
            train_ratio: Ratio of data to use for training vs validation
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_data, val_data, train_examples, val_examples)
        """
        random.seed(seed)
        np.random.seed(seed)

        # Get all markdown files
        markdown_files = list(Path(input_dir).glob("*.md"))
        print(f"Found {len(markdown_files)} markdown files")

        # Parse all files
        article_data = []
        for file_path in tqdm(markdown_files, desc="Parsing files"):
            try:
                article = self.parse_markdown_file(str(file_path))
                article_data.append(article)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Create examples
        examples = self.create_training_examples(article_data)
        print(f"Created {len(examples)} examples")

        # Split into train and validation
        random.shuffle(examples)
        split_idx = int(len(examples) * train_ratio)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        print(
            f"Train examples: {len(train_examples)}, Validation examples: {len(val_examples)}"
        )

        # Preprocess train and validation sets
        train_data = self.preprocess_for_bert(train_examples)
        val_data = self.preprocess_for_bert(val_examples)

        return train_data, val_data, train_examples, val_examples


def main():
    parser = argparse.ArgumentParser(description="Train article recommender model")
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing markdown files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save trained model"
    )
    parser.add_argument(
        "--bert_model", default="bert-base-uncased", help="BERT model to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training vs validation",
    )
    parser.add_argument(
        "--use_domains", action="store_true", help="Whether to use domain information"
    )
    parser.add_argument(
        "--num_domains", type=int, default=100, help="Number of unique domains"
    )
    parser.add_argument(
        "--freeze_bert", action="store_true", help="Whether to freeze BERT parameters"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_cuda",
        default=True,
        action="store_true",
        help="Whether to use CUDA if available",
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )
    print(f"Using device: {device}")

    # Process the data directly without serialization
    print("Processing markdown files and preparing datasets...")
    processor = MarkdownProcessor(
        tokenizer_name=args.bert_model, max_length=args.max_length
    )

    train_data, val_data, train_examples, val_examples = processor.process_data(
        input_dir=args.input_dir, train_ratio=args.train_ratio, seed=args.seed
    )

    # Save examples for reference (optional)
    with open(
        os.path.join(args.output_dir, "train_examples.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(train_examples, f, indent=2)

    with open(
        os.path.join(args.output_dir, "val_examples.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(val_examples, f, indent=2)

    # Create dataloaders
    train_dataset = TensorDataset(
        train_data["input_ids"], train_data["attention_mask"], train_data["labels"]
    )

    val_dataset = TensorDataset(
        val_data["input_ids"], val_data["attention_mask"], val_data["labels"]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    print(f"Initializing model with {args.bert_model}...")
    model = ArticleRecommender(
        bert_model_name=args.bert_model,
        embedding_dim=768,  # Default for BERT
        num_domains=args.num_domains,
        use_domains=args.use_domains,
        freeze_bert=args.freeze_bert,
    )

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    # Print model information
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    # Save arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        early_stopping_patience=args.patience,
    )

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot training history
    plot_training_history(train_losses, val_losses, args.output_dir)

    # Save training history
    history = {"train_losses": train_losses, "val_losses": val_losses}

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("Training completed!")


if __name__ == "__main__":
    main()
