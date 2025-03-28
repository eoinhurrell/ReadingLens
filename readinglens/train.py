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
from data_processor import MarkdownProcessor


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
        train_rmse = 0.0  # Use RMSE instead of accuracy for regression
        train_total = 0

        train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
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

            # Calculate RMSE
            train_rmse += torch.sqrt(torch.mean((outputs - labels) ** 2)).item() * input_ids.size(0)
            train_total += input_ids.size(0)

            # Update progress bar
            train_progress.set_postfix(
                {
                    "loss": loss.item(),
                    "rmse": train_rmse / train_total if train_total > 0 else 0,
                }
            )

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader.dataset)
        train_losses.append(avg_train_loss)
        train_rmse = train_rmse / train_total if train_total > 0 else 0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_total = 0

        with torch.no_grad():
            val_progress = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
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

                # Calculate RMSE
                val_rmse += torch.sqrt(torch.mean((outputs - labels) ** 2)).item() * input_ids.size(
                    0
                )
                val_total += input_ids.size(0)

                # Update progress bar
                val_progress.set_postfix(
                    {
                        "loss": loss.item(),
                        "rmse": val_rmse / val_total if val_total > 0 else 0,
                    }
                )

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader.dataset)
        val_losses.append(avg_val_loss)
        val_rmse = val_rmse / val_total if val_total > 0 else 0

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}")

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}")
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


def plot_training_history(train_losses: List[float], val_losses: List[float], output_dir: str):
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


def generate_ratings_from_highlights(examples, highlight_stats, max_rating=5.0, scaling_factor=1.0):
    """
    Generate rating values for training examples based on highlight statistics

    Args:
        examples: List of training examples
        highlight_stats: Statistics about highlights in the dataset
        max_rating: Maximum rating value (e.g., 5.0)
        scaling_factor: Factor to adjust the distribution spread

    Returns:
        Updated examples with ratings
    """
    # Group examples by article title
    article_examples = {}
    for ex in examples:
        title = ex["article_title"]
        if title not in article_examples:
            article_examples[title] = []
        article_examples[title].append(ex)

    # Calculate ratings for each article
    for title, article_exs in article_examples.items():
        # Count positive examples (highlights)
        highlight_count = sum(1 for ex in article_exs if ex["label"] == 1)

        # Calculate rating based on z-score
        z_score = (highlight_count - highlight_stats["mean_count"]) / highlight_stats["std_count"]

        # Convert z-score to rating scale
        # Mid-point of rating scale + scaled z-score
        mid_rating = max_rating / 2
        rating = mid_rating + (z_score * scaling_factor)

        # Ensure rating is within bounds (1-max_rating)
        rating = max(1.0, min(max_rating, rating))

        # Also factor in the length of highlights
        highlight_lengths = [len(ex["text"].split()) for ex in article_exs if ex["label"] == 1]
        avg_highlight_length = sum(highlight_lengths) / max(1, len(highlight_lengths))

        # Adjust rating slightly based on average highlight length (longer = potentially more valuable)
        # This is a small adjustment to add some variability
        length_factor = 0.2  # Small influence factor
        if avg_highlight_length > 100:  # Longer highlights
            rating = min(max_rating, rating + length_factor)
        elif avg_highlight_length < 50:  # Very short highlights
            rating = max(1.0, rating - length_factor)

        # Update ratings for all examples from this article
        for ex in article_exs:
            # Use different rating values for highlights vs. non-highlights
            if ex["label"] == 1:
                # Highlights get the article rating
                ex["rating"] = rating
            else:
                # Non-highlights get lower ratings
                ex["rating"] = max(1.0, rating - 2.0)

    # Flatten the examples back into a list
    result = []
    for exs in article_examples.values():
        result.extend(exs)

    return result


def main():
    parser = argparse.ArgumentParser(description="Train article recommender model")
    parser.add_argument("--input_dir", required=True, help="Directory containing markdown files")
    parser.add_argument("--output_dir", required=True, help="Directory to save trained model")
    parser.add_argument("--bert_model", default="bert-base-uncased", help="BERT model to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training vs validation",
    )
    parser.add_argument(
        "--use_domains", action="store_true", help="Whether to use domain information"
    )
    parser.add_argument("--num_domains", type=int, default=100, help="Number of unique domains")
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
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_rating", type=float, default=5.0, help="Maximum rating value")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="Rating scaling factor")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # Process the data
    print("Processing markdown files and preparing datasets...")
    processor = MarkdownProcessor(tokenizer_name=args.bert_model, max_length=args.max_length)

    # Get all markdown files
    markdown_files = list(Path(args.input_dir).glob("*.md"))
    print(f"Found {len(markdown_files)} markdown files")

    # Parse all files
    article_data = []
    for file_path in tqdm(markdown_files, desc="Parsing files"):
        try:
            article = processor.parse_markdown_file(str(file_path))
            article_data.append(article)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Calculate highlight statistics
    highlight_counts = [processor.get_highlight_count(article) for article in article_data]

    # Calculate statistics
    mean_count = sum(highlight_counts) / len(highlight_counts) if highlight_counts else 3.0
    std_count = (
        (sum((c - mean_count) ** 2 for c in highlight_counts) / len(highlight_counts)) ** 0.5
        if highlight_counts and len(highlight_counts) > 1
        else 1.0  # Default to 1.0 to avoid division by zero
    )

    # Add stats to metadata
    highlight_stats = {
        "mean_count": mean_count,
        "std_count": std_count,
        "min_count": min(highlight_counts) if highlight_counts else 0,
        "max_count": max(highlight_counts) if highlight_counts else 0,
        "median_count": sorted(highlight_counts)[len(highlight_counts) // 2]
        if highlight_counts
        else 0,
    }

    # Save highlight statistics
    with open(os.path.join(args.output_dir, "highlight_stats.json"), "w", encoding="utf-8") as f:
        json.dump(highlight_stats, f, indent=2)

    print(f"Highlight statistics: mean={mean_count:.2f}, std={std_count:.2f}")

    # Create examples
    examples = processor.create_training_examples(article_data)
    print(f"Created {len(examples)} examples")

    # Generate ratings based on highlight statistics
    rated_examples = generate_ratings_from_highlights(
        examples, highlight_stats, max_rating=args.max_rating, scaling_factor=args.scaling_factor
    )

    # Split into train and validation
    random.shuffle(rated_examples)
    split_idx = int(len(rated_examples) * args.train_ratio)
    train_examples = rated_examples[:split_idx]
    val_examples = rated_examples[split_idx:]

    print(f"Train examples: {len(train_examples)}, Validation examples: {len(val_examples)}")

    # Save examples for reference
    with open(os.path.join(args.output_dir, "train_examples.json"), "w", encoding="utf-8") as f:
        json.dump(train_examples, f, indent=2)

    with open(os.path.join(args.output_dir, "val_examples.json"), "w", encoding="utf-8") as f:
        json.dump(val_examples, f, indent=2)

    # Prepare the datasets for training
    train_texts = [example["text"] for example in train_examples]
    train_ratings = [example["rating"] for example in train_examples]

    val_texts = [example["text"] for example in val_examples]
    val_ratings = [example["rating"] for example in val_examples]

    # Tokenize
    train_encodings = processor.tokenizer.batch_encode_plus(
        train_texts,
        add_special_tokens=True,
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    val_encodings = processor.tokenizer.batch_encode_plus(
        val_texts,
        add_special_tokens=True,
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Create tensors for ratings
    train_ratings_tensor = torch.tensor(train_ratings).float()
    val_ratings_tensor = torch.tensor(val_ratings).float()

    # Create dataloaders
    train_dataset = TensorDataset(
        train_encodings["input_ids"], train_encodings["attention_mask"], train_ratings_tensor
    )

    val_dataset = TensorDataset(
        val_encodings["input_ids"], val_encodings["attention_mask"], val_ratings_tensor
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    print(f"Initializing model with {args.bert_model}...")
    model = ArticleRecommender(
        bert_model_name=args.bert_model,
        embedding_dim=768,  # Default for BERT
        num_domains=args.num_domains,
        use_domains=args.use_domains,
        freeze_bert=args.freeze_bert,
        mean_highlight_count=mean_count,
        std_highlight_count=std_count,
        scaling_factor=args.scaling_factor,
        max_rating=args.max_rating,
    )

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()  # Changed to MSE loss for continuous rating values

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
