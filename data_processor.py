#!/usr/bin/env python
import argparse
import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
import torch


class MarkdownProcessor:
    def __init__(self, tokenizer_name: str = "bert-base-uncased", max_length: int = 512):
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
        highlights = re.findall(r"## (.+?)(?=\n\n|$)(.*?)(?=\n## |\n# |$)", content, re.DOTALL)
        highlighted_sections = []

        for heading, content in highlights:
            highlighted_sections.append({"heading": heading.strip(), "content": content.strip()})

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
            num_negative = min(int(num_positive * neg_to_pos_ratio), len(negative_paragraphs))

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

    def process_and_save(
        self, input_dir: str, output_dir: str, train_ratio: float = 0.8, seed: int = 42
    ):
        """
        Process all markdown files in input_dir and save processed data to output_dir

        Args:
            input_dir: Directory containing markdown files
            output_dir: Directory to save processed data
            train_ratio: Ratio of data to use for training vs validation
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

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

        # Save article data for reference
        with open(os.path.join(output_dir, "article_data.json"), "w", encoding="utf-8") as f:
            json.dump(article_data, f, indent=2)

        # Split into train and validation
        random.shuffle(examples)
        split_idx = int(len(examples) * train_ratio)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        print(f"Train examples: {len(train_examples)}, Validation examples: {len(val_examples)}")

        # Preprocess and save train set
        train_data = self.preprocess_for_bert(train_examples)
        # Use pickle protocol to ensure proper serialization of dictionary with tensors
        torch.save(
            {
                "input_ids": train_data["input_ids"],
                "attention_mask": train_data["attention_mask"],
                "labels": train_data["labels"],
            },
            os.path.join(output_dir, "train_data.pt"),
            pickle_protocol=4,
        )

        # Preprocess and save validation set
        val_data = self.preprocess_for_bert(val_examples)
        torch.save(
            {
                "input_ids": val_data["input_ids"],
                "attention_mask": val_data["attention_mask"],
                "labels": val_data["labels"],
            },
            os.path.join(output_dir, "val_data.pt"),
            pickle_protocol=4,
        )

        # Save examples for reference
        with open(os.path.join(output_dir, "train_examples.json"), "w", encoding="utf-8") as f:
            json.dump(train_examples, f, indent=2)

        with open(os.path.join(output_dir, "val_examples.json"), "w", encoding="utf-8") as f:
            json.dump(val_examples, f, indent=2)

        print(f"Processed data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process markdown files for training a recommender model"
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing markdown files")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed data")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of data for training vs validation"
    )
    parser.add_argument(
        "--tokenizer", default="bert-base-uncased", help="Huggingface tokenizer to use"
    )
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    processor = MarkdownProcessor(tokenizer_name=args.tokenizer, max_length=args.max_length)

    processor.process_and_save(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
