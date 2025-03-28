#!/usr/bin/env python
import argparse
import os
import re
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
import requests
from bs4 import BeautifulSoup
import html2text
from tqdm import tqdm
from urllib.parse import urlparse

from model import ArticleRecommender


class ArticleProcessor:
    """
    Process articles from URLs for prediction
    """
    
    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        device: str = "cpu"
    ):
        """
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            device: Device to run model on
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.device = device
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = False
        self.html_converter.single_line_break = True
        self.html_converter.mark_code = True
        
    def fetch_article(self, url: str) -> str:
        """
        Fetch article content from URL
        
        Args:
            url: URL to fetch
            
        Returns:
            Raw HTML content
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            raise ValueError(f"Failed to fetch article from {url}: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive whitespace, etc.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with single newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_article_content(self, html: str, url: str) -> Dict:
        """
        Extract article content from HTML
        
        Args:
            html: HTML content
            url: URL of the article
            
        Returns:
            Dictionary with article content and metadata
        """
        soup = BeautifulSoup(html, "lxml")
        
        # Try to get title
        title = None
        if soup.title:
            title = soup.title.text.strip()
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()
        
        # Convert to markdown
        markdown = self.html_converter.handle(str(soup))
        clean_markdown = self.clean_text(markdown)
        
        # Get domain from URL
        domain = urlparse(url).netloc
        
        # Extract potential article sections (paragraphs)
        sections = []
        paragraphs = re.split(r"\n\n+", clean_markdown)
        
        for para in paragraphs:
            para = para.strip()
            if para and len(para.split()) > 10:  # Skip very short paragraphs
                sections.append(para)
        
        return {
            "title": title,
            "domain": domain,
            "url": url,
            "content": clean_markdown,
            "sections": sections
        }
    
    def process_article_for_prediction(self, url: str) -> Dict:
        """
        Process article from URL for prediction
        
        Args:
            url: URL of the article
            
        Returns:
            Dictionary with processed article data
        """
        # Fetch article
        html = self.fetch_article(url)
        
        # Extract content
        article_data = self.extract_article_content(html, url)
        
        # Process sections for model input
        processed_sections = []
        
        for i, section in enumerate(article_data["sections"]):
            # Tokenize section
            encoded = self.tokenizer.encode_plus(
                section,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            processed_sections.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "text": section,
                "index": i
            })
        
        article_data["processed_sections"] = processed_sections
        return article_data


class ArticleRecommenderPredictor:
    """
    Predictor class for article recommendation
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "bert-base-uncased",
        device: str = "cpu",
        top_k: int = 3
    ):
        """
        Args:
            model_path: Path to the trained model
            tokenizer_name: Name of the tokenizer to use
            device: Device to run model on
            top_k: Number of top sections to consider for overall score
        """
        self.device = device
        self.top_k = top_k
        
        # Load model
        self.model = ArticleRecommender.load(model_path, device=device)
        self.model.to(device)
        self.model.eval()
        
        # Initialize article processor
        self.processor = ArticleProcessor(
            tokenizer_name=tokenizer_name,
            device=device
        )
    
    def predict(self, url: str) -> Dict:
        """
        Generate predictions for an article
        
        Args:
            url: URL of the article
            
        Returns:
            Dictionary with predictions and article data
        """
        # Process article
        article_data = self.processor.process_article_for_prediction(url)
        
        # Generate predictions for each section
        section_scores = []
        
        with torch.no_grad():
            for section in tqdm(article_data["processed_sections"], desc="Predicting"):
                # Forward pass
                output = self.model(
                    input_ids=section["input_ids"],
                    attention_mask=section["attention_mask"]
                )
                
                # Get score
                score = output.item()
                
                section_scores.append({
                    "index": section["index"],
                    "text": section["text"],
                    "score": score
                })
        
        # Sort sections by score (descending)
        sorted_sections = sorted(section_scores, key=lambda x: x["score"], reverse=True)
        
        # Calculate overall article score (average of top-k sections)
        top_sections = sorted_sections[:self.top_k]
        overall_score = np.mean([s["score"] for s in top_sections]) if top_sections else 0.0
        
        # Prepare result
        result = {
            "url": url,
            "title": article_data["title"],
            "domain": article_data["domain"],
            "overall_score": float(overall_score),
            "top_sections": top_sections,
            "all_sections": sorted_sections
        }
        
        return result
    
    def format_result(self, result: Dict) -> str:
        """
        Format prediction result as human-readable text
        
        Args:
            result: Prediction result
            
        Returns:
            Formatted result
        """
        output = []
        output.append(f"# Article Rating: {result['title']}")
        output.append(f"URL: {result['url']}")
        output.append(f"Domain: {result['domain']}")
        output.append(f"Overall Score: {result['overall_score']:.2f} / 1.00")
        output.append("")
        
        output.append("## Potential Highlights")
        for i, section in enumerate(result["top_sections"]):
            output.append(f"### Highlight {i+1} (Score: {section['score']:.2f})")
            output.append(section["text"])
            output.append("")
        
        return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Predict article rating and highlights")
    parser.add_argument("--url", required=True, help="URL of the article to rate")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--output_file", help="Path to save prediction results (optional)")
    parser.add_argument("--tokenizer", default="bert-base-uncased", help="Huggingface tokenizer to use")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top sections to consider for overall score")
    parser.add_argument("--use_cuda", action="store_true", help="Whether to use CUDA if available")
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Initialize predictor
    predictor = ArticleRecommenderPredictor(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer,
        device=device,
        top_k=args.top_k
    )
    
    try:
        # Generate prediction
        print(f"Processing article: {args.url}")
        result = predictor.predict(args.url)
        
        # Format result
        formatted_result = predictor.format_result(result)
        
        # Print or save result
        if args.output_file:
            # Save to file
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(formatted_result)
                
            # Also save raw result as JSON
            json_path = os.path.splitext(args.output_file)[0] + ".json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                
            print(f"Results saved to {args.output_file} and {json_path}")
        else:
            # Print to console
            print("\n" + formatted_result)
            
    except Exception as e:
        print(f"Error predicting article rating: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())