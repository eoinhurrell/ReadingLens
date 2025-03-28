#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from typing import Dict, List, Optional, Union, Tuple


class FeatureInteraction(nn.Module):
    """
    Feature interaction layer based on DLRM architecture.
    Performs dot product interactions between dense features.
    """

    def __init__(self, num_features: int):
        """
        Args:
            num_features: Number of input features
        """
        super().__init__()
        self.num_features = num_features

    def forward(self, dense_features: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise dot product interactions between features

        Args:
            dense_features: Tensor of shape (batch_size, num_features, embedding_dim)

        Returns:
            Tensor of shape (batch_size, num_interactions) containing interaction terms
        """
        batch_size = dense_features.size(0)

        # Compute dot product interactions (all pairs of features)
        # For all i, j where i < j
        dot_products = []
        for i in range(self.num_features):
            for j in range(i + 1, self.num_features):
                # For each pair, compute dot product
                interaction = torch.sum(dense_features[:, i, :] * dense_features[:, j, :], dim=1)
                dot_products.append(interaction)

        # Stack all interactions
        if dot_products:
            return torch.stack(dot_products, dim=1)
        else:
            # In case there are no interactions (only one feature)
            return torch.zeros((batch_size, 0), device=dense_features.device)


class DomainEmbedding(nn.Module):
    """
    Simple embedding layer for domain information
    """

    def __init__(self, num_domains: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_domains + 1, embedding_dim)  # +1 for unknown domains

    def forward(self, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for domain IDs

        Args:
            domain_ids: Tensor of shape (batch_size,) containing domain IDs

        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        return self.embedding(domain_ids)


class ArticleRecommender(nn.Module):
    """
    Hybrid model combining BERT and DLRM concepts for article recommendation
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        num_domains: int = 100,
        use_domains: bool = False,
        freeze_bert: bool = False,
        mean_highlight_count: float = 3.0,
        std_highlight_count: float = 1.0,
        highlight_threshold: float = 0.5,
        scaling_factor: float = 1.0,
        max_rating: float = 5.0,
    ):
        """
        Args:
            bert_model_name: Name of the pretrained BERT model
            embedding_dim: Dimension of embeddings
            num_domains: Number of unique domains (for domain embedding)
            use_domains: Whether to use domain information
            freeze_bert: Whether to freeze BERT parameters
            mean_highlight_count: Mean number of highlights across training corpus
            std_highlight_count: Standard deviation of highlight count
            highlight_threshold: Threshold for considering a section as highlighted
            scaling_factor: Factor for scaling z-scores to rating scale
            max_rating: Maximum value for rating scale
        """
        super().__init__()

        # BERT for text encoding
        self.bert = BertModel.from_pretrained(bert_model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Embedding dimension from BERT
        self.embedding_dim = embedding_dim

        # Domain embedding (optional)
        self.use_domains = use_domains
        if use_domains:
            self.domain_embedding = DomainEmbedding(num_domains, embedding_dim)
            self.num_features = 2  # BERT + domain
        else:
            self.num_features = 1  # BERT only

        # Feature interaction layer
        self.feature_interaction = FeatureInteraction(self.num_features)

        # Number of interaction terms
        num_interactions = self.num_features * (self.num_features - 1) // 2

        # Final MLP layers
        mlp_input_dim = embedding_dim + num_interactions

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Highlight statistics for rating calculation
        self.mean_highlight_count = mean_highlight_count
        self.std_highlight_count = std_highlight_count
        self.highlight_threshold = highlight_threshold
        self.scaling_factor = scaling_factor
        self.max_rating = max_rating

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            input_ids: Tensor of shape (batch_size, seq_length) from BERT tokenizer
            attention_mask: Tensor of shape (batch_size, seq_length) from BERT tokenizer
            domain_ids: Optional tensor of shape (batch_size,) containing domain IDs

        Returns:
            Tensor of shape (batch_size, 1) containing prediction scores
        """
        # Get BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embedding = bert_output.pooler_output  # [batch_size, embedding_dim]

        # Prepare dense features list
        dense_features = [bert_embedding]

        # Add domain embeddings if enabled
        if self.use_domains and domain_ids is not None:
            domain_embedding = self.domain_embedding(domain_ids)
            dense_features.append(domain_embedding)

        # Stack features for interaction
        dense_features_stacked = torch.stack(
            dense_features, dim=1
        )  # [batch_size, num_features, embedding_dim]

        # Get interaction terms
        interactions = self.feature_interaction(dense_features_stacked)

        # Concatenate first feature (BERT) with interactions
        concatenated = torch.cat([bert_embedding, interactions], dim=1)

        # Final prediction
        logits = self.mlp(concatenated)

        # For ratings from 1 to max_rating, we need a linear output without sigmoid
        # We use a scaled sigmoid and offset it to produce values in the [1, max_rating] range
        scaled_logits = 1.0 + (self.max_rating - 1.0) * torch.sigmoid(logits)

        return scaled_logits

    def save(self, path: str):
        """
        Save the model

        Args:
            path: Path to save the model
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "bert_model_name": self.bert.config._name_or_path,
                "embedding_dim": self.embedding_dim,
                "use_domains": self.use_domains,
                "mean_highlight_count": self.mean_highlight_count,
                "std_highlight_count": self.std_highlight_count,
                "highlight_threshold": self.highlight_threshold,
                "scaling_factor": self.scaling_factor,
                "max_rating": self.max_rating,
                "config": {
                    "bert_model_name": self.bert.config._name_or_path,
                    "embedding_dim": self.embedding_dim,
                    "use_domains": self.use_domains,
                    "num_features": self.num_features,
                    "mean_highlight_count": self.mean_highlight_count,
                    "std_highlight_count": self.std_highlight_count,
                    "highlight_threshold": self.highlight_threshold,
                    "scaling_factor": self.scaling_factor,
                    "max_rating": self.max_rating,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """
        Load a saved model

        Args:
            path: Path to the saved model
            device: Device to load the model to

        Returns:
            Loaded ArticleRecommender model
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]

        model = cls(
            bert_model_name=config["bert_model_name"],
            embedding_dim=config["embedding_dim"],
            use_domains=config.get("use_domains", False),
            mean_highlight_count=config.get("mean_highlight_count", 3.0),
            std_highlight_count=config.get("std_highlight_count", 1.0),
            highlight_threshold=config.get("highlight_threshold", 0.5),
            scaling_factor=config.get("scaling_factor", 1.0),
            max_rating=config.get("max_rating", 5.0),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model
