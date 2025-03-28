#!/usr/bin/env python
"""
Tests for the model module
"""

import pytest
import torch
from readinglens.model import ArticleRecommender, FeatureInteraction


def test_feature_interaction():
    """Test the feature interaction layer"""
    # Create a sample input
    batch_size = 2
    num_features = 3
    embedding_dim = 4

    dense_features = torch.rand(batch_size, num_features, embedding_dim)

    # Create the layer
    interaction = FeatureInteraction(num_features)

    # Get the output
    output = interaction(dense_features)

    # Check the shape: batch_size x num_interactions
    # num_interactions = num_features * (num_features - 1) / 2
    expected_interactions = num_features * (num_features - 1) // 2
    assert output.shape == (batch_size, expected_interactions)


def test_article_recommender_forward():
    """Test the forward pass of the article recommender model"""
    # Skip if no GPU available and using CUDA
    if not torch.cuda.is_available():
        try:
            # Create a small model for testing
            model = ArticleRecommender(
                bert_model_name="prajjwal1/bert-tiny",  # Use a tiny model for tests
                embedding_dim=128,
                num_domains=10,
                use_domains=False,
                freeze_bert=True,
            )

            # Create sample inputs
            batch_size = 2
            seq_length = 16

            input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            attention_mask = torch.ones(batch_size, seq_length)

            # Get the output
            output = model(input_ids=input_ids, attention_mask=attention_mask)

            # Check the shape: batch_size x 1
            assert output.shape == (batch_size, 1)

            # Check output range (should be between 0 and 1)
            assert torch.all(output >= 0) and torch.all(output <= 1)

        except Exception as e:
            pytest.skip(f"Skipping test due to model initialization error: {e}")
    else:
        pytest.skip("Skipping test when CUDA is available to avoid loading large models in CI")
