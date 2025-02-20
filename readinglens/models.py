from typing import Dict, List

import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig
from torchrec.models.dlrm import DLRM
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(
        self, model_name: str = "distilbert-base-uncased", output_dim: int = 64
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, output_dim).to(
            self.device
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        # Tokenize and encode text
        encodings = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Move encodings to the same device as the model
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Get BERT embeddings (use CLS token)
        outputs = self.bert(**encodings)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # Project to desired dimension
        return self.projection(cls_embeddings)


class ReadingLensModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        num_dense_features: int = 1,  # text embedding will be treated as dense
        dense_arch_layer_sizes: List[int] = [512, 256, 64],
        over_arch_layer_sizes: List[int] = [512, 512, 256, 1],
    ):
        super().__init__()

        # Text encoder
        self.text_encoder = TextEncoder(output_dim=embedding_dim)

        # Define embedding bag configuration correctly
        embedding_bag_config = [
            EmbeddingBagConfig(
                name="rating_bucket",
                embedding_dim=embedding_dim,
                num_embeddings=10,
                feature_names=["rating_bucket"],
            )
        ]

        # Create embedding bag collection
        self.embedding_bags = EmbeddingBagCollection(
            tables=embedding_bag_config,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Create DLRM model
        self.dlrm = DLRM(
            embedding_bag_collection=self.embedding_bags,
            dense_in_features=num_dense_features
            * embedding_dim,  # text embedding dimension
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
        )

    def forward(
        self, texts: List[str], sparse_features: Dict[str, torch.Tensor] = None
    ):
        # Encode text
        text_embeddings = self.text_encoder(texts)

        # If no sparse features provided, create empty dict with zero tensors
        if sparse_features is None:
            batch_size = len(texts)
            sparse_features = {
                "rating_bucket": torch.zeros((batch_size, 1), dtype=torch.long)
            }

        # Forward pass through DLRM
        return self.dlrm(
            dense_features=text_embeddings.view(text_embeddings.shape[0], -1),
            sparse_features=sparse_features,
        )

    def create_sparse_features(
        self, batch_size: int, device: torch.device
    ) -> KeyedJaggedTensor:
        """Create sparse features in the correct format for TorchRec."""
        values = torch.zeros(batch_size, dtype=torch.int32)
        lengths = torch.ones(batch_size, dtype=torch.int32)

        return KeyedJaggedTensor(
            keys=["rating_bucket"],
            values=values,
            lengths=lengths,
        ).to(device)

    def predict_rating(self, text: str) -> float:
        """Convenience method to predict rating for a single text."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            # Get text embedding
            text_embedding = self.text_encoder([text])

            # Create sparse features in the correct format
            sparse_features = self.create_sparse_features(1, device)

            # Make prediction
            prediction = self.dlrm(
                dense_features=text_embedding.view(text_embedding.shape[0], -1),
                sparse_features=sparse_features,
            )
            return prediction.item()

    def save(self):
        save_model(self, "./output/model.safetensors")

    def load(self):
        load_model(self, "./output/model.safetensors")


# Training function
def train_model(model, train_loader, num_epochs=50, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for texts, ratings in train_loader:
            optimizer.zero_grad()

            # Create sparse features in the correct format
            sparse_features = model.create_sparse_features(len(texts), device)
            # if torch.cuda.is_available():
            #     sparse_features = sparse_features.to(device)
            ratings = ratings.to(device)

            # Forward pass
            predictions = model(texts, sparse_features)
            loss = criterion(predictions.squeeze(), ratings)

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
