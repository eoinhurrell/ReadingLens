import torch
from torch.utils.data import Dataset

# Sample training data
training_texts = [
    "This article perfectly explains the complex topic in simple terms",  # Rating: 9.5
    "Good information but could be more concise",  # Rating: 7.0
    "The writing is confusing and hard to follow",  # Rating: 3.0
    "Excellent insights with practical examples",  # Rating: 9.0
    "Basic introduction but lacks depth",  # Rating: 5.0
    "Clear and well-structured explanation",  # Rating: 8.5
    "Too many technical terms without proper explanation",  # Rating: 4.0
    "Engaging writing style with valuable content",  # Rating: 8.0
    "Outdated information, needs revision",  # Rating: 2.5
    "Perfect balance of theory and practice",  # Rating: 9.0
]

training_ratings = torch.tensor(
    [9.5, 7.0, 3.0, 9.0, 5.0, 8.5, 4.0, 8.0, 2.5, 9.0], dtype=torch.float32
)


# Create dataset class
class TextRatingDataset(Dataset):
    def __init__(self):
        self.texts = training_texts
        self.ratings = training_ratings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.ratings[idx]
