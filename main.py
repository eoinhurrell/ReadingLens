import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hard-coded data for article highlights (for a single user)
# Format: (highlight_text, category, rating)
# Categories represent different interest areas for the single user
training_highlights = [
    # Technology/ML highlights
    (
        "Machine learning algorithms require careful tuning for optimal performance",
        "tech",
        5,
    ),
    ("Neural networks have revolutionized the field of computer vision", "tech", 4),
    (
        "Transfer learning allows models to leverage knowledge from pretraining",
        "tech",
        5,
    ),
    (
        "Data preprocessing is a critical step in any machine learning pipeline",
        "tech",
        3,
    ),
    # Literature highlights
    ("The novel explores profound themes of identity and belonging", "literature", 3),
    (
        "Character development is subtle yet transformative throughout the narrative",
        "literature",
        2,
    ),
    (
        "The author's use of metaphor creates a rich tapestry of meaning",
        "literature",
        3,
    ),
    # Finance highlights
    (
        "Effective personal finance begins with creating and following a budget",
        "finance",
        4,
    ),
    (
        "Index funds offer diversification at a lower cost than actively managed funds",
        "finance",
        5,
    ),
    (
        "Compound interest is often described as the eighth wonder of the world",
        "finance",
        4,
    ),
]

# Candidate highlights to predict ratings for
candidate_highlights = [
    (
        "Deep learning architectures include convolutional networks for image processing",
        "tech",
    ),
    (
        "The protagonist's journey mirrors the author's own experiences with displacement",
        "literature",
    ),
    ("Emergency funds should ideally cover 3-6 months of living expenses", "finance"),
    (
        "Attention mechanisms have significantly improved natural language processing models",
        "tech",
    ),
    (
        "The novel's ambiguous ending has sparked debate among literary critics",
        "literature",
    ),
    (
        "Tax-advantaged accounts are cornerstone tools for retirement planning",
        "finance",
    ),
]

# Create text features using TF-IDF
vectorizer = TfidfVectorizer(max_features=100)
all_highlights = [item[0] for item in training_highlights] + [
    item[0] for item in candidate_highlights
]
text_features = vectorizer.fit_transform(all_highlights).toarray()

# Create category mappings
categories = list(set([item[1] for item in training_highlights]))
category_to_idx = {cat: idx for idx, cat in enumerate(categories)}


# Modified Two-Tower Model for a single user
class SingleUserTwoTowerModel(nn.Module):
    def __init__(self, text_feature_dim, num_categories, embedding_dim=32):
        super(SingleUserTwoTowerModel, self).__init__()

        # Interest Category Tower (replacing user tower)
        # This represents different aspects of the single user's interests
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)

        # Item Tower (neural network for processing text features)
        self.item_tower = nn.Sequential(
            nn.Linear(text_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim),
        )

        # Additional layer to model user's general preferences
        self.user_preference = nn.Parameter(torch.randn(embedding_dim))

        # Initialize weights
        nn.init.xavier_uniform_(self.category_embedding.weight)
        # Use normal initialization for the 1D user preference vector instead of xavier_uniform_
        nn.init.normal_(self.user_preference, mean=0.0, std=0.01)

    def forward(self, category_ids, item_features):
        # Get embeddings from both towers
        category_embeddings = self.category_embedding(category_ids)
        item_embeddings = self.item_tower(item_features)

        # Combine user preference with category
        user_embeddings = category_embeddings + self.user_preference

        # Normalize embeddings for cosine similarity
        user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)
        item_embeddings = nn.functional.normalize(item_embeddings, p=2, dim=1)

        # Compute rating prediction (similarity score)
        similarity = torch.sum(user_embeddings * item_embeddings, dim=1)
        ratings = 2.5 + 2.5 * similarity  # Map from [-1,1] to [0,5] range

        return ratings


# Prepare training data
category_ids = []
highlight_indices = []
ratings = []

for highlight, category, rating in training_highlights:
    category_ids.append(category_to_idx[category])
    idx = all_highlights.index(highlight)
    highlight_indices.append(idx)
    ratings.append(rating)

category_ids = torch.LongTensor(category_ids)
ratings = torch.FloatTensor(ratings)
highlight_features = torch.FloatTensor(text_features[highlight_indices])

# Create model, loss function, and optimizer
model = SingleUserTwoTowerModel(
    text_feature_dim=text_features.shape[1],
    num_categories=len(categories),
    embedding_dim=32,
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 300
batch_size = len(category_ids)  # Use all data in each batch (small dataset)

print("Training the Single-User Two-Tower Model...")
for epoch in range(num_epochs):
    # Forward pass
    predicted_ratings = model(category_ids, highlight_features)

    # Compute loss
    loss = criterion(predicted_ratings, ratings)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Function to get highlight recommendations based on interest category
def get_recommendations(category_type=None, top_k=3):
    model.eval()

    # Get indices of candidate highlights
    candidate_indices = [all_highlights.index(h[0]) for h in candidate_highlights]
    candidate_features = torch.FloatTensor(text_features[candidate_indices])

    # If category is specified, filter by that category
    # Otherwise, predict for all categories and take the best overall
    if category_type:
        filtered_candidates = [
            (i, idx)
            for i, (h, cat) in enumerate(candidate_highlights)
            if cat == category_type
        ]

        if not filtered_candidates:
            print(f"No candidates found for category: {category_type}")
            return []

        indices, feature_indices = zip(*filtered_candidates)
        category_tensor = torch.LongTensor([category_to_idx[category_type]]).repeat(
            len(indices)
        )
        filtered_features = torch.FloatTensor(
            np.array([text_features[candidate_indices[i]] for i in indices])
        )

        with torch.no_grad():
            # Predict ratings for filtered candidates
            ratings = model(category_tensor, filtered_features)

        # Create recommendations list
        recommendations = [
            (candidate_highlights[indices[i]][0], float(ratings[i]))
            for i in range(len(indices))
        ]
    else:
        # Get predictions for all candidates in all categories
        all_predictions = []

        for cat in categories:
            cat_id = category_to_idx[cat]
            cat_tensor = torch.LongTensor([cat_id]).repeat(len(candidate_highlights))

            with torch.no_grad():
                ratings = model(cat_tensor, candidate_features)

            for i, rating in enumerate(ratings):
                all_predictions.append((candidate_highlights[i][0], float(rating), cat))

        # Take the best rating for each highlight
        best_ratings = {}
        for highlight, rating, cat in all_predictions:
            if highlight not in best_ratings or rating > best_ratings[highlight][0]:
                best_ratings[highlight] = (rating, cat)

        recommendations = [(h, r) for h, (r, _) in best_ratings.items()]

    # Sort highlights by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:top_k]


# Generate and display recommendations
print("\n=== ARTICLE HIGHLIGHT RECOMMENDATIONS FOR SINGLE USER ===")

# Overall recommendations across all categories
print("\nTop overall recommendations:")
recommendations = get_recommendations()
for i, (highlight, rating) in enumerate(recommendations):
    print(f'  {i+1}. Rating: {rating:.2f}/5.00 - "{highlight}"')

# Recommendations by category
for category in categories:
    print(f"\nTop recommendations for {category.capitalize()} category:")
    recommendations = get_recommendations(category_type=category)
    for i, (highlight, rating) in enumerate(recommendations):
        print(f'  {i+1}. Rating: {rating:.2f}/5.00 - "{highlight}"')

# Personalization analysis
print("\n=== INTEREST ANALYSIS ===")
weights = model.category_embedding.weight.detach().cpu().numpy()
for i, category in enumerate(categories):
    norm = np.linalg.norm(weights[i])
    print(f"{category.capitalize()} interest strength: {norm:.2f}")
