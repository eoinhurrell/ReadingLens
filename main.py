import torch
from torch.utils.data import DataLoader

from readinglens.datasets import TextRatingDataset
from readinglens.models import ReadingLensModel, train_model


def main():
    # Initialize model
    model = ReadingLensModel()

    # Create dataset and dataloader
    dataset = TextRatingDataset()
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)

    # Train the model
    print("Training the model...")
    train_model(model, train_loader)

    model.save()
    # Test with a new text
    test_text = "This comprehensive guide provides valuable insights for beginners"

    # Make prediction
    model.eval()
    with torch.no_grad():
        predicted_rating = model.predict_rating(test_text)
        print("\nTest Results:")
        print(f"Text: {test_text}")
        print(f"Predicted Rating: {predicted_rating:.1f}/10")


if __name__ == "__main__":
    main()
