"""
Inference API for Vision Classification Framework
Provides easy-to-use prediction functions for trained models
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os


class Predictor:
    """
    Inference API for making predictions with trained models

    Example usage:
        predictor = Predictor('path/to/model.pth', 'resnet50', ['cat', 'dog'])
        prediction = predictor.predict_image('path/to/image.jpg')
    """

    def __init__(self, model_path, model_name, classes, attention_name=None, attention_index=4, device=None):
        """
        Initialize the predictor

        Args:
            model_path: Path to saved model weights (.pth file)
            model_name: Name of the model architecture
            classes: List of class names
            attention_name: Optional attention mechanism name
            attention_index: Attention layer index
            device: Device to run inference on (cuda/cpu)
        """
        from src.modules import pretrained_network

        self.model_name = model_name
        self.classes = classes
        self.num_classes = len(classes)
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = pretrained_network(model_name, attention_name, attention_index, self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        print(f"[INFO]  Predictor initialized with {model_name} on {self.device}")
        print(f"[INFO]  Classes: {classes}")

    def predict_image(self, image_path, top_k=1):
        """
        Predict class for a single image

        Args:
            image_path: Path to the image file
            top_k: Return top k predictions

        Returns:
            Dictionary with predictions
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)

        # Format results
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                'class': self.classes[idx.item()],
                'class_index': idx.item(),
                'probability': prob.item(),
                'confidence': f"{prob.item()*100:.2f}%"
            })

        result = {
            'image_path': image_path,
            'predictions': predictions,
            'top_prediction': predictions[0]['class'],
            'top_probability': predictions[0]['probability']
        }

        return result

    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict classes for multiple images

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for inference

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            valid_paths = []

            # Load and preprocess batch
            for path in batch_paths:
                if not os.path.exists(path):
                    print(f"[WARNING] Skipping missing image: {path}")
                    continue
                try:
                    image = Image.open(path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"[WARNING] Error loading {path}: {e}")
                    continue

            if not batch_tensors:
                continue

            # Stack tensors and predict
            batch_input = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_input)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, predictions = torch.max(probabilities, 1)

            # Format results
            for path, pred_idx, prob in zip(valid_paths, predictions, max_probs):
                results.append({
                    'image_path': path,
                    'predicted_class': self.classes[pred_idx.item()],
                    'class_index': pred_idx.item(),
                    'probability': prob.item(),
                    'confidence': f"{prob.item()*100:.2f}%"
                })

        return results

    def predict_from_folder(self, folder_path, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """
        Predict classes for all images in a folder

        Args:
            folder_path: Path to folder containing images
            extensions: Tuple of valid image extensions

        Returns:
            List of prediction dictionaries
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Collect all image paths
        image_paths = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(folder_path, filename))

        print(f"[INFO]  Found {len(image_paths)} images in {folder_path}")

        if not image_paths:
            print("[WARNING] No images found!")
            return []

        # Batch predict
        results = self.predict_batch(image_paths)

        return results

    def predict_with_PIL_image(self, pil_image):
        """
        Predict class for a PIL Image object

        Args:
            pil_image: PIL Image object

        Returns:
            Dictionary with prediction
        """
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, prediction = torch.max(probabilities, 1)

        result = {
            'predicted_class': self.classes[prediction.item()],
            'class_index': prediction.item(),
            'probability': max_prob.item(),
            'confidence': f"{max_prob.item()*100:.2f}%",
            'all_probabilities': {
                self.classes[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }

        return result

    def get_class_probabilities(self, image_path):
        """
        Get probabilities for all classes

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary mapping class names to probabilities
        """
        result = self.predict_image(image_path, top_k=self.num_classes)
        probabilities = {
            pred['class']: pred['probability']
            for pred in result['predictions']
        }
        return probabilities


def predict_single(model_path, image_path, model_name, classes, attention_name=None, attention_index=4):
    """
    Quick prediction for a single image (convenience function)

    Args:
        model_path: Path to saved model weights
        image_path: Path to image
        model_name: Model architecture name
        classes: List of class names
        attention_name: Optional attention mechanism
        attention_index: Attention layer index

    Returns:
        Predicted class name
    """
    predictor = Predictor(model_path, model_name, classes, attention_name, attention_index)
    result = predictor.predict_image(image_path)
    return result['top_prediction']


if __name__ == "__main__":
    # Example usage
    print("Inference API for Vision Classification Framework")
    print("=" * 60)
    print("\nExample usage:")
    print("""
    from src.inference import Predictor

    # Initialize predictor
    predictor = Predictor(
        model_path='log/exp/best.pth',
        model_name='resnet50',
        classes=['cat', 'dog', 'bird']
    )

    # Predict single image
    result = predictor.predict_image('test.jpg')
    print(f"Predicted: {result['top_prediction']}")

    # Predict multiple images
    results = predictor.predict_batch(['img1.jpg', 'img2.jpg'])

    # Predict from folder
    results = predictor.predict_from_folder('test_images/')
    """)
