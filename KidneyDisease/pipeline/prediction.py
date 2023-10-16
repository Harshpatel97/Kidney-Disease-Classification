import os
from typing import Any
import torch
from PIL import Image
from torchvision import transforms

class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
data_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor()
])

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        
    def predict(self):
        # Load model
        model = torch.load("artifacts\models\model.pth",
                           map_location=torch.device('cpu')).to('cpu')

        image = Image.open(self.filename)
        
        # Transform the target image and add a batch dimension
        img = data_transform(image).unsqueeze(0)
        
        # Put model into evaluation mode and turn on inference mode
        model.eval()
        with torch.inference_mode():
            pred_probs = torch.softmax(model(img), dim=1)
        
        # Create a prediction label and prediction probability dictionary for each prediction class.
        pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
        Predictions = dict(sorted(pred_labels_and_probs.items(), key=lambda item: item[1], reverse=True))
        return [{'image': Predictions}]
        