import os
import joblib
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

idx_to_label = {
    0 : "angry" ,
    1 : "crying",
    2 : "embarrassed",
    3 : "happy",
    4 : "pleased" ,
    5 : "sad",
    6 : "shock"
}

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

full_model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(128, 256, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(256 * 3 * 3, 512),
    nn.ReLU(),
    nn.Linear(512, 64),
    nn.ReLU(),
    nn.Linear(64, 7)
)

full_model.load_state_dict( torch.load( "feature_extractor_trained.pth",map_location="cpu"))
full_model.eval()
feature_extractor =  nn.Sequential(*list(full_model.children())[:-1])
classifier  = joblib.load("emotion_classifier.pkl")

for file in sorted( os.listdir( ".")):
    if (file.startswith("testimg")  and file.endswith(".png") ):
        image =  Image.open(file)
        image =  transform(image)
        image  = image.unsqueeze(0)
        with torch.no_grad():
            embedding  = feature_extractor(image)
        embedding = embedding.numpy()
        prediction = classifier.predict(embedding)[0]
        emotion = idx_to_label[prediction]
        print(f"{file} -> {emotion}")