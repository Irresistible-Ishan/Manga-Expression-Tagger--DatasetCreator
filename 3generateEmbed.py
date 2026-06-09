import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

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

full_model.load_state_dict(torch.load("feature_extractor_trained.pth",map_location="cpu"))
full_model.eval()
feature_extractor = nn.Sequential(*list(full_model.children())[:-1])
df = pd.read_csv("dataset_labels.csv")
label_map = {"angry": 0,"crying": 1,"embarrassed": 2,"happy": 3,"pleased": 4,"sad": 5,"shock": 6}
embeddings = []
labels = []
for _ , row in df.iterrows():
    image = Image.open(row["IMAGE"])
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        embedding = feature_extractor(image)
    embeddings.append(embedding.squeeze().numpy())
    labels.append(label_map[row["EXPRESSION"]])
embeddings = np.array(embeddings)
labels = np.array(labels)
np.save( "embeddings.npy" , embeddings )
np.save( "labels.npy" , labels )
print( "Embeddings:" , embeddings.shape )
print(  "Labels:" , labels.shape )