import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms

label_map = {
    "angry": 0,
    "crying": 1,
    "embarrassed": 2,
    "happy": 3,
    "pleased": 4,
    "sad": 5,
    "shock": 6
}

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["IMAGE"]
        image = Image.open(img_path)
        image = transform(image)
        label = label_map[self.df.iloc[idx]["EXPRESSION"]]
        return image, label
    
dataset = EmotionDataset("dataset_labels.csv")
loader = DataLoader( dataset , batch_size=32 ,shuffle=True )

model = nn.Sequential(
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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( model.parameters() , lr=0.0005)
epochs = 30
for epoch in range(epochs):
    running_loss =  0
    for images , labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion( outputs , labels )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {running_loss/len(loader):.4f}"
    )
torch.save( model.state_dict() , "feature_extractor_trained.pth" )
print("Saved!")