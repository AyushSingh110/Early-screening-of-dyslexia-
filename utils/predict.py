import torch
import torch.nn as nn
from torchvision import models, transforms
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(config.BASE_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model
