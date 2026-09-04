# Shared utilities for SWB-DM project
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# def get_cifar10_loaders(num_clients=20, alpha=0.5, batch_size=32): (slashed before exprement.py)
def get_cifar10_loaders(num_clients=20, alpha=0.5, batch_size=32, root_size=1000):#(added before run exprement.py)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    targets = np.array(trainset.targets)
    client_indices = [[] for _ in range(num_clients)]
    for c in range(10):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])
            
    client_loaders = [DataLoader(Subset(trainset, idxs), batch_size=batch_size, shuffle=True) for idxs in client_indices]
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)

        # Add this right before the return statement ( added before exprement.py):
    np.random.seed(1234) # Fixed seed for root dataset reproducibility
    root_indices = np.random.choice(len(trainset), root_size, replace=False).tolist()
    root_loader = DataLoader(Subset(trainset, root_indices), batch_size=batch_size, shuffle=True)
    
    return client_loaders, test_loader, root_loader
    # return client_loaders, test_loader(slashed before exprement.py)

# Slashed before expremental attack 
# def train_client(model, dataloader, epochs=2, lr=0.01):
#     model.train()
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
#     for _ in range(epochs):
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(images), labels)
#             loss.backward()
#             optimizer.step()
#     return model.state_dict()


def train_client(model, dataloader, epochs=2, lr=0.01, attack_type=None, noise_scale=1.0, num_classes=10):
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if attack_type == "label_flip":
                labels = (num_classes - 1) - labels
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    state = model.state_dict()
    if attack_type == "sign_flip":
        for k in state:
            update = state[k].float() - initial_state[k].float()
            state[k] = (initial_state[k].float() - update).to(state[k].dtype)
    elif attack_type == "gaussian":
        for k in state:
            update = state[k].float() - initial_state[k].float()
            sigma = update.norm() / max(update.numel() ** 0.5, 1) * noise_scale
            state[k] = (initial_state[k].float() + torch.randn_like(update) * sigma).to(state[k].dtype)
    return state


def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

