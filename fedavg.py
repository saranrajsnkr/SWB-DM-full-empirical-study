import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

# Set device and random seeds for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# --- 1. MODEL ARCHITECTURE ---
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

# --- 2. DIRICHLET NON-IID PARTITIONING ---
def partition_cifar10_dirichlet(dataset, num_clients=20, alpha=0.5):
    targets = np.array(dataset.targets)
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to split indices
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])
            
    return [Subset(dataset, idxs) for idxs in client_indices]

# --- 3. LOCAL CLIENT TRAINING ---
def train_client(model, dataloader, epochs=2, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return model.state_dict()

# --- 4. FEDAVG AGGREGATION ---
def fed_avg(global_model, client_states, client_sample_counts):
    total_samples = sum(client_sample_counts)
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float)
        for state, num_samples in zip(client_states, client_sample_counts):
            global_dict[key] += state[key] * (num_samples / total_samples)
            
    global_model.load_state_dict(global_dict)
    return global_model

# --- 5. EVALUATION ---
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

# --- 6. MAIN SIMULATION LOOP ---
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)

    num_clients = 20
    client_subsets = partition_cifar10_dirichlet(trainset, num_clients=num_clients, alpha=0.5)
    client_loaders = [DataLoader(sub, batch_size=32, shuffle=True) for sub in client_subsets]
    client_sample_counts = [len(sub) for sub in client_subsets]

    global_model = SimpleCNN().to(device)
    num_rounds = 50
    sample_rate = 0.5 # Select 10 out of 20 clients each round

    print(f"Starting FedAvg across {num_clients} clients (Dirichlet alpha=0.5)...")
    for round_idx in range(1, num_rounds + 1):
        num_sampled = int(num_clients * sample_rate)
        sampled_client_ids = np.random.choice(num_clients, num_sampled, replace=False)
        
        client_states = []
        sampled_counts = []
        
        for cid in sampled_client_ids:
            local_model = copy.deepcopy(global_model)
            updated_state = train_client(local_model, client_loaders[cid], epochs=2, lr=0.01)
            client_states.append(updated_state)
            sampled_counts.append(client_sample_counts[cid])
            
        global_model = fed_avg(global_model, client_states, sampled_counts)
        
        if round_idx % 5 == 0 or round_idx == 1:
            acc = evaluate(global_model, test_loader)
            print(f"Round [{round_idx}/{num_rounds}] - Test Accuracy: {acc:.2f}%")