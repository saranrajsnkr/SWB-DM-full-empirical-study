import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# --- 1. MODEL & DATA SETUP ---
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

def partition_cifar10_dirichlet(dataset, num_clients=20, alpha=0.5):
    targets = np.array(dataset.targets)
    client_indices = [[] for _ in range(num_clients)]
    for c in range(10):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])
    return [Subset(dataset, idxs) for idxs in client_indices]

def train_client(model, dataloader, epochs=2, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

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

# --- 2. AGGREGATION FILTERS & MOMENTUM CACHE ---

def coordinate_wise_median(client_states):
    """Computes coordinate-wise median across client parameter tensors."""
    aggregated_dict = {}
    keys = client_states[0].keys()
    for key in keys:
        stacked = torch.stack([state[key].float() for state in client_states], dim=0)
        aggregated_dict[key] = torch.median(stacked, dim=0).values
    return aggregated_dict

def krum_aggregation(client_states, num_byzantine=2):
    """Selects the single update update vector closest to its N-f-2 neighbors."""
    n = len(client_states)
    keys = client_states[0].keys()
    
    # Flatten state dicts into single vectors for distance evaluation
    flat_vectors = []
    for state in client_states:
        flat = torch.cat([state[k].float().view(-1) for k in keys])
        flat_vectors.append(flat)
    flat_vectors = torch.stack(flat_vectors)
    
    scores = []
    for i in range(n):
        distances = torch.norm(flat_vectors - flat_vectors[i], dim=1)
        sorted_dists, _ = torch.sort(distances)
        # Sum distances to the closest (n - num_byzantine - 2) neighbors
        score = torch.sum(sorted_dists[1 : n - num_byzantine - 1])
        scores.append(score.item())
        
    best_idx = np.argmin(scores)
    return client_states[best_idx]

class DelayedMomentumServer:
    """Implements Yamada's Delayed Momentum Aggregation (D-Byz-SGDM)."""
    def __init__(self, global_model, num_clients):
        self.global_model = global_model
        self.num_clients = num_clients
        # Initialize historical momentum buffer for ALL clients
        initial_state = global_model.state_dict()
        self.momentum_buffer = {
            cid: copy.deepcopy(initial_state) for cid in range(num_clients)
        }

    def update_and_aggregate(self, sampled_client_ids, sampled_states, aggregator_fn):
        # 1. Update the cache for newly sampled clients
        for cid, state in zip(sampled_client_ids, sampled_states):
            self.momentum_buffer[cid] = copy.deepcopy(state)
            
        # 2. Extract full pool of 20 states (Fresh + Cached)
        full_pool = [self.momentum_buffer[cid] for cid in range(self.num_clients)]
        
        # 3. Aggregate across the complete 20-client pool
        aggregated_dict = aggregator_fn(full_pool)
        self.global_model.load_state_dict(aggregated_dict)

# --- 3. EXECUTION COMPARISON ---
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

    # Test Delayed Momentum Server with Median Aggregator
    global_model = SimpleCNN().to(device)
    server = DelayedMomentumServer(global_model, num_clients=num_clients)
    
    print("Testing Delayed Momentum + Median Aggregator (No Attacks Yet)...")
    for round_idx in range(1, 11):
        sampled_ids = np.random.choice(num_clients, 10, replace=False)
        sampled_states = []
        
        for cid in sampled_ids:
            local_model = copy.deepcopy(server.global_model)
            state = train_client(local_model, client_loaders[cid], epochs=2, lr=0.01)
            sampled_states.append(state)
            
        server.update_and_aggregate(sampled_ids, sampled_states, aggregator_fn=coordinate_wise_median)
        
        if round_idx % 2 == 0:
            acc = evaluate(server.global_model, test_loader)
            print(f"Round [{round_idx}/10] - Test Accuracy: {acc:.2f}%")