import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# DQN parameters
ALPHA = 0.001  # Learning rate
GAMMA = 0.99  # Discount factor
EPSILON = 0.00  # Exploration probability
TAU = 0.005 # Soft update parameter

Q_NETWORK_FILE = "q_network_22000.pth"

# Define the neural network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize network
state_dim = 16  # Based on preprocessed state
action_dim = 6  # Six possible actions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_network = DQN(state_dim, action_dim).to(device)
target_network = DQN(state_dim, action_dim).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.AdamW(q_network.parameters(), lr=ALPHA, amsgrad=True)
loss_fn = nn.SmoothL1Loss()

# Load model if exists
if os.path.exists(Q_NETWORK_FILE):
    q_network.load_state_dict(torch.load(Q_NETWORK_FILE, map_location=device))

def preprocess_state(state):
    # taxi_row, taxi_col, _, _, _, _, _, _, _, _, obstacle_n, obstacle_s, obstacle_e, obstacle_w, passenger_look, destination_look = state
    # return np.array([taxi_row, taxi_col, obstacle_n, obstacle_s, obstacle_e, obstacle_w, passenger_look, destination_look], dtype=np.float32)
    
    return np.array(state, dtype=np.float32)

def get_action(obs, epsilon=EPSILON):
    """Choose an action using epsilon-greedy policy."""
    state = preprocess_state(obs)
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2, 3, 4, 5])
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_network(state_tensor)
        return torch.argmax(q_values).item()

def update_q_network(batch):
    """Update Q-network using a batch from experience replay."""
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(np.array([preprocess_state(s) for s in states]), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array([preprocess_state(s) for s in next_states]), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_network(next_states).max(1).values
    target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
    
    loss = loss_fn(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Soft update of the target network's weights
# θ′ ← τ θ + (1 −τ )θ′
def soft_update_target_network():
    target_state_dict = target_network.state_dict()
    q_state_dict = q_network.state_dict()
    for key in q_state_dict:
        target_state_dict[key] = q_state_dict[key] * TAU + target_state_dict[key] * (1 - TAU)
    target_network.load_state_dict(target_state_dict)

def save_q_network(episode):
    """Save the Q-network to a file."""
    # torch.save(q_network.state_dict(), Q_NETWORK_FILE)
    torch.save(q_network.state_dict(), f"q_network_{episode}.pth")
