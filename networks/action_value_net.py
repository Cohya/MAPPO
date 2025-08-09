import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPolicyValue(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_size=128):
        super().__init__()
        input_dim = int(torch.prod(torch.tensor(obs_shape)))  # Flattened obs size
        
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Policy head
        self.policy_logits = nn.Linear(hidden_size, action_dim)
        
        # Value head
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        x = obs.view(obs.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        logits = self.policy_logits(x)
        value = self.value(x)
        
        return logits, value
