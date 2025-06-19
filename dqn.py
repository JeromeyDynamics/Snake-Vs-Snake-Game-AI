import torch
import torch.nn as nn
import random

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """Initializes the DQN model with the given sizes.

        Args:
            input_size (int): The size of the input state vector.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output vector representing the q values.
        """
        
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Computes the forward pass of the DQN model.
        
        Args:
            x (torch.tensor): The input state vector.
        
        Returns:
            torch.tensor: The output vector representing the q values.
        """
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if needed
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def select_action(state, q_network, epsilon):
    """Selects an action based on an epsilon-greedy policy.

    Args:
        state (list): The state vector.
        q_network (DQN): The DQN model used to compute the q values.
        epsilon (float): The probability of taking a random action.

    Returns:
        int: The action to take.
    """
    
    if random.random() < epsilon:
        return random.randint(0, 3)  # Random action
    else:
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = q_network(state)
        return torch.argmax(q_values).item()

def update_network(q_network, target_network, optimizer, memory):
    """
    Updates the Q-network using a batch of experiences from replay memory.

    This function samples a batch of experiences from the replay memory, computes 
    the loss between the predicted Q-values and the target Q-values, and performs 
    a gradient descent step to update the Q-network's weights.

    Args:
        q_network (DQN): The Q-network being trained.
        target_network (DQN): The target Q-network used to compute the target Q-values.
        optimizer (torch.optim.Optimizer): The optimizer used to update the Q-network.
        memory (ReplayMemory): The replay memory containing past experiences.

    Returns:
        None
    """

    if len(memory) < 128:  # Increased batch size
        return
    batch = memory.sample(128)  # Larger batch for efficiency
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_network(next_states).max(1)[0]
    targets = rewards + 0.99 * next_q_values * (1 - dones)

    loss = nn.functional.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
