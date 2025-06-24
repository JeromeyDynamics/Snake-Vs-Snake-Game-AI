import torch
import torch.nn as nn
import random

#the DQN class defines a neural network that akes in the current state of the game as input and outputs a set of Q-values for each of the 4 directions the snake could move in
#this lets the AI evaluate and choose the best move at each step of the way while playing the game
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """Initializes the DQN model with the given sizes.

        Args:
            input_size (int): The size of the input state vector. The DQN will recieve this many (13) pieces of information about the game.
            hidden_size (int): The size of the hidden layer. The amount of "neurons" (128) that are processing the numbers from the input and outputting new numbers.
            output_size (int): The size of the output vector representing the q values. The DQN will output 4 numbers each representing the directions that the snake can take: up, down, left, right.
        """
        
        #set up everything needed for a PyTorch neural network by calling the parent class (nn.Module)
        super(DQN, self).__init__()

        # the 3 layers of this neural network are:
        # 1. input layer (13 inputs, 128 hidden neurons) - takes in the raw game state and does some of the initial processing
        # 2. hidden layer (128 hidden neurons, 128 hidden neurons) - does more processing and learns more complex patterns
        # 3. output layer (128 hidden neurons, 4 outputs) - outputs the Q-values for each of the 4 directions (up, down, left, right) the snake could move in

        #this creates the first fully connected (linear) layer of the neural network
        #it takes the input size (13) and the hidden size (128) and creates a layer that connects each input to each hidden neuron
        #it takes the input vector and transform it into a new vector of size hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)

        #this creates the second fully connected layer of the neural network
        #it takes the output of the first layer (128) and the hidden size (128) and creates a layer that connects each input to each hidden neuron
        #it takes the output from the first layer (128) and transforms it into another vector of the same size (128)
        #this adds more "processing power" to the network, allowing it to learn more complex patterns
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        #this creates the third (output) layer of the neural network
        #it takes the output of the second layer (128) and the output size (4) and creates a layer that connects each input to each output neuron
        #it takes the ouput from the second layer and transforms it into a vector of size output_size (4)
        #the output of this layer is the Q-values for each possible action
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Computes the forward pass of the DQN model.
        
        Args:
            x (torch.tensor): The input state vector.
        
        Returns:
            torch.tensor: The output vector representing the q values.
        """
        
        #gets the number of the dimensions of the input data and checks if it is equal to 1, if it is that means that the input data is 1-dimensional (like a simple list of numbers)
        if len(x.shape) == 1:
            #It adds an extra dimension to the input data, turning it into a 2-dimensional tensor which is expected by the model.
            x = x.unsqueeze(0)
        
        #processes the input data through the first layer of the model (with the 128 output neurons described in the hidden_size parameter)
        x = torch.relu(self.fc1(x))
        #processes the output of the first layer through the second layer of the model (with the 128 output neurons described in the hidden_size parameter)
        x = torch.relu(self.fc2(x))
        #processes the output of the second layer through the third layer of the model (with the 4 output neurons described in the output_size parameter) to get the movement values
        x = self.fc3(x)
        #returns the final 4 numbers describing the movement values
        return x

#used during gameplay to make the snake move in the best direction, but doesn't change the "brain" of the snake
def select_action(state, q_network, epsilon):
    """Selects an action based on an epsilon-greedy policy.

    Args:
        state (list): The state vector.
        q_network (DQN): The DQN model used to compute the q values.
        epsilon (float): The probability of taking a random action.

    Returns:
        int: The action to take.
    """
    
    #the epsilon variable holds the percent chance that it will take for the snake to take a random action out of the 4 directions it could move
    if random.random() < epsilon:
        #4 options for the snake to move in so there are 4 options for the random number to be
        return random.randint(0, 3)
    else:
        #if the random chance doesn't happen, then the snake will use its "brain" to make an actually smart decision
        #this converts the game state (simply a list of numbers) into a PyTorch tensor as a 32-bit floating point number
        state = torch.tensor(state, dtype=torch.float32)
        #this is a PyTorch feature that tells the model to not compute gradients for this operation
        #this is because we're simply making predictions, not training the model
        with torch.no_grad():
            #this runs the game state through the neural network to get the Q-values for each of the 4 directions
            q_values = q_network(state)
        #this returns the index of the direction with the highest Q-value; this will make the snake go in the best direction with the data aquired from the neural network
        return torch.argmax(q_values).item()

#this one is like the select_action function but it is used to update the "brain" of the snake with the gradients being actually used here to do so
#used in the pretrain.py file to update the "brain" of the snake in training to make it better at moving towards the fruits
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

    #if the memory is less than 128, then the function will return nothing and not do anything
    #it does that because it wouldn't have enough data to train the model effectively with, so few memories
    #before gaining enough memories the snake will just move randomly to gain memories and know what is working and helping and what doesn't
    if len(memory) < 128:
        return

    #take 128 tandom samples from experiences in the memory
    batch = memory.sample(128)

    #separates each experience into 5 separate lists: states, actions, rewards, next_states, and dones
    states, actions, rewards, next_states, dones = zip(*batch)

    #converts the 5 separate lists from the experiences into PyTorch tensors to be used for efficient computing
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long) #has whole numbers so it is a long integer instead of a float like the other tensors
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    #q_network(states) -> runs all 128 states through the network (128 sets of 4 Q-values/possible actions); actions.unsqueeze(1) -> adds a dimension to the actions tensor to match the dimensions of the q_values tensor
    #.gather(1, actions.unsqueeze(1)) -> for each experience, it selects the Q-values of the action that was taken; .squeeze(1) -> removes the extra dimension added to the actions tensor
    #all together -> for each experience, it selects the Q-values of the action that was taken
    #if experience 0 took action 2 (left), and Q-values were [1.2, -0.3, 2.8, 0.9] -> we extract 2.8 (the Q-value for action 2)
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    #next_states -> the next state of the snake after the action was taken (where the snake ended up after each action)
    #.max(1)[0] -> finds the maximum Q-value for each of the 128 next states (the best possible Q-value for each of the 128 next states)
    #target_network(next_states) -> runs all 128 next states through the target network (128 sets of 4 Q-values/possible actions) which is more stable than the main network
    #the main network is the network that we're actually training and updating
    #the target network is a stable copy of the main network used to compute targets for learning which is updated less frequently (used for finding what the Q-values "should" be)
    next_q_values = target_network(next_states).max(1)[0]

    #rewards -> the reward for the action that was just taken
    #0.99 -> the discount factor which is how much we value future rewards compared to immediate rewards
    #next_q_values -> the best possible Q-value for each of the 128 next states
    #dones -> a list of boolean values that tells us whether each game ended after taking the action
    #this is the Bellman equation which calulates what the Q-values "should" be
    targets = rewards + 0.99 * next_q_values * (1 - dones)

    #this uses the MSE (Mean Squared Error) equation to calculate the loss between the predicted Q-values and the target Q-values
    #equation: loss = 1/n * sum((predicted - target)^2)
    #the q_values are the predicted Q-values (gotten from the neural network which is trained to make good decisions which should be close to the target Q-values) and the targets are the target Q-values (what the Q-values should be;gotten from the Bellman equation)
    loss = nn.functional.mse_loss(q_values, targets)

    #this resets the gradients to 0 so that the gradients from the previous iteration don't affect the current iteration
    #PyTorch gets gradientds by defaul and without clearing, new gradients would be added to the old ones causing incorrect weight updates
    optimizer.zero_grad()

    #this calculates the gradients of the loss with respect to the weights of the neural network
    #it needs to calculate how much the weights should change to reduce the loss (lower the difference between the predicted and target values)
    #Forward pass: Input → Network → Output → Loss
    #Backward pass: Loss → Gradients for each weight
    #these gradients are numbers that tell us how much each weight in the neural network should change to reduce the loss/make predictions more accurate
    loss.backward()
    
    #this prevents the gradients from becoming too large which could cause the model to "explode" and not actually learn anything at all
    #with gradients being too large, the model will make huge weight updates, which could cause the network to become unstable and stop learning
    #this function calculates the total magnitude of all of the gradients and makes sure that it doesn't exceed 1.0 by scaling all of the gradients down proportionally (this keeps training stable)
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
    
    #this is the function that actually updates the network weights using the computed gradients
    #it takes the current weights and adjusts them by the amount specified by the gradients
    #equation: new weights = old weights - learning rate * gradients
    optimizer.step()