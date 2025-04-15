import pygame
from snake import SnakeGame
from dqn import DQN, select_action, update_network
from memory import ReplayMemory
import numpy as np
import torch

# Initialize game and AI components
game = SnakeGame()
memory = ReplayMemory(10000)
q_network = DQN(12, 128, 4)
target_network = DQN(12, 128, 4)
target_network.load_state_dict(q_network.state_dict())
optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)
epsilon = 1.0

def main():
    global epsilon
    running = True
    while running:
        state = game.get_state()
        action = select_action(state, q_network, epsilon)
        next_state, reward, done = game.step(action)
        memory.push(state, action, reward, next_state, done)
        update_network(q_network, target_network, optimizer, memory)
        if epsilon > 0.01:
            epsilon *= 0.995
        
        if done:
            game.reset()
            
if __name__ == "__main__":
    main()
