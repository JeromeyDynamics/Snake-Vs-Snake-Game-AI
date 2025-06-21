import pygame
from snake import SnakeGame
from dqn import DQN, select_action, update_network
from memory import ReplayMemory
import numpy as np
import torch
import os
import pickle

# Initialize game and AI components

game = SnakeGame()

# Agent 1
memory1 = ReplayMemory(10000)
q_network1 = DQN(13, 128, 4)
target_network1 = DQN(13, 128, 4)
target_network1.load_state_dict(q_network1.state_dict())
optimizer1 = torch.optim.Adam(q_network1.parameters(), lr=0.0001)
epsilon1 = 1.0

# Agent 2
memory2 = ReplayMemory(10000)
q_network2 = DQN(13, 128, 4)
target_network2 = DQN(13, 128, 4)
target_network2.load_state_dict(q_network2.state_dict())
optimizer2 = torch.optim.Adam(q_network2.parameters(), lr=0.0001)
epsilon2 = 1.0

MAX_STEPS = 500
TARGET_UPDATE_FREQ = 100
UPDATE_FREQ = 4
step_count = 0
reward_history1 = []
reward_history2 = []

# Load pretrained models if available
if os.path.exists('snake_agent1.pth'):
    q_network1.load_state_dict(torch.load('snake_agent1.pth'))
    target_network1.load_state_dict(q_network1.state_dict())
    print('Loaded pretrained weights for Agent 1')
if os.path.exists('snake_agent2.pth'):
    q_network2.load_state_dict(torch.load('snake_agent2.pth'))
    target_network2.load_state_dict(q_network2.state_dict())
    print('Loaded pretrained weights for Agent 2')

# Load memory buffers if available
if os.path.exists('memory1.pkl'):
    memory1.load('memory1.pkl')
    print(f'Loaded memory for Agent 1 ({len(memory1)} experiences)')
if os.path.exists('memory2.pkl'):
    memory2.load('memory2.pkl')
    print(f'Loaded memory for Agent 2 ({len(memory2)} experiences)')

# Load training state if available
if os.path.exists('training_state.pkl'):
    with open('training_state.pkl', 'rb') as f:
        training_state = pickle.load(f)
    epsilon1 = training_state['epsilon1']
    epsilon2 = training_state['epsilon2']
    step_count = training_state['step_count']
    print(f'Loaded training state: epsilon1={epsilon1:.4f}, epsilon2={epsilon2:.4f}, step_count={step_count}')

def main():
    global epsilon1, epsilon2, step_count
    running = True
    episode = 0
    while running:
        steps = 0
        total_reward1 = 0
        total_reward2 = 0
        game.reset()
        done1 = False
        done2 = False
        while not (done1 or done2) and steps < MAX_STEPS and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            state1 = game.get_state(1)
            state2 = game.get_state(2)
            action1 = select_action(state1, q_network1, epsilon1)
            action2 = select_action(state2, q_network2, epsilon2)
            (next_state1, reward1, done1), (next_state2, reward2, done2) = game.step(action1, action2)
            memory1.push(state1, action1, reward1, next_state1, done1)
            memory2.push(state2, action2, reward2, next_state2, done2)
            
            if step_count % UPDATE_FREQ == 0:
                update_network(q_network1, target_network1, optimizer1, memory1)
                update_network(q_network2, target_network2, optimizer2, memory2)
            
            total_reward1 += reward1
            total_reward2 += reward2
            step_count += 1
            steps += 1
            if step_count % TARGET_UPDATE_FREQ == 0:
                target_network1.load_state_dict(q_network1.state_dict())
                target_network2.load_state_dict(q_network2.state_dict())
            if epsilon1 > 0.01:
                epsilon1 *= 0.9999
            if epsilon2 > 0.01:
                epsilon2 *= 0.9999
        reward_history1.append(total_reward1)
        reward_history2.append(total_reward2)
        episode += 1
        if episode % 100 == 0:
            avg_reward1 = sum(reward_history1[-100:]) / min(100, len(reward_history1))
            avg_reward2 = sum(reward_history2[-100:]) / min(100, len(reward_history2))
            print(f"Episode {episode} | AvgR1: {avg_reward1:.2f} | AvgR2: {avg_reward2:.2f}")

if __name__ == "__main__":
    main()
