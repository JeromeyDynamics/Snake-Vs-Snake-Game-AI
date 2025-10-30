import torch
from src import SnakeGame
from src.dqn import DQN, select_action, update_network
from src.memory import ReplayMemory
import time
import pickle

#the pretraining parameters
EPISODES = 2000  #increased for better learning
MAX_STEPS = 200  #reduced for faster episodes
TARGET_UPDATE_FREQ = 200  #less frequent target updates
UPDATE_FREQ = 16  #update every 16 steps for much faster training
BATCH_SIZE = 128  #larger batch size for efficiency
SAVE_PATH1 = 'data/snake_agent1.pth'
SAVE_PATH2 = 'data/snake_agent2.pth'
MEMORY_PATH1 = 'data/memory1.pkl'
MEMORY_PATH2 = 'data/memory2.pkl'
TRAINING_STATE_PATH = 'data/training_state.pkl'
INFO_PATH = 'info/ai_info.txt'

#the initial game and AI components
game = SnakeGame(render=False)
memory1 = ReplayMemory(10000)
q_network1 = DQN(13, 128, 4)  #smaller hidden layer for better generalization
target_network1 = DQN(13, 128, 4)
target_network1.load_state_dict(q_network1.state_dict())
optimizer1 = torch.optim.Adam(q_network1.parameters(), lr=0.0001)  #lower learning rate for stability
epsilon1 = 1.0
memory2 = ReplayMemory(10000)
q_network2 = DQN(13, 128, 4)  #smaller hidden layer for better generalization
target_network2 = DQN(13, 128, 4)
target_network2.load_state_dict(q_network2.state_dict())
optimizer2 = torch.optim.Adam(q_network2.parameters(), lr=0.0001)  #lower learning rate for stability
epsilon2 = 1.0

#overwrites ai_info.txt at the start of the simulation
with open(INFO_PATH, 'w') as f:
    f.write('episode,score1,score2,avg_reward1,avg_reward2\n')

step_count = 0
reward_history1 = []
reward_history2 = []
episode_lengths = []
start_time = time.time()

for episode in range(EPISODES):
    game.reset()
    done1 = False
    done2 = False
    start_score1 = game.score1
    start_score2 = game.score2
    total_reward1 = 0
    total_reward2 = 0
    steps = 0
    while not (done1 or done2) and steps < MAX_STEPS:
        state1 = game.get_state(1)
        state2 = game.get_state(2)
        action1 = select_action(state1, q_network1, epsilon1)
        action2 = select_action(state2, q_network2, epsilon2)
        (next_state1, reward1, done1), (next_state2, reward2, done2) = game.step(action1, action2)
        memory1.push(state1, action1, reward1, next_state1, done1)
        memory2.push(state2, action2, reward2, next_state2, done2)
        
        #only updates networks every UPDATE_FREQ steps to speed up the training process
        if step_count % UPDATE_FREQ == 0:
            update_network(q_network1, target_network1, optimizer1, memory1)
            update_network(q_network2, target_network2, optimizer2, memory2)
        
        total_reward1 += reward1
        total_reward2 += reward2
        step_count += 1
        steps += 1
        
        #periodically updates target networks
        if step_count % TARGET_UPDATE_FREQ == 0:
            target_network1.load_state_dict(q_network1.state_dict())
            target_network2.load_state_dict(q_network2.state_dict())
        
        #lowers epsilon decay
        if epsilon1 > 0.01:
            epsilon1 *= 0.9999
        if epsilon2 > 0.01:
            epsilon2 *= 0.9999
    
    episode_lengths.append(steps)
    reward_history1.append(total_reward1)
    reward_history2.append(total_reward2)
    
    #writes stats every 100 episodes
    if (episode+1) % 100 == 0 or episode == EPISODES-1:
        score1 = game.score1 - start_score1
        score2 = game.score2 - start_score2
        avg_reward1 = sum(reward_history1[-100:]) / min(100, len(reward_history1))
        avg_reward2 = sum(reward_history2[-100:]) / min(100, len(reward_history2))
        avg_length = sum(episode_lengths[-100:]) / min(100, len(episode_lengths))
        elapsed_time = time.time() - start_time
        episodes_per_second = (episode + 1) / elapsed_time
        memory_size1 = len(memory1)
        memory_size2 = len(memory2)
        with open(INFO_PATH, 'a') as f:
            f.write(f"{episode+1},{score1},{score2},{avg_reward1:.2f},{avg_reward2:.2f}\n")
        print(f"Episode {episode+1}/{EPISODES} | Score1: {score1} | Score2: {score2} | AvgR1: {avg_reward1:.2f} | AvgR2: {avg_reward2:.2f} | AvgLen: {avg_length:.1f} | Speed: {episodes_per_second:.1f} ep/s | Mem: {memory_size1}/{memory_size2}")

#saves the trained models
torch.save(q_network1.state_dict(), SAVE_PATH1)
torch.save(q_network2.state_dict(), SAVE_PATH2)

#saves the memory buffers
memory1.save(MEMORY_PATH1)
memory2.save(MEMORY_PATH2)

#saves the training state (from the epsilon values and step count)
training_state = {
    'epsilon1': epsilon1,
    'epsilon2': epsilon2,
    'step_count': step_count
}
with open(TRAINING_STATE_PATH, 'wb') as f:
    pickle.dump(training_state, f)

total_time = time.time() - start_time
print(f"Pretraining complete in {total_time:.1f} seconds. Models, memory, and training state saved.")
