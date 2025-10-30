import os

#paths to all the snake player data files
FILES_TO_DELETE = [
    'data/memory1.pkl',      # Agent 1's replay memory
    'data/memory2.pkl',      # Agent 2's replay memory
    'data/snake_agent1.pth', # Agent 1's neural network weights
    'data/snake_agent2.pth', # Agent 2's neural network weights
    'info/ai_info.txt',      # Training statistics
    'data/training_state.pkl' # Training state (epsilon values, step count)
]

print("Resetting all snake player data...")

for file in FILES_TO_DELETE:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted {file}")
    else:
        print(f"{file} does not exist (already reset)")

print("\nAll snake data has been reset!")
print("Run pretrain.py to train the snakes from scratch, or run main.py to start with untrained snakes.") 