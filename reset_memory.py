import os

# Paths to all the snake player data files
snake_files = [
    'memory1.pkl',      # Agent 1's replay memory
    'memory2.pkl',      # Agent 2's replay memory  
    'snake_agent1.pth', # Agent 1's neural network weights
    'snake_agent2.pth', # Agent 2's neural network weights
    'training_state.pkl' # Training state (epsilon values, step count)
]

print("Resetting all snake player data...")

for file in snake_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted {file}")
    else:
        print(f"{file} does not exist (already reset)")

print("\nAll snake data has been reset!")
print("Run pretrain.py to train the snakes from scratch, or run main.py to start with untrained snakes.") 