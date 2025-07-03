"""
Game configuration constants for the Snake AI game.
"""

# Display settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20
FPS = 8

# Colors (RGB)
SNAKE1_COLOR = (0, 255, 0)      # Green
SNAKE2_COLOR = (0, 128, 255)    # Blue
APPLE_COLOR = (255, 0, 0)       # Red
BACKGROUND_COLOR = (0, 0, 0)    # Black

# Initial positions
SNAKE1_START_POS = (100, 100)
SNAKE2_START_POS = (700, 500)

# Initial directions
SNAKE1_START_DIRECTION = 'right'
SNAKE2_START_DIRECTION = 'left'

# Reward values
REWARD_STEP = -0.01
REWARD_APPLE_INDIVIDUAL = 1.0
REWARD_APPLE_BOTH = 0.5
REWARD_CLOSER_TO_APPLE = 0.02
REWARD_HEAD_COLLISION = 0.1
REWARD_WIN = 0.3
REWARD_DEATH = -1.0

# State normalization
DISTANCE_NORMALIZATION = 1000.0 