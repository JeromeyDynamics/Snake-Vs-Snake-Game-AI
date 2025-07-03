"""
Game state management and state representation for the Snake AI game.
"""

import math
import random
from .game_config import *


class GameState:
    """Manages the current state of the game."""
    
    def __init__(self):
        self.snake1_pos = [SNAKE1_START_POS]
        self.snake2_pos = [SNAKE2_START_POS]
        self.apple_pos = self._get_random_grid_position(exclude=self.snake1_pos + self.snake2_pos)
        self.direction1 = SNAKE1_START_DIRECTION
        self.direction2 = SNAKE2_START_DIRECTION
        self.score1 = 0
        self.score2 = 0
        self.done1 = False
        self.done2 = False
    
    def _get_random_grid_position(self, exclude=None):
        """
        Returns a random grid position within the screen boundaries that is not 
        currently occupied by the snake. The position is calculated based on the 
        grid size, ensuring that the coordinates are aligned with the grid.
        """
        if exclude is None:
            exclude = []
        while True:
            grid_x = random.randint(0, (SCREEN_WIDTH // GRID_SIZE) - 1) * GRID_SIZE
            grid_y = random.randint(0, (SCREEN_HEIGHT // GRID_SIZE) - 1) * GRID_SIZE
            if (grid_x, grid_y) not in exclude:
                return (grid_x, grid_y)
    
    def reset(self):
        """Resets the game state to initial conditions."""
        self.snake1_pos = [SNAKE1_START_POS]
        self.snake2_pos = [SNAKE2_START_POS]
        self.apple_pos = self._get_random_grid_position(exclude=self.snake1_pos + self.snake2_pos)
        self.direction1 = SNAKE1_START_DIRECTION
        self.direction2 = SNAKE2_START_DIRECTION
        self.score1 = getattr(self, 'score1', 0)
        self.score2 = getattr(self, 'score2', 0)
        self.done1 = False
        self.done2 = False
    
    def get_state(self, snake_num):
        """
        Returns the current state of the game as a list of integers.

        The state includes the snake's head position, apple position, 
        direction vector, and danger indicators. The direction vector 
        represents the current direction of the snake as a one-hot encoded 
        list with four elements corresponding to right, left, up, and down 
        respectively. The danger indicators are binary values representing 
        whether the next position in a given direction (up, down, left, 
        right) is occupied by the snake's body, indicating a potential 
        collision.

        Returns:
            list: A list containing the snake's head position (x, y), apple 
            position (x, y), direction vector, and danger indicators.
        """
        if snake_num == 1:
            head_x, head_y = self.snake1_pos[0]
            snake_pos = self.snake1_pos
            other_snake = self.snake2_pos
            direction = self.direction1
        else:
            head_x, head_y = self.snake2_pos[0]
            snake_pos = self.snake2_pos
            other_snake = self.snake1_pos
            direction = self.direction2
        
        apple_x, apple_y = self.apple_pos
        
        # Calculate distance to apple
        distance_to_apple = math.sqrt((head_x - apple_x)**2 + (head_y - apple_y)**2)
        
        # Direction vector
        direction_vec = [0, 0, 0, 0]
        if direction == 'right':
            direction_vec[0] = 1
        elif direction == 'left':
            direction_vec[1] = 1
        elif direction == 'up':
            direction_vec[2] = 1
        elif direction == 'down':
            direction_vec[3] = 1
        
        # Danger indicators (more detailed)
        danger_up = int((head_x, head_y - GRID_SIZE) in snake_pos or (head_x, head_y - GRID_SIZE) in other_snake or head_y - GRID_SIZE < 0)
        danger_down = int((head_x, head_y + GRID_SIZE) in snake_pos or (head_x, head_y + GRID_SIZE) in other_snake or head_y + GRID_SIZE >= SCREEN_HEIGHT)
        danger_left = int((head_x - GRID_SIZE, head_y) in snake_pos or (head_x - GRID_SIZE, head_y) in other_snake or head_x - GRID_SIZE < 0)
        danger_right = int((head_x + GRID_SIZE, head_y) in snake_pos or (head_x + GRID_SIZE, head_y) in other_snake or head_x + GRID_SIZE >= SCREEN_WIDTH)
        
        # Normalize positions to smaller values
        head_x_norm = head_x / SCREEN_WIDTH
        head_y_norm = head_y / SCREEN_HEIGHT
        apple_x_norm = apple_x / SCREEN_WIDTH
        apple_y_norm = apple_y / SCREEN_HEIGHT
        distance_norm = distance_to_apple / DISTANCE_NORMALIZATION
        
        state = [
            head_x_norm, head_y_norm, apple_x_norm, apple_y_norm, distance_norm,
            direction_vec[0], direction_vec[1], direction_vec[2], direction_vec[3],
            danger_up, danger_down, danger_left, danger_right
        ]
        return state 