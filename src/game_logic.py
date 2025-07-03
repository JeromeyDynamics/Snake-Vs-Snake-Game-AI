"""
Game logic for movement, collision detection, and reward calculation.
"""

import math
from .game_config import *


class GameLogic:
    """Handles the core game logic including movement, collisions, and rewards."""
    
    def __init__(self, game_state):
        self.game_state = game_state
    
    def update_directions(self, action1, action2):
        """Updates snake directions based on actions."""
        # Update direction for snake 1
        if action1 == 0 and self.game_state.direction1 != 'left':
            self.game_state.direction1 = 'right'
        elif action1 == 1 and self.game_state.direction1 != 'right':
            self.game_state.direction1 = 'left'
        elif action1 == 2 and self.game_state.direction1 != 'down':
            self.game_state.direction1 = 'up'
        elif action1 == 3 and self.game_state.direction1 != 'up':
            self.game_state.direction1 = 'down'

        # Update direction for snake 2
        if action2 == 0 and self.game_state.direction2 != 'left':
            self.game_state.direction2 = 'right'
        elif action2 == 1 and self.game_state.direction2 != 'right':
            self.game_state.direction2 = 'left'
        elif action2 == 2 and self.game_state.direction2 != 'down':
            self.game_state.direction2 = 'up'
        elif action2 == 3 and self.game_state.direction2 != 'up':
            self.game_state.direction2 = 'down'
    
    def move_snakes(self):
        """Moves both snakes based on their current directions."""
        # Store old positions for distance calculation
        old_head1 = self.game_state.snake1_pos[0]
        old_head2 = self.game_state.snake2_pos[0]

        # Move snake 1
        head1_x, head1_y = self.game_state.snake1_pos[0]
        if self.game_state.direction1 == 'right':
            head1_x += GRID_SIZE
        elif self.game_state.direction1 == 'left':
            head1_x -= GRID_SIZE
        elif self.game_state.direction1 == 'up':
            head1_y -= GRID_SIZE
        elif self.game_state.direction1 == 'down':
            head1_y += GRID_SIZE
        new_head1 = (head1_x, head1_y)

        # Move snake 2
        head2_x, head2_y = self.game_state.snake2_pos[0]
        if self.game_state.direction2 == 'right':
            head2_x += GRID_SIZE
        elif self.game_state.direction2 == 'left':
            head2_x -= GRID_SIZE
        elif self.game_state.direction2 == 'up':
            head2_y -= GRID_SIZE
        elif self.game_state.direction2 == 'down':
            head2_y += GRID_SIZE
        new_head2 = (head2_x, head2_y)

        self.game_state.snake1_pos.insert(0, new_head1)
        self.game_state.snake2_pos.insert(0, new_head2)
        
        return old_head1, old_head2, new_head1, new_head2
    
    def handle_apple_collection(self, new_head1, new_head2):
        """Handles apple collection and returns rewards and growth flags."""
        reward1 = REWARD_STEP
        reward2 = REWARD_STEP
        grow1 = False
        grow2 = False

        # Apple collection with normalized rewards
        both_on_apple = new_head1 == self.game_state.apple_pos and new_head2 == self.game_state.apple_pos
        if both_on_apple:
            self.game_state.apple_pos = self.game_state._get_random_grid_position(
                exclude=self.game_state.snake1_pos + self.game_state.snake2_pos
            )
            self.game_state.score1 += 1
            self.game_state.score2 += 1
            reward1 = REWARD_APPLE_BOTH
            reward2 = REWARD_APPLE_BOTH
            grow1 = True
            grow2 = True
        elif new_head1 == self.game_state.apple_pos:
            self.game_state.apple_pos = self.game_state._get_random_grid_position(
                exclude=self.game_state.snake1_pos + self.game_state.snake2_pos
            )
            self.game_state.score1 += 1
            reward1 = REWARD_APPLE_INDIVIDUAL
            grow1 = True
        elif new_head2 == self.game_state.apple_pos:
            self.game_state.apple_pos = self.game_state._get_random_grid_position(
                exclude=self.game_state.snake1_pos + self.game_state.snake2_pos
            )
            self.game_state.score2 += 1
            reward2 = REWARD_APPLE_INDIVIDUAL
            grow2 = True
        
        return reward1, reward2, grow1, grow2
    
    def calculate_distance_rewards(self, old_head1, old_head2, new_head1, new_head2, reward1, reward2):
        """Calculates additional rewards based on distance to apple."""
        # Small reward for moving closer to food
        old_dist1 = math.sqrt((old_head1[0] - self.game_state.apple_pos[0])**2 + (old_head1[1] - self.game_state.apple_pos[1])**2)
        new_dist1 = math.sqrt((new_head1[0] - self.game_state.apple_pos[0])**2 + (new_head1[1] - self.game_state.apple_pos[1])**2)
        if new_dist1 < old_dist1:
            reward1 += REWARD_CLOSER_TO_APPLE
        
        old_dist2 = math.sqrt((old_head2[0] - self.game_state.apple_pos[0])**2 + (old_head2[1] - self.game_state.apple_pos[1])**2)
        new_dist2 = math.sqrt((new_head2[0] - self.game_state.apple_pos[0])**2 + (new_head2[1] - self.game_state.apple_pos[1])**2)
        if new_dist2 < old_dist2:
            reward2 += REWARD_CLOSER_TO_APPLE
        
        return reward1, reward2
    
    def handle_snake_growth(self, grow1, grow2):
        """Handles snake growth by removing tails if not growing."""
        # Only grow if ate apple, otherwise pop tail
        if not grow1:
            self.game_state.snake1_pos.pop()
        if not grow2:
            self.game_state.snake2_pos.pop()
    
    def detect_collisions(self, new_head1, new_head2):
        """Detects all types of collisions and returns collision information."""
        head1_in_body2 = new_head1 in self.game_state.snake2_pos[1:]
        head2_in_body1 = new_head2 in self.game_state.snake1_pos[1:]
        head1_in_self = new_head1 in self.game_state.snake1_pos[1:]
        head2_in_self = new_head2 in self.game_state.snake2_pos[1:]
        out1 = new_head1[0] >= SCREEN_WIDTH or new_head1[0] < 0 or new_head1[1] >= SCREEN_HEIGHT or new_head1[1] < 0
        out2 = new_head2[0] >= SCREEN_WIDTH or new_head2[0] < 0 or new_head2[1] >= SCREEN_HEIGHT or new_head2[1] < 0
        heads_collide = new_head1 == new_head2
        
        return {
            'head1_in_body2': head1_in_body2,
            'head2_in_body1': head2_in_body1,
            'head1_in_self': head1_in_self,
            'head2_in_self': head2_in_self,
            'out1': out1,
            'out2': out2,
            'heads_collide': heads_collide
        }
    
    def handle_collisions(self, collisions, reward1, reward2):
        """Handles collision outcomes and updates rewards and game state."""
        self.game_state.done1 = False
        self.game_state.done2 = False

        # If both heads collide, both get a point and reset
        if collisions['heads_collide']:
            self.game_state.score1 += 1
            self.game_state.score2 += 1
            reward1 = REWARD_HEAD_COLLISION
            reward2 = REWARD_HEAD_COLLISION
            self.game_state.done1 = True
            self.game_state.done2 = True
        # If snake 1 hits snake 2's body, snake 1 loses, snake 2 gets a point
        elif collisions['head1_in_body2'] or collisions['head1_in_self'] or collisions['out1']:
            self.game_state.score2 += 1
            reward1 = REWARD_DEATH
            reward2 = REWARD_WIN
            self.game_state.done1 = True
            self.game_state.done2 = True
        # If snake 2 hits snake 1's body, snake 2 loses, snake 1 gets a point
        elif collisions['head2_in_body1'] or collisions['head2_in_self'] or collisions['out2']:
            self.game_state.score1 += 1
            reward1 = REWARD_WIN
            reward2 = REWARD_DEATH
            self.game_state.done1 = True
            self.game_state.done2 = True
        
        return reward1, reward2 