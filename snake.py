import pygame
import random
import math

class SnakeGame:
    def __init__(self, render=True):
        """
        Initializes the SnakeGame by setting up the pygame environment,
        creating the display screen, setting the window caption, initializing
        the clock and font, defining the grid size, and resetting the game state.
        """

        self.render = render
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption('AI Snake - Two Player')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        self.grid_size = 20
        self.reset()
    
    def get_random_grid_position(self, exclude=None):
        """
        Returns a random grid position within the screen boundaries that is not 
        currently occupied by the snake. The position is calculated based on the 
        grid size, ensuring that the coordinates are aligned with the grid.
        """

        if exclude is None:
            exclude = []
        while True:
            grid_x = random.randint(0, (800 // self.grid_size) - 1) * self.grid_size
            grid_y = random.randint(0, (600 // self.grid_size) - 1) * self.grid_size
            if (grid_x, grid_y) not in exclude:
                return (grid_x, grid_y)

    def reset(self):
        """
        Resets the game state by resetting the snake position, generating a new
        apple position, setting the direction to right, and resetting the score.
        """
        self.snake1_pos = [(100, 100)]
        self.snake2_pos = [(700, 500)]
        self.apple_pos = self.get_random_grid_position(exclude=self.snake1_pos + self.snake2_pos)
        self.direction1 = 'right'
        self.direction2 = 'left'
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
        danger_up = int((head_x, head_y - self.grid_size) in snake_pos or (head_x, head_y - self.grid_size) in other_snake or head_y - self.grid_size < 0)
        danger_down = int((head_x, head_y + self.grid_size) in snake_pos or (head_x, head_y + self.grid_size) in other_snake or head_y + self.grid_size >= 600)
        danger_left = int((head_x - self.grid_size, head_y) in snake_pos or (head_x - self.grid_size, head_y) in other_snake or head_x - self.grid_size < 0)
        danger_right = int((head_x + self.grid_size, head_y) in snake_pos or (head_x + self.grid_size, head_y) in other_snake or head_x + self.grid_size >= 800)
        
        # Normalize positions to smaller values
        head_x_norm = head_x / 800.0
        head_y_norm = head_y / 600.0
        apple_x_norm = apple_x / 800.0
        apple_y_norm = apple_y / 600.0
        distance_norm = distance_to_apple / 1000.0  # Normalize distance
        
        state = [
            head_x_norm, head_y_norm, apple_x_norm, apple_y_norm, distance_norm,
            direction_vec[0], direction_vec[1], direction_vec[2], direction_vec[3],
            danger_up, danger_down, danger_left, danger_right
        ]
        return state

    def step(self, action1, action2):
        """
        Advances the game state by one step given the provided action.

        The action is an integer representing the direction the snake should
        move in, where 0 is right, 1 is left, 2 is up, and 3 is down.

        Returns:
            tuple: A tuple containing the new state, reward, and done flag.
        """
        # Update directions
        if action1 == 0 and self.direction1 != 'left':
            self.direction1 = 'right'
        elif action1 == 1 and self.direction1 != 'right':
            self.direction1 = 'left'
        elif action1 == 2 and self.direction1 != 'down':
            self.direction1 = 'up'
        elif action1 == 3 and self.direction1 != 'up':
            self.direction1 = 'down'

        if action2 == 0 and self.direction2 != 'left':
            self.direction2 = 'right'
        elif action2 == 1 and self.direction2 != 'right':
            self.direction2 = 'left'
        elif action2 == 2 and self.direction2 != 'down':
            self.direction2 = 'up'
        elif action2 == 3 and self.direction2 != 'up':
            self.direction2 = 'down'

        # Store old positions for distance calculation
        old_head1 = self.snake1_pos[0]
        old_head2 = self.snake2_pos[0]

        # Move snakes
        head1_x, head1_y = self.snake1_pos[0]
        head2_x, head2_y = self.snake2_pos[0]
        if self.direction1 == 'right':
            head1_x += self.grid_size
        elif self.direction1 == 'left':
            head1_x -= self.grid_size
        elif self.direction1 == 'up':
            head1_y -= self.grid_size
        elif self.direction1 == 'down':
            head1_y += self.grid_size
        new_head1 = (head1_x, head1_y)

        if self.direction2 == 'right':
            head2_x += self.grid_size
        elif self.direction2 == 'left':
            head2_x -= self.grid_size
        elif self.direction2 == 'up':
            head2_y -= self.grid_size
        elif self.direction2 == 'down':
            head2_y += self.grid_size
        new_head2 = (head2_x, head2_y)

        self.snake1_pos.insert(0, new_head1)
        self.snake2_pos.insert(0, new_head2)

        reward1 = -0.01  # Small negative reward for each step
        reward2 = -0.01
        grow1 = False
        grow2 = False

        # Apple collection with normalized rewards
        both_on_apple = new_head1 == self.apple_pos and new_head2 == self.apple_pos
        if both_on_apple:
            self.apple_pos = self.get_random_grid_position(exclude=self.snake1_pos + self.snake2_pos)
            self.score1 += 1
            self.score2 += 1
            reward1 = 0.5  # Normalized reward for getting food (even if both get it)
            reward2 = 0.5
            grow1 = True
            grow2 = True
        elif new_head1 == self.apple_pos:
            self.apple_pos = self.get_random_grid_position(exclude=self.snake1_pos + self.snake2_pos)
            self.score1 += 1
            reward1 = 1.0  # Normalized reward for getting food individually
            grow1 = True
        elif new_head2 == self.apple_pos:
            self.apple_pos = self.get_random_grid_position(exclude=self.snake1_pos + self.snake2_pos)
            self.score2 += 1
            reward2 = 1.0  # Normalized reward for getting food individually
            grow2 = True
        else:
            # Small reward for moving closer to food
            old_dist1 = math.sqrt((old_head1[0] - self.apple_pos[0])**2 + (old_head1[1] - self.apple_pos[1])**2)
            new_dist1 = math.sqrt((new_head1[0] - self.apple_pos[0])**2 + (new_head1[1] - self.apple_pos[1])**2)
            if new_dist1 < old_dist1:
                reward1 += 0.02  # Smaller normalized reward for moving closer to food
            
            old_dist2 = math.sqrt((old_head2[0] - self.apple_pos[0])**2 + (old_head2[1] - self.apple_pos[1])**2)
            new_dist2 = math.sqrt((new_head2[0] - self.apple_pos[0])**2 + (new_head2[1] - self.apple_pos[1])**2)
            if new_dist2 < old_dist2:
                reward2 += 0.02

        # Only grow if ate apple, otherwise pop tail
        if not grow1:
            self.snake1_pos.pop()
        if not grow2:
            self.snake2_pos.pop()

        # Collision detection
        head1_in_body2 = new_head1 in self.snake2_pos[1:]
        head2_in_body1 = new_head2 in self.snake1_pos[1:]
        head1_in_self = new_head1 in self.snake1_pos[1:]
        head2_in_self = new_head2 in self.snake2_pos[1:]
        out1 = head1_x >= 800 or head1_x < 0 or head1_y >= 600 or head1_y < 0
        out2 = head2_x >= 800 or head2_x < 0 or head2_y >= 600 or head2_y < 0
        heads_collide = new_head1 == new_head2

        self.done1 = False
        self.done2 = False

        # If both heads collide, both get a point and reset
        if heads_collide:
            self.score1 += 1
            self.score2 += 1
            reward1 = 0.1  # Small neutral reward for collision (both survive)
            reward2 = 0.1
            self.done1 = True
            self.done2 = True
        # If snake 1 hits snake 2's body, snake 1 loses, snake 2 gets a point
        elif head1_in_body2 or head1_in_self or out1:
            self.score2 += 1
            reward1 = -1.0  # Normalized negative reward for death
            reward2 = 0.3   # Normalized positive reward for winning
            self.done1 = True
            self.done2 = True
        # If snake 2 hits snake 1's body, snake 2 loses, snake 1 gets a point
        elif head2_in_body1 or head2_in_self or out2:
            self.score1 += 1
            reward1 = 0.3   # Normalized positive reward for winning
            reward2 = -1.0  # Normalized negative reward for death
            self.done1 = True
            self.done2 = True

        if self.render:
            self.draw()
            self.clock.tick(8)

        state1 = self.get_state(1)
        state2 = self.get_state(2)
        return (state1, reward1, self.done1), (state2, reward2, self.done2)

    def draw(self):
        """
        Renders the current game state onto the display screen.

        This includes clearing the screen, drawing the snakes at their current positions 
        with different colors, drawing the apple with a red rectangle, and displaying 
        the current scores at the top-left and top-right corners. Finally, it updates the display 
        to reflect these changes.
        """

        if not self.render:
            return
        self.screen.fill((0, 0, 0))
        # Snake 1: Green
        for segment in self.snake1_pos:
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(segment[0], segment[1], self.grid_size, self.grid_size))
        # Snake 2: Blue
        for segment in self.snake2_pos:
            pygame.draw.rect(self.screen, (0, 128, 255), pygame.Rect(segment[0], segment[1], self.grid_size, self.grid_size))
        # Apple: Red
        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.apple_pos[0], self.apple_pos[1], self.grid_size, self.grid_size))
        # Scores
        score1_text = self.font.render(f'P1 Score: {self.score1}', True, (0, 255, 0))
        score2_text = self.font.render(f'P2 Score: {self.score2}', True, (0, 128, 255))
        self.screen.blit(score1_text, (10, 10))
        self.screen.blit(score2_text, (650, 10))
        pygame.display.flip()
