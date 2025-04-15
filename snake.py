import pygame
import random

class SnakeGame:
    def __init__(self):
        """
        Initializes the SnakeGame by setting up the pygame environment,
        creating the display screen, setting the window caption, initializing
        the clock and font, defining the grid size, and resetting the game state.
        """

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('AI Snake')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.grid_size = 20
        self.reset()
    
    def get_random_grid_position(self):
        """
        Returns a random grid position within the screen boundaries that is not 
        currently occupied by the snake. The position is calculated based on the 
        grid size, ensuring that the coordinates are aligned with the grid.
        """

        while True:
            grid_x = random.randint(0, (800 // self.grid_size) - 1) * self.grid_size
            grid_y = random.randint(0, (600 // self.grid_size) - 1) * self.grid_size
            if (grid_x, grid_y) not in self.snake_pos:
                return (grid_x, grid_y)

    def reset(self):
        """
        Resets the game state by resetting the snake position, generating a new
        apple position, setting the direction to right, and resetting the score.
        """
        self.snake_pos = [(100, 100)]
        self.apple_pos = self.get_random_grid_position()
        self.direction = 'right'
        self.score = 0

    def get_state(self):
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

        head_x, head_y = self.snake_pos[0]
        apple_x, apple_y = self.apple_pos
        direction_vec = [0, 0, 0, 0]
        if self.direction == 'right':
            direction_vec[0] = 1
        elif self.direction == 'left':
            direction_vec[1] = 1
        elif self.direction == 'up':
            direction_vec[2] = 1
        elif self.direction == 'down':
            direction_vec[3] = 1
        state = [
            head_x, head_y, apple_x, apple_y,
            direction_vec[0], direction_vec[1],
            direction_vec[2], direction_vec[3],
            int((head_x, head_y + self.grid_size) in self.snake_pos),  # danger up
            int((head_x, head_y - self.grid_size) in self.snake_pos),  # danger down
            int((head_x - self.grid_size, head_y) in self.snake_pos),  # danger left
            int((head_x + self.grid_size, head_y) in self.snake_pos)   # danger right
        ]
        return state

    def step(self, action):
        """
        Advances the game state by one step given the provided action.

        The action is an integer representing the direction the snake should
        move in, where 0 is right, 1 is left, 2 is up, and 3 is down.

        Returns:
            tuple: A tuple containing the new state, reward, and done flag.
        """
        if action == 0 and self.direction != 'left':
            self.direction = 'right'
        elif action == 1 and self.direction != 'right':
            self.direction = 'left'
        elif action == 2 and self.direction != 'down':
            self.direction = 'up'
        elif action == 3 and self.direction != 'up':
            self.direction = 'down'

        head_x, head_y = self.snake_pos[0]
        if self.direction == 'right':
            head_x += self.grid_size
        elif self.direction == 'left':
            head_x -= self.grid_size
        elif self.direction == 'up':
            head_y -= self.grid_size
        elif self.direction == 'down':
            head_y += self.grid_size

        new_head = (head_x, head_y)
        self.snake_pos.insert(0, new_head)

        reward = 0
        if self.snake_pos[0] == self.apple_pos:
            self.apple_pos = self.get_random_grid_position()
            self.score += 1
            reward = 1
        else:
            self.snake_pos.pop()

        done = head_x >= 800 or head_x < 0 or head_y >= 600 or head_y < 0 or self.snake_pos[0] in self.snake_pos[1:]
        reward = -1 if done else reward

        self.draw()
        self.clock.tick(8)

        return self.get_state(), reward, done

    def draw(self):
        """
        Renders the current game state onto the display screen.

        This includes clearing the screen, drawing the snake at its current position 
        with green rectangles, drawing the apple with a red rectangle, and displaying 
        the current score at the top-left corner. Finally, it updates the display 
        to reflect these changes.
        """

        self.screen.fill((0, 0, 0))
        for segment in self.snake_pos:
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(segment[0], segment[1], self.grid_size, self.grid_size))
        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.apple_pos[0], self.apple_pos[1], self.grid_size, self.grid_size))
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
