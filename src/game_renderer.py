"""
Pygame renderer for the Snake AI game.
"""

import pygame
from .game_config import *


class GameRenderer:
    """Handles all rendering functionality for the game."""
    
    def __init__(self, render=True):
        self.render = render
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('AI Snake - Two Player')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
    
    def draw(self, game_state):
        """
        Renders the current game state onto the display screen.

        This includes clearing the screen, drawing the snakes at their current positions 
        with different colors, drawing the apple with a red rectangle, and displaying 
        the current scores at the top-left and top-right corners. Finally, it updates the display 
        to reflect these changes.
        """
        if not self.render:
            return
        
        self.screen.fill(BACKGROUND_COLOR)
        
        # Snake 1: Green
        for segment in game_state.snake1_pos:
            pygame.draw.rect(self.screen, SNAKE1_COLOR, 
                           pygame.Rect(segment[0], segment[1], GRID_SIZE, GRID_SIZE))
        
        # Snake 2: Blue
        for segment in game_state.snake2_pos:
            pygame.draw.rect(self.screen, SNAKE2_COLOR, 
                           pygame.Rect(segment[0], segment[1], GRID_SIZE, GRID_SIZE))
        
        # Apple: Red
        pygame.draw.rect(self.screen, APPLE_COLOR, 
                        pygame.Rect(game_state.apple_pos[0], game_state.apple_pos[1], GRID_SIZE, GRID_SIZE))
        
        # Scores
        score1_text = self.font.render(f'P1 Score: {game_state.score1}', True, SNAKE1_COLOR)
        score2_text = self.font.render(f'P2 Score: {game_state.score2}', True, SNAKE2_COLOR)
        self.screen.blit(score1_text, (10, 10))
        self.screen.blit(score2_text, (650, 10))
        
        pygame.display.flip()
    
    def tick(self):
        """Advances the game clock by one frame."""
        if self.render:
            self.clock.tick(FPS) 