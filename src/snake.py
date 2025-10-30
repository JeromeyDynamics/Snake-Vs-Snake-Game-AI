"""
Main SnakeGame class that orchestrates all game components.
"""

from .game_state import GameState
from .game_logic import GameLogic
from .game_renderer import GameRenderer


class SnakeGame:
    """
    Main game class that coordinates state management, game logic, and rendering.
    """
    
    def __init__(self, render=True):
        """
        Initializes the SnakeGame by setting up the game state, logic, and renderer.
        """
        self.game_state = GameState()
        self.game_logic = GameLogic(self.game_state)
        self.renderer = GameRenderer(render)
        self.render = render
    
    def reset(self):
        """Resets the game state."""
        self.game_state.reset()
    
    def get_state(self, snake_num):
        """Returns the current state for the specified snake."""
        return self.game_state.get_state(snake_num)
    
    def step(self, action1, action2):
        """
        Advances the game state by one step given the provided actions.

        The actions are integers representing the direction each snake should
        move in, where 0 is right, 1 is left, 2 is up, and 3 is down.

        Returns:
            tuple: A tuple containing the new states, rewards, and done flags for both snakes.
        """
        #updates directions
        self.game_logic.update_directions(action1, action2)
        
        #moves snakes
        old_head1, old_head2, new_head1, new_head2 = self.game_logic.move_snakes()
        
        #handles apple collection
        reward1, reward2, grow1, grow2 = self.game_logic.handle_apple_collection(new_head1, new_head2)
        
        #calculates distance-based rewards
        reward1, reward2 = self.game_logic.calculate_distance_rewards(
            old_head1, old_head2, new_head1, new_head2, reward1, reward2
        )
        
        #handles the growth of the snake
        self.game_logic.handle_snake_growth(grow1, grow2)
        
        #detects and handles collisions with the snakes
        collisions = self.game_logic.detect_collisions(new_head1, new_head2)
        reward1, reward2 = self.game_logic.handle_collisions(collisions, reward1, reward2)
        
        #renders if it is enabled
        if self.render:
            self.renderer.draw(self.game_state)
            self.renderer.tick()
        
        #gets the final states
        state1 = self.get_state(1)
        state2 = self.get_state(2)
        
        return (state1, reward1, self.game_state.done1), (state2, reward2, self.game_state.done2)

    @property
    def score1(self):
        return self.game_state.score1

    @property
    def score2(self):
        return self.game_state.score2 