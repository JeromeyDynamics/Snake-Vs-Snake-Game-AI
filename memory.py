import random
from collections import deque
import pickle

class ReplayMemory:
    def __init__(self, capacity):
        """Initializes the ReplayMemory object with a given capacity.

        Parameters
        ----------
        capacity : int
            The maximum size of the memory buffer.
        """
        
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Adds a new experience to the memory buffer.

        Parameters
        ----------
        *args
            The experience tuple, which must be in the following order:
            state, action, reward, next_state, done.
        """
        
        self.memory.append(args)

    def sample(self, batch_size):
        """Randomly samples a batch of experiences from the memory buffer.

        Parameters
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns
        -------
        list
            A list of sampled experiences, each in the form of a tuple
            (state, action, reward, next_state, done).
        """

        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current number of experiences stored in the memory buffer.

        Returns
        -------
        int
            The number of experiences in the memory buffer.
        """

        return len(self.memory)
    
    def save(self, filename):
        """Saves the memory buffer to a file.
        
        Parameters
        ----------
        filename : str
            The filename to save the memory to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(list(self.memory), f)
    
    def load(self, filename):
        """Loads the memory buffer from a file.
        
        Parameters
        ----------
        filename : str
            The filename to load the memory from.
        """
        try:
            with open(filename, 'rb') as f:
                memory_list = pickle.load(f)
                self.memory = deque(memory_list, maxlen=self.memory.maxlen)
        except FileNotFoundError:
            print(f"Memory file {filename} not found. Starting with empty memory.")
