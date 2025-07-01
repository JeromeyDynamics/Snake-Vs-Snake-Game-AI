import random
from collections import deque
import pickle

#this class is a container that stores and manages the past game experiences of the AI, so that it can learn from them later
class ReplayMemory:
    def __init__(self, capacity):
        """Initializes the ReplayMemory object with a given capacity.

        Parameters
        ----------
        capacity : int
            The maximum size of the memory buffer.
        """
        
        #declaring a new public variable called memory in the ReplayMemory class
        #the maxlen=capacity is because the deque class __init__ function has other parameters that don't need to be set, but come before the maxlen parameter, so it has to be specified
        #creates a new deque object with the maximum size being capacity; this is a memory buffer that can store a limited number of experiences
        self.memory = deque(maxlen=capacity)

    #the *args means that the function can take any number of arguments, and they will be stored in a tuple called args
    #but in this case the tuple must have 5 elements, which is state, action, reward, next_state, and done
    def push(self, *args):
        """Adds a new experience to the memory buffer.

        Parameters
        ----------
        *args
            The experience tuple, which must be in the following order:
            state, action, reward, next_state, done.
        """
        
        #adds a new experience to the memory buffer with the args tuple
        #the elements in the args tuple were what the state was, what action it took, what reward it got, what the next state became, and whether the game ended
        #this lets the AI look back at the memory buffer to look back at past experiences to learn from them
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

        #this will randomly sample a batch of experiences from the memory
        #returns a list of tuples, each containing a state, action, reward, next_state, and done
        #the tuples are the experiences from the memory buffer with the amount being batch_size
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current number of experiences stored in the memory buffer.

        Returns
        -------
        int
            The number of experiences in the memory buffer.
        """

        #returns the current number of experiences in the memory buffer
        return len(self.memory)
    
    def save(self, filename):
        """Saves the memory buffer to a file.
        
        Parameters
        ----------
        filename : str
            The filename to save the memory to.
        """

        #saves the memory buffer to a pkl file
        #uses the pickle module to serialize the memory buffer and save it to a file
        #this lets you pause and resume training because it is saved to a file
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
            #attemps to open the file given by the filename parameter
            with open(filename, 'rb') as f:
                #loads the memory buffer from the file into this list variable
                memory_list = pickle.load(f)
                #this creates a new deque object from the memory_list variable which holds the experiences from the file
                #the max length of the deque object is the same as the current memory buffer so that the memory buffer can be swapped out without losing any experiences
                self.memory = deque(memory_list, maxlen=self.memory.maxlen)
        except FileNotFoundError:
            #if the file is not found where the filename parameter says it is, then it will give an error which will print the following message
            print(f"Memory file {filename} not found. Starting with empty memory.")
