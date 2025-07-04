#the __init__.py file is used to make the snake package a module
#the code in this file runs when the package is imported

#this line imports the SnakeGame class from the snake.py file in this directory
#this means that the SnakeGame class is available directly when you import this package
from .snake import SnakeGame

#this line makes the SnakeGame class available when you import the package
#__all__ is a list of public objects that are exported when the package is imported
#in this case, it makes the SnakeGame class available when you import the package
#ex: "from src import *" will only make the SnakeGame class available
__all__ = ['SnakeGame']