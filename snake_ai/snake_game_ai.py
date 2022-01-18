import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
Point = namedtuple('Point', ['x', 'y'])
BLOCK_SIZE = 20
GAME_SPEED = 20
# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
NBLUE = (0, 0, 255)
LBLUE = (0, 100, 255)
BLACK = (0, 0, 0)

class Direction(Enum):
  RIGHT = 1
  LEFT = 2
  UP = 3
  DOWN = 4

class SnakeGameAI:
  def __init__(self, width=640, height=480):
    # Default size of game screen
    self.width = width
    self.height = height
    # Init display
    self.display = pygame.display.set_mode((self.width, self.height))
    pygame.display.set_caption('Snake')
    self.clock = pygame.time.Clock()
    self.reset_game_state()

  def _place_food(self):
    """
    Helper function to spawn a food in a random location
    """
    x = random.randint(0, (self.width - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
    y = random.randint(0, (self.height - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
    self.food = Point(x, y)

    if self.food in self.snake:
      self._place_food()

  def _update_ui(self):
    self.display.fill(BLACK)

    for point in self.snake:
      # Body design
      pygame.draw.rect(
        self.display,
        NBLUE,
        pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE)
      )
      pygame.draw.rect(
        self.display,
        LBLUE,
        pygame.Rect(point.x + 4, point.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8)
      )
    # Food
    pygame.draw.rect(
      self.display,
      RED,
      pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
    )
    score_text = font.render('Score: ' + str(self.score), True, WHITE)
    self.display.blit(score_text, [0, 0])
    pygame.display.flip()

  def _move(self, action):
    # Direction starts with right since self.direction is default right
    directions_clockwise = [
      Direction.RIGHT, Direction.DOWN,
      Direction.LEFT, Direction.UP
    ]
    directions_index = directions_clockwise.index(self.direction)

    # AI actions = [Straight, Right, Left]
    # Not sure why this method is used for navigating
    if np.array_equal(action, [0, 1, 0]):
      # [0, 1, 0] is making a right turn
      # Pick next index in directions_clockwise: R -> D -> L -> U
      new_direction = directions_clockwise[(directions_index + 1) % 4]
    elif np.array_equal(action, [0, 0, 1]):
      # [0, 0, 1] is making a left turn
      # Pick previous index in directions_clockwise: R -> U -> L -> D
      new_direction = directions_clockwise[(directions_index - 1) % 4]
    else:
      # [1, 0, 0] is going straight
      # No change in direction
      new_direction = directions_clockwise[directions_index]
    
    self.direction = new_direction
    x = self.snake_head.x
    y = self.snake_head.y
    
    if self.direction == Direction.RIGHT:
      x += BLOCK_SIZE
    elif self.direction == Direction.LEFT:
      x -= BLOCK_SIZE
    elif self.direction == Direction.UP:
      y -= BLOCK_SIZE
    elif self.direction == Direction.DOWN:
      y += BLOCK_SIZE

    self.snake_head = Point(x, y)

  def is_collision(self, pos=None):
    if pos is None:
      pos = self.snake_head
    # Check if it hits boundary
    if (pos.x > self.width - BLOCK_SIZE
        or pos.x < 0
        or pos.y > self.height - BLOCK_SIZE
        or pos.y < 0):
      return True
    # Check if it hits itself
    if pos in self.snake[1:]:
      return True
    return False

  def reset_game_state(self):
    # Init game state
    self.direction = Direction.RIGHT
    self.snake_head = Point(self.width/2, self.height/2)
    self.snake = [
      self.snake_head,
      Point(self.snake_head.x - BLOCK_SIZE, self.snake_head.y),
      Point(self.snake_head.x - BLOCK_SIZE*2, self.snake_head.y)
    ]
    self.score = 0
    self.food = None
    self._place_food()
    self.frame_iteration = 0

  def play_step(self, action):
    """
    Gameplay loop
    """
    self.frame_iteration += 1
    # Collect user input
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

    # Move
    self._move(action)
    self.snake.insert(0, self.snake_head)
    # Check if game is over
    reward = 0
    game_over = False
    if self.is_collision() or self.frame_iteration > 100*len(self.snake):
      game_over = True
      reward = -10
      return game_over, self.score, reward
    # Place new food or move
    if self.snake_head == self.food:
      self.frame_iteration = 0
      self.score += 1
      reward = 10
      self._place_food()
    else:
      self.snake.pop()
    # Update UI and clock
    self._update_ui()
    self.clock.tick(GAME_SPEED)
    # Return game_over and score
    return game_over, self.score, reward