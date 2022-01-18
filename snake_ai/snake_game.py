import pygame
import random
from enum import Enum
from collections import namedtuple

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

class SnakeGame:
  def __init__(self, width=640, height=480):
    # Default size of game screen
    self.width = width
    self.height = height
    # Init display
    self.display = pygame.display.set_mode((self.width, self.height))
    pygame.display.set_caption('Snake')
    self.clock = pygame.time.Clock()

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

  def _move(self, direction):
    x = self.snake_head.x
    y = self.snake_head.y
    if direction == Direction.RIGHT:
      x += BLOCK_SIZE
    elif direction == Direction.LEFT:
      x -= BLOCK_SIZE
    elif direction == Direction.UP:
      y -= BLOCK_SIZE
    elif direction == Direction.DOWN:
      y += BLOCK_SIZE

    self.snake_head = Point(x, y)

  def _is_collision(self):
    # Check if it hits boundary
    if (self.snake_head.x > self.width - BLOCK_SIZE
        or self.snake_head.x < 0
        or self.snake_head.y > self.height - BLOCK_SIZE
        or self.snake_head.y < 0):
      return True
    # Check if it hits itself
    if self.snake_head in self.snake[1:]:
      return True
    return False

  def play_step(self):
    """
    Gameplay loop
    """
    # Collect user input
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_RIGHT:
          self.direction = Direction.RIGHT
        elif event.key == pygame.K_LEFT:
          self.direction = Direction.LEFT
        elif event.key == pygame.K_UP:
          self.direction = Direction.UP
        elif event.key == pygame.K_DOWN:
          self.direction = Direction.DOWN
    # Move
    self._move(self.direction)
    self.snake.insert(0, self.snake_head)
    # Check if game is over
    game_over = False
    if self._is_collision():
      game_over = True
      return game_over, self.score
    # Place new food or move
    if self.snake_head == self.food:
      self.score += 1
      self._place_food()
    else:
      self.snake.pop()
    # Update UI and clock
    self._update_ui()
    self.clock.tick(GAME_SPEED)
    # Return game_over and score
    return game_over, self.score

def main():
  game = SnakeGame()

  while True:
    game_over, score = game.play_step()

    if game_over:
      break

  print('Final Score', score)
  pygame.quit()

if __name__ == '__main__':
  main()