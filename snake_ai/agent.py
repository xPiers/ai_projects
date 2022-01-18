import torch
import random
import numpy as np
from collections import deque
from snake_game_ai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from plotter import plot
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
  def __init__(self):
    self.num_games = 0
    self.epsilon = 0 # Randomness
    self.gamma = 0.9 # Discount rate, arbitrarily chosen, but must be less than 1
    self.memory = deque(maxlen=MAX_MEMORY) # pops opposite side when full, in this case, left most value
    # 11 states based on states list below, 256 hidden layer nodes arbitrarily chosen, 3 output because [straight, left, right] is movement
    self.model = Linear_QNet(11, 256, 3)
    self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

  def get_state(self, game):
    snake_head = game.snake[0]
    # Area to the direction of the head
    left_face = Point(snake_head.x - BLOCK_SIZE, snake_head.y)
    right_face = Point(snake_head.x + BLOCK_SIZE, snake_head.y)
    # Inverted because positions start from the top left corner being [0, 0]
    upper_face = Point(snake_head.x, snake_head.y - BLOCK_SIZE)
    lower_face = Point(snake_head.x, snake_head.y + BLOCK_SIZE)
    # Get current heading
    direction_left = game.direction == Direction.LEFT
    direction_right = game.direction == Direction.RIGHT
    direction_up = game.direction == Direction.UP
    direction_down = game.direction == Direction.DOWN
    # Game state of snake
    state = [
      # Danger straight ahead of snake
      (direction_left and game.is_collision(left_face))
      or (direction_right and game.is_collision(right_face))
      or (direction_up and game.is_collision(upper_face))
      or (direction_down and game.is_collision(lower_face)),
      # Danger to the right of snake
      (direction_left and game.is_collision(upper_face))
      or (direction_right and game.is_collision(lower_face))
      or (direction_up and game.is_collision(right_face))
      or (direction_down and game.is_collision(left_face)),
      # Danger to the left of snake
      (direction_left and game.is_collision(lower_face))
      or (direction_right and game.is_collision(upper_face))
      or (direction_up and game.is_collision(left_face))
      or (direction_down and game.is_collision(right_face)),
      # Current heading in list [left, right, up, down]
      # Only one will be true at any given point
      direction_left,
      direction_right,
      direction_up,
      direction_down,
      # Location of food relative to snake [left, right, above, below]
      # Also a list and only one will be true at any time
      game.food.x < snake_head.x,
      game.food.x > snake_head.x,
      game.food.y < snake_head.y, # Less than because [0, 0] is top left
      game.food.y > snake_head.y 
    ]
    # Conver state list into numpy array
    return np.array(state, dtype=int)

  def remember(self, state, action, reward, next_state, game_over):
    # Safe info to memory
    # Pops left memory if it exceeds memory size
    self.memory.append((state, action, reward, next_state, game_over))

  def train_short_memory(self, state, action, reward, next_state, game_over):
    # Trains AI with one game step (move)
    self.trainer.train_step(state, action, reward, next_state, game_over)

  def train_long_memory(self):
    # Trains with a batch of game 'memory'
    if len(self.memory) > BATCH_SIZE:
      # Get a random subsample of tuples from memory
      mini_sample = random.sample(self.memory, BATCH_SIZE)
    else:
      # If we don't have a BATCH_SIZE of memory, use it all
      mini_sample = self.memory

    # Zips all states, actions, etc into one tuple each using the * (splat) operator
    states, actions, rewards, next_states, game_overs = zip(*mini_sample)
    self.trainer.train_step(states, actions, rewards, next_states, game_overs)

  def get_action(self, state):
    # Make a random move
    # Tradeoff between exploration vs exploitation
    # 80 is hard coded, can be changed
    self.epsilon = 80 - self.num_games/4
    final_move = [0, 0, 0]
    # Make a random move if RNG hits
    if random.randint(0, 200) < self.epsilon:
      # Chooses a random movement from [straight, left, right]
      move = random.randint(0, 2)
      final_move[move] = 1
    else:
      # Exploitation used instead
      # Make a prediction based off the model using a tensor
      # TODO Learn more about tensors and models
      state0 = torch.tensor(state, dtype=torch.float)
      # Prediction will be a list of random numbers
      # i.e. [5.0, 2.7, 0.1]
      prediction = self.model(state0)
      # Get movement from that prediction by setting max as the movement
      # i.e. [1, 0, 0] from [5.0, 2.7, 0.1]
      # torch.argmax returns index with the highest value
      move = torch.argmax(prediction).item()
      final_move[move] = 1
    return final_move

def train():
  plot_scores = deque()
  plot_mean_scores = deque()
  total_score = 0
  record = 0
  agent = Agent()
  game = SnakeGameAI()

  while True:
    # Get old state
    old_state = agent.get_state(game)
    # Get move
    final_move = agent.get_action(old_state)
    # Perform move and get new state
    game_over, score, reward = game.play_step(final_move)
    new_state = agent.get_state(game)
    # Train short memory
    agent.train_short_memory(old_state, final_move, reward, new_state, game_over)
    # Remember
    agent.remember(old_state, final_move, reward, new_state, game_over)

    if game_over:
      # Train long memory, also called replay memory, experience replay
      # Trains with all the data from previous games
      # Plot results
      game.reset_game_state()
      agent.num_games += 1
      agent.train_long_memory()

      if score > record:
        record = score
        agent.model.save()

      print('Game:', agent.num_games, 'Score:', score, 'Record:', record)
      
      plot_scores.append(score)
      total_score += score
      mean_score = total_score / agent.num_games
      plot_mean_scores.append(mean_score)
      plot(list(plot_scores), list(plot_mean_scores))
      # TODO fix OPM error

def main():
  train()

if __name__ == '__main__':
  main()