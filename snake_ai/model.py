import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
  # Sizes are the number of nodes in the respective layer
  def __init__(self, input_size, hidden_size, output_size):
    # TODO: Wtf is a super initializer
    super().__init__()
    # Create two linear layers
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, output_size)

  # Always needs to make a forward function in PyTorch
  # x is the tensor
  def forward(self, x):
    # F.relu is an activation function
    x = F.relu(self.linear1(x))
    x = self.linear2(x)
    return x

  def save(self, file_name='model.pth'):
    model_folder_path = './model'
    file_name = os.path.join(model_folder_path, file_name)

    if not os.path.exists(model_folder_path):
      os.makedirs(model_folder_path)
    # Save the brains
    torch.save(self.state_dict(), file_name)

class QTrainer:
  def __init__(self, model, learning_rate, gamma):
    self.model = model
    self.learning_rate = learning_rate
    self.gamma = gamma
    # Choose an optimizer
    # Adam was current choice, but there are others
    # TODO Look at other optimizers
    self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
    # Criterion is the loss function, Mean Squared Error (MSE) is used
    self.criterion = nn.MSELoss()

  def train_step(self, state, action, reward, next_state, game_over):
    # Default will be either x for single value or (n, x) for multiple values
    state = torch.tensor(state, dtype=torch.float)
    next_state = torch.tensor(next_state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float)

    if len(state.shape) == 1:
      # Reshape  from x to (1, x) with 1 being dimension
      state = torch.unsqueeze(state, 0)
      next_state = torch.unsqueeze(next_state, 0)
      action = torch.unsqueeze(action, 0)
      reward = torch.unsqueeze(reward, 0)
      game_over = (game_over, )

    # Predicted Q values with current state
    # Prediction has 3 values
    # Q_new = R + y * max(next predicted Q value)
    # We only do this if the game is not over
    # prediction.clone()
    # predictions[argmax(action)] = Q_new
    prediction = self.model(state)
    target = prediction.clone()

    # All sizes should be the same, game_over is chosen here
    for i in range(len(game_over)):
      Q_new = reward[i]
      if not game_over[i]:
        Q_new = reward[i] + self.gamma*torch.max(self.model(next_state[i]))
      target[i][torch.argmax(action).item()] = Q_new
    
    # Reset the gradients to 0, something you have to do in PyTorch
    self.optimizer.zero_grad()
    loss = self.criterion(target, prediction)
    # Apply backpropogation
    loss.backward()

    self.optimizer.step()