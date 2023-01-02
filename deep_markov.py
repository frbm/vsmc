from numpy import load
import torch
import torch.nn as nn
import torch.optim as optim


# Define the RNN model
class RNNModel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(RNNModel, self).__init__()
    
    # Define the necessary layers and parameters
    self.mu = nn.Linear(input_size, hidden_size)
    self.sigma = nn.Linear(input_size, hidden_size)
    self.eta = nn.Linear(input_size, output_size)
    
  def forward(self, x):
    # Compute the hidden state using the mu and sigma layers
    mu = self.mu(x)
    sigma = self.sigma(x)
    x = mu + torch.exp(sigma / 2) * torch.randn_like(mu)
    
    # Compute the output using the eta layer
    y = torch.poisson(input=torch.exp(self.eta(x)))
    return y


# Define the proposal distribution
class ProposalDistribution(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(ProposalDistribution, self).__init__()
    
    # Define the necessary layers and parameters
    self.mu_x = nn.Linear(input_size, hidden_size)
    self.sigma_x = nn.Linear(input_size, hidden_size)
    self.mu_y = nn.Linear(input_size, hidden_size)
    self.sigma_y = nn.Linear(input_size, hidden_size)
    
  def forward(self, x, y):
      x_dist = torch.normal(mean=self.mu_x(x), std=torch.exp(self.sigma_x(x)))
      y_dist = torch.normal(mean=self.mu_y(y), std=torch.exp(self.sigma_y(y)))
      return torch.clip(x_dist * y_dist, 1e-5, 10)

def compute_loss_SMC(rnn_model, proposal_dist, particles, weights, y_t):
  # Compute the log likelihood of the observations under the model
  log_likelihood = 0
  for particle, weight in zip(particles, weights):
    log_likelihood += weight * rnn_model(particle)
  
  # Compute the log likelihood of the particles under the proposal distribution
  log_proposal = 0
  for particle, weight in zip(particles, weights):
    log_proposal += weight * proposal_dist(y_t, particle)
  
  # Compute the loss as the negative ELBO
  loss = -(log_likelihood - log_proposal)
  return loss

def compute_loss(data, rnn_model, proposal_dist, num_particles=10):
  # Initialize the total loss
  total_loss = 0
  
  # Loop through the data and compute the loss at each time step
  for t, y_t in enumerate(data):
    # Initialize the particles and weights
    particles = []
    weights = []
    
    # Sample particles from the proposal distribution
    for i in range(num_particles):
      if i > 0:
        x_t = proposal_dist(y_t, particles[-1])
      else:
        x_t = proposal_dist(y_t, torch.zeros_like(y_t))
      particles+= [x_t]
      weights+= [1.0 / num_particles]
    
    # Compute the loss at this time step
    loss = compute_loss_SMC(rnn_model, proposal_dist, particles, weights, y_t)
    
    # Add the loss to the total loss
    total_loss += loss
    
  return torch.mean(total_loss)

def update_weights(rnn_model, optimizer, loss):
  # Clear the gradients
  optimizer.zero_grad()
  
  # Compute the gradients
  loss.backward()
  
  # Update the weights
  optimizer.step()


if __name__ == '__main__':
  n_epochs = 100

  path = "./data.npy"

  try:
    data = torch.Tensor(load(path))
    # data = torch.normal(0, 1, (10, 10)) # replace with real data
  except:
    FileNotFoundError("File data.npy not found in vsmc")
  
  input_size = data.shape[0]
  hidden_size = 10
  output_size = data.shape[1]

  # Define the RNN model
  rnn_model = RNNModel(input_size, hidden_size, output_size)

  # Define the optimizer
  optimizer = optim.Adam(rnn_model.parameters(), lr=0.01)

  # Define the proposal distribution
  proposal_dist = ProposalDistribution(input_size, hidden_size)

  # Define the optimizer
  optimizer = optim.Adam(rnn_model.parameters(), lr=0.01)

  # Loop through the data and update the weights
  for epoch in range(n_epochs):
    # Compute the loss
    loss = compute_loss(data, rnn_model, proposal_dist, num_particles=8)

    # Update the weights
    update_weights(rnn_model, optimizer, loss)

    # Print the loss
    print("Epoch: {}, Loss: {}".format(epoch, loss))
