import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Other implementation of the IWAE for test

class IWAE_SimplerImplementation(nn.Module):
    # Calculates loss explicitly.
    def __init__(self, num_hidden1, num_hidden2, latent_space):
        super(IWAE_SimplerImplementation, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=784, out_features=num_hidden1),
            nn.ReLU(),
            nn.Linear(num_hidden1, num_hidden2),
            nn.ReLU(),
        )

        self.fc21 = nn.Linear(in_features=num_hidden2, out_features=latent_space)
        self.fc22 = nn.Linear(in_features=num_hidden2, out_features=latent_space)

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=latent_space, out_features=num_hidden2),
            nn.ReLU(),
            nn.Linear(num_hidden2, num_hidden1),
            nn.ReLU(),
        )
        self.decode = nn.Linear(in_features=num_hidden1, out_features=784)
        self.latent = latent_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def encode(self, x):
        x = self.fc1(x)
        mu = self.fc21(x)
        log_var = self.fc22(x)

        # Reparameterize
        eps = torch.randn_like(log_var)
        h = mu + torch.exp(log_var * 0.5) * eps
        return mu, log_var, h, eps

    def forward(self, x):
        """
        Purely to see reconstruction, not for calculating loss.
        """

        # Encode
        mu, log_var, h, eps = self.encode(x)

        # decode
        recon_X = self.fc3(h)
        recon_X = torch.sigmoid(self.decode(recon_X))
        return recon_X

    def calc_loss(self, x, beta):

        # Encode
        mu, log_var, h, eps = self.encode(x)

        # Calculating P(x,h)
        log_Ph = torch.sum(-0.5 * h ** 2 - 0.5 * torch.log(2 * h.new_tensor(np.pi)),
                           -1)  # equivalent to lognormal if mu=0,std=1 (i think)
        recon_X = torch.sigmoid(self.decode(self.fc3(h)))  # Creating reconstructions
        log_PxGh = torch.sum(x * torch.log(recon_X) + (1 - x) * torch.log(1 - recon_X),
                             -1)  # Bernoulli decoder: Appendix c.1 Kingma p(x|h)
        log_Pxh = log_Ph + log_PxGh  # log(p(x,h))
        log_QhGx = torch.sum(-0.5 * (eps) ** 2 - 0.5 * torch.log(2 * h.new_tensor(np.pi)) - 0.5 * log_var,
                             -1)  # Evaluation in lognormal

        # Weighting according to equation 13 from IWAE paper
        log_weight = (log_Pxh - log_QhGx).detach().data
        log_weight = log_weight - torch.max(log_weight, 0)[0]
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)

        # scaling
        loss = torch.mean(-torch.sum(weight * (log_PxGh + (log_Ph - log_QhGx)*beta), 0))

        return loss

    def sample(self, n_samples):
        eps = torch.randn((n_samples, self.latent)).to(self.device)
        sample = self.fc3(eps)
        sample = torch.sigmoid(self.decode(sample))
        return sample
    
###My personal implementation

class Block(nn.Module):
    #I define a generic Block to get a better understanding of the architecture
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Tanh())
        
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.transform(x)
        mu = self.fc_mu(out)
        log_sigma = self.fc_logsigma(out)
        return mu, log_sigma
    
class IWAE(nn.Module):
    def __init__(self, dim_h1, dim_h2, dim_x):
        super(IWAE, self).__init__()

        self.dim_h1 = dim_h1
        self.dim_h2 = dim_h2
        self.dim_x = dim_x

        ### Encoder
        
        self.encoder_h1 = Block(dim_x, 200, dim_h1)
        
        self.encoder_h2 = Block(dim_h1, 100, dim_h2)
        
        ### Decoder
        
        self.decoder_h1 = Block(dim_h2, 100, dim_h1) 
        
        self.decoder_x =  nn.Sequential(nn.Linear(dim_h1, 200),
                                        nn.Tanh(),
                                        nn.Linear(200, 200),
                                        nn.Tanh(),
                                        nn.Linear(200, dim_x)
                                        )  
        
    def reparameterize(self, mu, log_sigma):
        #Reparametrization trick

        sigma = torch.exp(0.5*log_sigma) # standard deviation
        eps = torch.randn_like(sigma) # `randn_like` as we need the same size
        sample = mu + (eps * sigma) # sampling as if coming from the input space
        
        return sample
    
    def encoder(self, x):
        
        #q(h1|x) 
        mu_h1, log_sigma_h1 = self.encoder_h1(x)
        h1 = self.reparameterize(mu_h1, log_sigma_h1)
        
        #q(h2|h1) 
        mu_h2, log_sigma_h2 = self.encoder_h2(h1)
        h2 = self.reparameterize(mu_h2, log_sigma_h2)
        
        return (h1, mu_h1, log_sigma_h1), (h2, mu_h2, log_sigma_h2)
    
    def decoder(self, h1, h2):
        
        #p(h1|h2) 
        mu_h1, log_sigma_h1 = self.decoder_h1(h2)
        h1 = self.reparameterize(mu_h1, log_sigma_h1)
        
        # Only one parameter because it is a Bernoulli distribution
        #p(x|h1) 
        p =   torch.sigmoid(self.decoder_x(h1))
        
        return (h1, mu_h1, log_sigma_h1), (p)
    
    def forward(self, x):
        (h1, mu_h1, log_sigma_h1), (h2, mu_h2, log_sigma_h2) = self.encoder(x) 
        _, p = self.decoder(h1, h2)        
        
        return ((h1, mu_h1, log_sigma_h1), (h2, mu_h2, log_sigma_h2)), (p)    
    
    def calc_loss(self, inputs):
        (h1, mu_h1, log_sigma_h1), (h2, mu_h2, log_sigma_h2) = self.encoder(inputs)
                
        ### Calculating q(h1,h2|x)
        #log q(h1|x) 
        log_Qh1Gx = torch.sum(- 0.5 * torch.log(2 * h1.new_tensor(np.pi))-0.5*((h1-mu_h1))**2/torch.exp(log_sigma_h1) - log_sigma_h1, -1)

        #log q(h2|h1) 
        log_Qh2Gh1 = torch.sum(- 0.5 * torch.log(2 * h2.new_tensor(np.pi))-0.5*((h2-mu_h2))**2/torch.exp(log_sigma_h2) - log_sigma_h2, -1)
        
        #log q(h2|x) = log q(h2|h1) + log q(h1|x) 
        log_Qh1h2Gx = log_Qh1Gx + log_Qh2Gh1
        
        (h1, mu_h1, sigma_h1), (p) = self.decoder(h1, h2)
        
        ### Calculating p(x,h1,h2)
        #log p(h2)
        log_Ph2 = torch.sum(-0.5*h2**2, -1)
        #log p(h1|h2) 
        log_Ph1Gh2 = torch.sum(- 0.5 * torch.log(2 * h1.new_tensor(np.pi)) - 0.5*((h1-mu_h1))**2/torch.exp(log_sigma_h1) - log_sigma_h1, -1)
        #log p(x|h1) 
        log_PxGh1 = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1)
        
        # log p(x,h1,h2) = log p(x|h1) + log p(h1|h2)  + log p(h2)
        log_Pxh1h2 = log_Ph2 + log_Ph1Gh2 + log_PxGh1

        # Weighting according to equation 13 from IWAE paper
        log_weight = (log_Pxh1h2 - log_Qh1h2Gx).detach().data
        log_weight = log_weight - torch.max(log_weight, 0)[0]
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        
        loss = -torch.mean(torch.sum(weight * (log_Pxh1h2 - log_Qh1h2Gx), 0))
        
        return loss
    
    def sim_q(self,y):
        
        (h1, mu_h1, log_sigma_h1), (h2, mu_h2, log_sigma_h2) = self.encoder(y) 
        return h1.T @ h2
    
    def sim_p(self,x):
        mu_h1, log_sigma_h1 = self.decoder_h1(x)
        h1 = self.reparameterize(mu_h1, log_sigma_h1)
        #p(x|h1) 
        x_recon  = torch.sigmoid(self.decoder_x(h1))
        
        return h1 @  x_recon

    def sample(self, n_samples):
            eps = torch.randn((n_samples, self.dim_h2))
            sample = self.sim_p(eps)
            return sample


import torchvision
import torch.optim as optim
import seaborn as sns

import matplotlib.pyplot as plt
import torch.nn as nn



def Plot_loss_curve(train_list, test_dict):
    x_tst = list(test_dict.keys())
    y_tst = list(test_dict.values())
    train_x_vals = np.arange(len(train_list))
    plt.figure(2)
    plt.xlabel('Num Steps')
    plt.ylabel('ELBO')
    plt.title('ELBO Loss Curve')
    plt.plot(train_x_vals, train_list, label='train')
    plt.plot(x_tst, y_tst, label='tst')
    plt.legend(loc='best')
    plt.locator_params(axis='x', nbins=10)

    plt.show()
    return

def create_canvas(x):
    rows = 10
    columns = 10

    plt.figure(1)
    canvas = np.zeros((28 * rows, columns * 28))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x[idx].detach().reshape((28, 28))

    return canvas

sns.set_style("darkgrid")

# Hyperparameters
gif_pics = True
batch_size = 250
lr = 1e-4
#num_epochs = 65
num_epochs = 5

train_log = []
test_log = {}
k = 0
num_samples = 5
beta = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Test_Implementation = False #False if you want to try my implementation



if Test_Implementation:
   net = IWAE_SimplerImplementation(1024, 512, 32).to(device)
else :
    net = IWAE(100, 50, 784).to(device)

optimizer = optim.Adam(net.parameters(), lr=lr)

# Data loading
t = torchvision.transforms.transforms.ToTensor()
train_data = torchvision.datasets.MNIST('./', train=True, transform=t, target_transform=None, download=True)
test_data = torchvision.datasets.MNIST('./', train=False, transform=t, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    for idx, train_iter in enumerate(train_loader):
        batch, label = train_iter[0], train_iter[1]
        batch = batch.view(batch.size(0), -1)  # flatten
        batch = batch.expand(num_samples, batch.shape[0], -1).to(device)  # make num_samples copies

        if Test_Implementation:
           batch_loss = net.calc_loss(batch,beta)
        else :
            batch_loss = net.calc_loss(batch)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        train_log.append(batch_loss.item())

        k += 1

    loss_batch_mean = []
    for idx, test_iter in enumerate(test_loader):
        batch, label = train_iter[0], train_iter[1]
        batch = batch.view(batch.size(0), -1)  # flatten
        batch = batch.expand(num_samples, batch.shape[0], batch.shape[1]).to(device)  # make num_samples copies

        if Test_Implementation:
           test_loss = net.calc_loss(batch,beta)
        else :
           test_loss = net.calc_loss(batch)

        loss_batch_mean.append(test_loss.detach().item())

    test_log[k] = np.mean(loss_batch_mean)
    if gif_pics and epoch % 2 == 0:
        batch = batch[0, :100, :].squeeze()
        
        if Test_Implementation:
            recon_x = net(batch)
        else:
            recon_x = net(batch)[1]

        samples = net.sample(100).detach().cpu()
        fig, axs = plt.subplots(1, 2, figsize=(5, 10))

        # Reconstructions
        recon_x = create_canvas(recon_x)
        axs[0].set_title('Epoch {} Reconstructions'.format(epoch + 1))
        axs[0].axis('off')
        axs[0].imshow(recon_x, cmap='gray')

        # Samples
        samples = create_canvas(samples)
        axs[1].set_title('Epoch {} Sampled Samples'.format(epoch + 1))
        axs[1].axis('off')
        axs[1].imshow(samples, cmap='gray')
        save_path = './Figure/GIF/gif_pic' + str(epoch + 1) + '.jpg'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print('[Epoch: {}/{}][Step: {}]\tTrain Loss: {},\tTest Loss: {}'.format(
        epoch + 1, num_epochs, k, round(train_log[k - 1], 2), round(test_log[k], 2)))

###### Loss Curve Plotting ######
Plot_loss_curve(train_log, test_log)
plt.savefig('./Figure/Figure_1.png', bbox_inches='tight')
plt.close()

###### Sampling #########
x = next(iter(train_loader))[0].to(device)
x = x.view(x.size(0), -1)[:100]  # flatten and limit to 100

if Test_Implementation:
    recon_x = net(batch)
else:
    recon_x = net(batch)[1]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
x_true = create_canvas(x.detach().cpu())
axs[0].set_title('Ground Truth MNIST Digits')
axs[0].axis('off')
axs[0].imshow(x_true, cmap='gray')

recon_x = create_canvas(recon_x.detach().cpu())
axs[1].set_title('Reconstructed MNIST Digits')
axs[1].axis('off')
axs[1].imshow(recon_x, cmap='gray')

samples = net.sample(100).detach().cpu()
samples = create_canvas(samples)
axs[2].set_title('Sampled MNIST Digits')
axs[2].axis('off')
axs[2].imshow(samples, cmap='gray')
plt.savefig('./Figure/Figure_2.png', bbox_inches='tight')
plt.close()
