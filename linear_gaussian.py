from math import pi
from vsmc import *
from torch.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)


class LinearGaussianStateSpaceSMC(VariationalSMC):
    def __init__(self, dx, dy, alpha, r, obs, n, t, scale):
        super().__init__(dx, dy, alpha, r, obs, n, t, scale)

    def generate_data(self, t=5):
        """
        Generates simulated data for a linear Gaussian hidden state model

        Returns the tensors x and y which contain the simulated hidden states and the simulated observations respectively
        """
        mu_0, s_0, a, q, c, rr = self.model_params
        # retrieves the model parameters stored in the model_params attribute of the object. These parameters are:
        # mu_0: the mean of the initial state.
        # s_0: the covariance of the initial state.
        # a: the transition matrix of the hidden state.
        # q: covariance matrix of the hidden state
        # c: measurement matrix of the model
        # rr: multiplication of the covariance matrix of the observation r by an identity matrix of size dy

        dx = mu_0.size(0)
        # recovers the dimension of the hidden state using the size of the tensor mu_0 (which corresponds to the average of the initial state)
        dy = rr.size(0)
        # the size of the observation is determined by the size of the tensor rr (which corresponds to the covariance of the observation)

        x = torch.zeros((t, dx))
        # will store the simulated hidden states
        y = torch.zeros((t, dy))
        # will store the simulated observations

        for s in range(t):
            if s > 0:
                # checks if the current time step is greater than 0
                # if it is, it means that the hidden state is already defined and that it can be used to simulate the following state
                mean = torch.matmul(a, x[s - 1, :])
                # average of the next state using the transition matrix a and the previous state x[s - 1, :]
                dist = MultivariateNormal(loc=mean, covariance_matrix=q)
                # creates a Gaussian multivariate distribution from the mean and covariance calculated previously
                x[s, :] = dist.rsample()
                # draws a sample of the next state from the multivariate Gaussian distribution created earlier
            else:
                dist = MultivariateNormal(loc=mu_0, covariance_matrix=s_0)
                # creates a Gaussian multivariate distribution from the mean and covariance of the initial state (mu_0 and s_0)
                x[s, :] = dist.rsample()
                # draws a sample of the initial state from the multivariate Gaussian distribution created previously

            mean = torch.matmul(c, x[s, :])
            # computes the average of the next observation using the projection matrix c and the current state x[s, :]
            dist = MultivariateNormal(loc=mean, covariance_matrix=rr)
            # creates a Gaussian multivariate distribution from the mean and covariance of the observation (mean and rr)
            y[s, :] = dist.rsample()
            # draws a sample of the following observation from the multivariate Gaussian distribution created earlier

        # returns the tensors x and y which contain the simulated hidden states and the simulated observations respectively
        return x, y

    def log_marginal_likelihood(self, t, y):
        """
        Computes the marginal likelihood of the given sequence of observations y using
        the Kalman filter algorithm

        Returns the marginal likelihood
        """
        mu_0, s_0, a, q, c, rr = self.model_params
        dx = mu_0.size(0)
        dy = rr.size(0)

        log_likelihood = 0.
        # initializes the marginal likelihood to 0
        x_fil = torch.zeros(dx)
        # initializes the filtered state to a zero tensor of size dx
        p_fil = torch.zeros((dx, dx))
        # initializes the covariance of the filtered state to a zero tensor of size dx x dx.
        x_pred = mu_0
        # initializes the predicted state to the average of the initial state
        p_pred = s_0
        # initializes the covariance of the predicted state to the covariance of the initial state (s_0)

        for s in range(t):
            if s > 0:
                #  the method computes the predicted state and the covariance of the predicted state for the next step using
                # the linear transition (a) and the covariance of the state (q)
                x_pred = torch.matmul(a, x_fil)
                p_pred = torch.matmul(a, torch.matmul(p_fil, a.T)) + q

            # update
            y_s = y[s, :] - torch.matmul(c, x_pred)
            b = torch.matmul(c, torch.matmul(p_pred, c.T)) + rr
            k = torch.linalg.solve(b, torch.matmul(c, p_pred)).T
            x_fil = x_pred + torch.matmul(k, y_s)
            p_fil = p_pred - torch.matmul(k, torch.matmul(c, p_pred))
            # updates the filtered state and the covariance of the filtered state using the predicted state and the covariance
            # of the predicted state as well as the current observation y[s, :] and the projection matrix (c) and the covariance
            # of the observation (rr)

            sgn, log_det = torch.linalg.slogdet(b)
            log_likelihood -= 0.5 * \
                (torch.sum(y_s * torch.linalg.solve(b, y_s)) +
                 log_det + dy * log(2 * pi))
            # calculates the likelihood of the current observation using the filtered state and the covariance of the filtered state,
            # and updates the marginal likelihood using the likelihood of the current observation

        return log_likelihood

    @staticmethod
    def log_normal(x, mu, sigma):
        """
        Computes the logarithm of the probability density of a tensor of data x given a multivariate normal distribution of mean mu
        and covariance sigma
        
        Returns the value of the logarithm of the probability density of the normal distribution of mean mu and covariance matrix sigma
        evaluated in x
        """
        dim = sigma.size(0)
        # covariance dimension
        sgn, log_det = torch.linalg.slogdet(sigma)
        # logarithm of the determinant of the covariance and the sign of the covariance
        log_norm = - 0.5 * (dim * log(2 * pi) + log_det)
        # initializes the logarithm of the probability density to the constant that does not depend on x
        p_rec = torch.linalg.inv(sigma)
        # calculates the inverse of the covariance
        log_norm = log_norm - 0.5 * torch.sum(torch.mul((x - mu), torch.matmul(p_rec, (x - mu).T).T), dim=1)
        # updates the logarithm of the probability density using the inverse of the covariance and the difference between x and mu

        test2 = x - mu
        test1 = torch.matmul(p_rec, (x - mu).T)
        # not used

        log_norm = log_norm - 0.5 * \
            torch.sum(
                torch.mul((x - mu), torch.matmul(p_rec, (x - mu).T).T), dim=1)
        # computes the value of the logarithm of the probability density of the normal distribution of mean mu and covariance matrix sigma
        # evaluated in x

        return log_norm

    def helper(self, t, x_p):
        """
        Computes the mean and variance of the distribution of the latent variable for a given step using the model parameters and
        the propagation parameters
        
        t: current stage of the simulation
        x_p: corresponds to the value of the latent variable in the previous step

        Returns the mean and variance in the form of torch tensors
        """
        mu_0, s_0, a, q, c, rr = self.model_params
        mu_t, lint, log_s2t = self.prop_params[(3*t):(3*t+3)]
        s2t = torch.exp(log_s2t)

        if t > 0:
            mu = mu_t + torch.matmul(a, x_p.T).T * lint
        else:
            mu = mu_t + lint * mu_0
        return mu, s2t

    def log_prop(self, t, x_c, x_p):
        """
        computes the log-likelihood of the multivariate normal distribution associated with the parameters mu and
        s2t using the value x_c as sample.
        """
        mu, s2t = self.helper(t, x_p)
        return self.log_normal(x_c, mu, torch.diag(s2t))

    def log_target(self, t, x_c, x_p, y):
        """
        Computes the log-likelihood of the model as a function of the observed data y, and the hidden state at time t, x_c, and
        the hidden state at the previous time, x_p

        The log-likelihood is the sum of the log-likelihoods of the normal distributions associated with the dynamics of the model (log_f)
        and the normal distribution associated with the generation of the observations (log_g).
        The dynamics of the model is described by the transition matrix a and the variance-covariance matrix q and the observation by
        the projection matrix c and the variance-covariance matrix rr.
        If t is greater than zero, then the hidden state at time t follows a normal distribution with mean equal to
        the transition matrix multiplied by the hidden state at the previous time, and variance-covariance equal to q.
        If t is equal to zero, then the hidden state at time t follows a normal distribution with mean equal to the initial mean m_0 and
        variance-covariance equal to s_0. 
        The observations follow a normal distribution with mean equal to the projection matrix multiplied by the hidden state at time t, 
        and variance-covariance equal to rr.
        """
        mu_0, s_0, a, q, c, rr = self.model_params

        if t > 0:
            log_f = self.log_normal(x_c, torch.matmul(a, x_p.T).T, q)
        else:
            log_f = self.log_normal(x_c, mu_0, s_0)
        log_g = self.log_normal(torch.matmul(c, x_c.T).T, y[t], rr)
        return log_f + log_g

    def log_weights(self, t, x_c, x_p, y):
        """
        Calculates the weights of the particles using the bootstrap weight formula
        """
        target = self.log_target(t, x_c, x_p, y)
        prop = self.log_prop(t, x_c, x_p)
        return target - prop

    def sim_prop(self, t, x_p):
        """
        Simulates the propagation of the state at time t+1 according to the state at time t
        """
        mu, s2t = self.helper(t, x_p)
        return mu + torch.randn(*x_p.size()) * torch.sqrt(s2t)

    def objective(self, y, adaptive_resampling=False):
        """
        Computes the objective of the hidden Markov chain

        Uses the forward function to calculate the lower bound of the hidden Markov chain and returns its opposite.
        This function can be used as a loss function for objective minimization.
        """
        return - self.forward(y, adaptive_resampling=adaptive_resampling)


if __name__ == '__main__':
    # Model hyper-parameters
    t = 10
    dx = 5
    dy = 3
    alpha = 0.42
    r = .1
    obs = 'sparse'

    # Training parameters
    scale = 0.5
    epochs = 1000
    lr = 0.001
    printing_freq = 100

    n = 6

    smc_model = LinearGaussianStateSpaceSMC(dx, dy, alpha, r, obs, n, t, scale)

    optimizer = Adam(smc_model.prop_params, lr=lr)

    print("Generating data...")
    x_true, y_true = smc_model.generate_data(t)

    lml_true = smc_model.log_marginal_likelihood(t, y_true)
    print(f'True log-marginal likelihood: {lml_true.item()}.')
    print('')

    # training loop
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        loss = smc_model.objective(y_true)
        loss.backward()
        optimizer.step()

        if epoch % printing_freq == 0:
            print(f'Epoch {epoch} of {epochs}.')
            print(f'Current ELBO: {-loss.item()}.')
            print('')

    print('True x:')
    print(x_true)
    print('')
    print('Simulated x:')
    print(smc_model.sim_q(y_true))
