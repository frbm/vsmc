from math import pi
from vsmc import *
from torch.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal


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
        mu_0, s_0, a, q, c, rr = self.model_params
        dx = mu_0.size(0)
        dy = rr.size(0)

        log_likelihood = 0.
        x_fil = torch.zeros(dx)
        p_fil = torch.zeros((dx, dx))
        x_pred = mu_0
        p_pred = s_0

        for s in range(t):
            if s > 0:
                x_pred = torch.matmul(a, x_fil)
                p_pred = torch.matmul(a, torch.matmul(p_fil, a.T)) + q

            # update
            y_s = y[s, :] - torch.matmul(c, x_pred)
            b = torch.matmul(c, torch.matmul(p_pred, c.T)) + rr
            k = torch.linalg.solve(b, torch.matmul(c, p_pred)).T
            x_fil = x_pred + torch.matmul(k, y_s)
            p_fil = p_pred - torch.matmul(k, torch.matmul(c, p_pred))

            sgn, log_det = torch.linalg.slogdet(b)
            log_likelihood -= 0.5 * \
                (torch.sum(y_s * torch.linalg.solve(b, y_s)) +
                 log_det + dy * log(2 * pi))

        return log_likelihood

    @staticmethod
    def log_normal(x, mu, sigma):
        dim = sigma.size(0)
        sgn, log_det = torch.linalg.slogdet(sigma)
        log_norm = - 0.5 * (dim * log(2 * pi) + log_det)
        p_rec = torch.linalg.inv(sigma)

        test2 = x - mu
        test1 = torch.matmul(p_rec, (x - mu).T)

        log_norm = log_norm - 0.5 * \
            torch.sum(
                torch.mul((x - mu), torch.matmul(p_rec, (x - mu).T).T), dim=1)
        return log_norm

    def helper(self, t, x_p):
        print(t)
        mu_0, s_0, a, q, c, rr = self.model_params
        mu_t, lint, log_s2t = self.prop_params[(3*t):(3*t+3)]
        s2t = torch.exp(log_s2t)

        if t > 0:
            mu = mu_t + torch.matmul(a, x_p.T).T * lint
        else:
            mu = mu_t + lint * mu_0
        return mu, s2t

    def log_prop(self, t, x_c, x_p):
        mu, s2t = self.helper(t, x_p)
        return self.log_normal(x_c, mu, torch.diag(s2t))

    def log_target(self, t, x_c, x_p, y):
        mu_0, s_0, a, q, c, rr = self.model_params

        if t > 0:
            log_f = self.log_normal(x_c, torch.matmul(a, x_p.T).T, q)
        else:
            log_f = self.log_normal(x_c, mu_0, s_0)
        log_g = self.log_normal(torch.matmul(c, x_c.T).T, y[t], rr)
        return log_f + log_g

    def log_weights(self, t, x_c, x_p, y):
        target = self.log_target(t, x_c, x_p, y)
        prop = self.log_prop(t, x_c, x_p)
        return target - prop

    def sim_prop(self, t, x_p):
        mu, s2t = self.helper(t, x_p)
        return mu + torch.randn(*x_p.size()) * torch.sqrt(s2t)

    def objective(self, y, adaptive_resampling=False):
        return - self.forward(y, adaptive_resampling=adaptive_resampling)


if __name__ == '__main__':
    # Model hyper-parametersf
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
    printing_freq = 10

    n = 4

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

    print('True y:')
    print(y_true)
    print('')
    print('Simulated y:')
    print(smc_model.sim_q(y_true))
