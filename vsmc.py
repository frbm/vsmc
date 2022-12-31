import torch
from math import log

torch.manual_seed(0)


class VariationalSMC:
    def __init__(self, dx, dy, alpha, r, obs, n, t, scale):
        """
        dx: the number of dimensions of the hidden state of the model
        dy: the number of dimensions of the observation
        alpha: a decay coefficient used to initialize the covariance matrix of the hidden state
        r: the covariance matrix of the observation
        obs: a type of observation (sparse or not)
        n: the number of particle points to use in the PKF
        t: number of time steps
        scale: parameter scaling
        """
        self.dx = dx
        self.dy = dy
        self.alpha = alpha
        self.r = r
        self.obs = obs
        self.n = n

        model_params = self.init_model_params(dx, dy, alpha, r, obs)
        prop_params = self.init_prop_params(t, dx, scale)
        self.all_params = []
        for param in model_params:
            self.all_params.append(param.requires_grad_(True))
        for param in prop_params:
            self.all_params.append(param.requires_grad_(True))
        self.model_params = self.all_params[:6]
        self.prop_params = self.all_params[6:]

    @staticmethod
    def init_model_params(dx, dy, alpha, r, obs):
        raise NotImplementedError('VariationalSMC class requires a init_model_params method.')

    @staticmethod
    def init_prop_params(t, dx, scale=0.5):
        raise NotImplementedError('VariationalSMC class requires init_prop_params method.')

    @torch.no_grad()
    def resampling(self, w):
        """
        Stratified resampling with "nograd_primitive" to ensure autograd
        takes no derivatives through it.
        """
        n = self.n
        # size of the weight vector, which corresponds to the number of particle points
        bins = torch.cumsum(w, dim=-1)
        # calculates the cumulative sum of the weight vector, i.e. the sum of each element of the vector with all the
        # elements that precede it.
        # this will allow us to determine the "bins" (weight intervals) that will be used for sampling
        ind = torch.arange(n)
        u = (ind + torch.rand(n)) / n
        # generates a vector of random values (noisy x -> x)
        # returns the index of the bin into which each u value falls
        # thus, using the cumulative sum of weights as "bins", we can select particle points based on their weights
        return torch.bucketize(u, bins)

    def vsmc_lower_bound(self, y, adaptive_resampling=False):
        """
        Estimation of the VSMC lower bound.
        """
        # constants
        t = y.size(0)  # size of the observation data
        dx = self.dx  # number of dimensions of the hidden state of the model
        n = self.n  # the number of particle points to use in the PKF

        # initialisation
        x = torch.zeros((n, dx))  # store the particle points at each step
        log_w = torch.zeros(n)  # store the logarithms of the particle point weights at each step
        w = torch.exp(log_w)  # calculates the weights by exponentiating the logarithms of the weights
        w /= torch.sum(w)
        log_z = 0.  # store the value of the lower bound of variational SMC at each step
        max_log_w = torch.max(log_w)
        ess = 1. / torch.sum(torch.square(w)) / n  # calculates the effective support rate (ESS) of the particle points

        for s in range(t):
            # resampling
            if adaptive_resampling:
                # checks if the adaptive sampling is activated
                # if it is, the method performs a different treatment depending on whether the ESS is lower or higher
                # than 0.5
                if ess < 0.5:
                    # sampling using the resampling method
                    # the selected particle points are stored in the ancestors variable and are used to update the
                    # current particle points x
                    # the log_z variable is also updated using the maximum value of the logarithms of the weights and
                    # the sum of the weights
                    ancestors = self.resampling(w)
                    xp = x[ancestors]
                    log_z = log_z + max_log_w + torch.log(torch.sum(w)) - log(n)
                    log_w = torch.zeros(n)
                else:
                    # the method does not sample and simply uses the current particle points to update the xp variable
                    xp = x
            else:
                if s > 0:
                    ancestors = self.resampling(w)
                    xp = x[ancestors]
                else:
                    xp = x

            # propagation
            x = self.sim_prop(s, xp, y)  # update the particle points using the proposition (proposal)

            # weighting
            if adaptive_resampling:
                # checks if adaptive sampling is enabled
                # if so, the method uses a different version of the log_weights function to update the logarithms of
                # the weights based on the current and previous particle points x and xp and the observation data y
                log_w = log_w + self.log_weights(s, x, xp, y)
            else:
                # if adaptive sampling is not enabled, the method uses the log_weights function to update the logarithms
                # of the weights based on the current and previous particle points x and xp and the observation data y
                log_w = self.log_weights(s, x, xp, y)
            max_log_w = torch.max(log_w)
            # calculates the maximum value of the logarithms of the weights
            w = torch.exp(log_w) / torch.exp(max_log_w)
            # calculates the weights by exponentiating the logarithms of the weights and subtracting the maximum value
            # from the logarithms of the weights

            if adaptive_resampling:
                if s == t - 1:
                    # checks if adaptive sampling is enabled
                    # if it is, and if it is the last time step, the method updates the log_z variable using the
                    # maximum value of the logarithms of the weights and the sum of the weights
                    log_z = log_z + max_log_w + torch.log(torch.sum(w)) - log(n)
            else:
                # if adaptive sampling is not activated or if it is not the last time step, the method updates the
                # log_z variable using the maximum value of the logarithms
                log_z = log_z + max_log_w + torch.log(torch.sum(w)) - log(n)
            w /= torch.sum(w)  # normalizes the weights by dividing them by their sum
            ess = 1. / torch.sum(torch.square(w)) / n
            # recalculates the effective support rate (ESS) of the particle points using the formula
        # returns the value of the lower bound of variational SMC (log_z) calculated at each step
        return log_z

    def sim_q(self, y):
        """
        Simulate hidden state trajectories of the model using the particle filter (PF)

        y: observation data
        verbose: a boolean indicating whether debugging information should be displayed

        Returns the path of the selected hidden state
        """
        # constants
        t = y.size(0)  # size of the observation data
        dx = self.dx  # number of dimensions of the hidden state of the model
        n = self.n  # retrieves the number of particle points used by the PKF

        # initialisation
        x = torch.zeros((n, t, dx))  # store the particle points at each step
        w = torch.zeros((n, t))  # will store the weights of the particle points at each step
        ess = torch.zeros(t)  # creates a vector of zeros of size t that will store the effective support rate (ESS)
        # values of the particle points at each step

        for s in range(t):
            # resampling
            if s > 0:
                # checks if it is not the first time step
                # if it is, the method performs resampling using the weights of the previous particle points
                # the selected particle points are stored in the ancestors variable and are used to update
                # the previous particle points x
                ancestors = self.resampling(w[:, s - 1])
                x[:, :s, :] = x[ancestors, :s, :]

            # Propagation
            x[:, s, :] = self.sim_prop(s, x[:, s - 1, :], y)

            # weighting
            log_w = self.log_weights(s, x[:, s, :], x[:, s - 1, :], y)
            # calls the log_weights method by passing it as input the current x[:, t, :] and previous x[:, s-1, :]
            # particle points as well as the observation data y.
            # the method returns the logarithms of the weights of the particle points
            max_log_w = torch.max(log_w)
            w[:, s] = torch.exp(log_w - max_log_w)
            # calculates the weights by exponentiating the logarithms of the weights and subtracting the maximum value
            # from the logarithms of the weights
            w[:, s] /= torch.sum(w[:, s])
            ess[s] = 1. / torch.sum(w[:, s] ** 2)

        # sample from the empirical approximation
        bins = torch.cumsum(w[:, -1], dim=-1)
        u = torch.rand(1)
        b = torch.bucketize(u, bins)
        # simulates a trajectory of the hidden state by selecting one of the particle points at random according to
        # their weights
        # to do this, it uses the torch.bucketize function which selects an index of an array based on the position of
        # a random value

        # returns the path of the selected hidden state
        return x[b, :, :]

    def sim_prop(self, t, x, y):
        raise NotImplementedError('VariationalSMC class requires sim_prop method.')

    def log_weights(self, t, x, xp, y):
        raise NotImplementedError('VariationalSMC class requires log_weights method.')
