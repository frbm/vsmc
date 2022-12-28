from math import log
import torch

torch.manual_seed(0)


class VariationalSMC:
    def __init__(self, dx, dy, alpha, r, obs, n, t, scale):
        self.dx = dx
        self.dy = dy
        self.alpha = alpha
        self.r = r
        self.obs = obs
        self.n = n

        self.model_params = self.init_model_params(dx, dy, alpha, r, obs)
        self.prop_params = self.init_prop_params(t, dx, scale)

    @staticmethod
    def init_model_params(dx, dy, alpha, r, obs):
        mu_0 = torch.zeros(dx)
        s_0 = torch.eye(dx)

        a = torch.zeros((dx, dx))
        for i in range(dx):
            for j in range(dx):
                a[i, j] = alpha ** (1 + abs(i - j))

        q = torch.eye(dx)
        c = torch.zeros((dy, dx))
        if obs == 'sparse':
            c[:dy, :dy] = torch.eye(dy)
        else:
            c = torch.randn((dy, dx))
        rr = r * torch.eye(dy)
        return mu_0, s_0, a, q, c, rr

    @staticmethod
    def init_prop_params(t, dx, scale=0.5):
        out = []
        for _ in range(t):
            bias = scale * torch.randn(dx)
            times = 1 + scale * torch.randn(dx)
            log_var = scale * torch.randn(dx)
            out += [bias.requires_grad_(True), times.requires_grad_(True), log_var.requires_grad_(True)]
        return out

    def generate_data(self, *args):
        raise NotImplementedError

    def log_marginal_likelihood(self, *args):
        raise NotImplementedError

    @torch.no_grad()
    def resampling(self, w):
        n = w.size(0)
        bins = torch.cumsum(w, dim=-1)
        ind = torch.arange(n)
        u = (ind + torch.rand(n))/n
        return torch.bucketize(u, bins)

    def forward(self, y, adaptive_resampling=False, verbose=False):
        """VSMC Lower Bound"""
        # constants
        t = y.size(0)
        dx = self.dx
        n = self.n

        # initialisation
        x = torch.zeros((n, dx))
        log_w = torch.zeros(n)
        w = torch.exp(log_w)
        w /= w.sum()
        log_z = 0.
        ess = 1/torch.sum(torch.square(w))/n

        for s in range(t):
            # resampling
            if adaptive_resampling:
                if ess < 0.5:
                    ancestors = self.resampling(w)
                    xp = x[ancestors]
                    log_z += torch.max(log_w) + torch.log(torch.sum(w)) - log(n)
                    log_w = torch.zeros(n)
                else:
                    xp = x.clone()
            else:
                if t > 0:
                    ancestors = self.resampling(w)
                    xp = x[ancestors]
                else:
                    xp = x.clone()

            # propagation
            x = self.sim_prop(s, xp)

            # weighting
            if adaptive_resampling:
                log_w += self.log_weights(s, x, xp, y)
            else:
                log_w = self.log_weights(s, x, xp, y)
            max_log_w = torch.max(log_w)
            w = torch.exp(log_w) / torch.exp(max_log_w)

            if adaptive_resampling:
                if s == t - 1:
                    log_z += max_log_w + torch.log(torch.sum(w)) - torch.log(w)
            else:
                log_z += max_log_w + torch.log(torch.sum(w)) - log(n)

            w /= w.sum()
            ess = 1/torch.sum(torch.square(w))/n

            if verbose:
                print(f'ESS: {ess}.')

            return log_z

    def log_weights(self, *args):
        raise NotImplementedError

    def sim_prop(self, *args):
        raise NotImplementedError

    def sim_q(self, y, verbose=False):
        # constants
        t = y.size(0)
        dx = self.dx
        n = self.n

        # initialisation
        x = torch.zeros((n, t, dx))
        w = torch.zeros((n, t))
        ess = torch.zeros(t)

        for s in range(t):
            # resampling
            if s > 0:
                ancestors = self.resampling(w[:, s-1])
                x[:, :s, :] = x[ancestors, :s, :]

            # propagation
            x[:, s, :] = self.sim_prop(s, x[:, s-1, :])

            # weighting
            log_w = self.log_weights(s, x[:, s, :], x[:, s-1, :], y)
            max_log_w = torch.max(log_w)
            w[:, s] = torch.exp(log_w - max_log_w)
            w[:, s] /= w[:, s].sum()
            ess[s] = 1/torch.sum(torch.square(w[:, s]))

        # sample from the empirical approximation
        bins = torch.cumsum(w[:, -1], dim=-1)
        u = torch.rand(1)
        b = torch.bucketize(u, bins)

        if verbose:
            print(f'Mean ESS: {ess.mean()}')
            print(f'Lowest ESS: {torch.min(ess)}')

        return x[b, :, :]
