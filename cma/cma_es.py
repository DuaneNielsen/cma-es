from math import floor, log, sqrt

import torch
from torch import nn as nn


def sample(n, sigma, mean, B, D):
    features = mean.size(0)
    z = torch.randn(features, n, device=mean.device, dtype=mean.dtype)
    s = mean.view(-1, 1) + sigma * B.matmul(D.matmul(z))
    return s.T, z.T


def simple_sample(features, n, mean, c):
    z = torch.randn(features, n, device=mean.device, dtype=mean.dtype)
    s = mean.view(-1, 1) + c.matmul(z)
    return s.T


def expect_multivariate_norm(N):
    return N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))


def get_policy(features, actions, depth):
    blocks = []
    for _ in range(0, depth):
        blocks += [nn.Linear(features, features), nn.ReLU()]
    return nn.Sequential(*blocks, nn.Linear(features, actions), nn.Softmax(dim=0))


class CMA(object):
    def _rank(self, results, rank_order='max'):
        if rank_order == 'max':
            ranked_results = sorted(results, key=lambda x: x['fitness'], reverse=True)
        elif rank_order == 'min':
            ranked_results = sorted(results, key=lambda x: x['fitness'])
        else:
            raise Exception(f'invalid value for kwarg type {rank_order}, valid values are max or min')

        return ranked_results

    def step(self, object_f, rank_order='max'):
        pass


class NaiveCovarianceMatrixAdaptation(CMA):
    def __init__(self, N, cma=None, samples=None, oversample=0):
        self.N = N
        self.recommended_steps = range(1, floor(1e3 * N ** 2))
        # variables
        self.mean = torch.zeros(N)
        self.c = torch.eye(N)
        self.samples = 4 + floor(3 * log(N)) * 2 if samples is None else samples
        self.oversample = oversample
        self.mu = self.samples // 2
        self.gen_count = 0
        self.cmu = self.mu / N ** 2 if cma is None else cma

    def step(self, objective_f, rank_order='max'):
        params = simple_sample(self.N, self.samples + floor(self.oversample), self.mean, self.c)
        self.oversample = self.oversample * 0.8
        # rank by fitness
        f = objective_f(params)
        results = [{'parameters': params[i], 'fitness': f.item()} for i, f in enumerate(f)]
        ranked_results = self._rank(results, rank_order)

        selected_results = ranked_results[0:self.mu]
        g = torch.stack([g['parameters'] for g in selected_results])

        mean_prev = self.mean.clone()
        self.mean = g.mean(0)
        g = g - mean_prev
        c_cma = torch.matmul(g.T, g) / self.mu
        self.c = (1 - self.cmu) * self.c + self.cmu * c_cma

        info = {'fitness_max': f.max(), 'fitness_mean': f.mean(), 'c_norm': self.c.norm()}
        self.gen_count += 1

        return ranked_results, info

    def __repr__(self):
        return f'N: {self.N}, samples: {self.samples}, mu: {self.mu}, cmu: {self.cmu}'


class SimpleCovarianceMatrixAdaptation(CMA):
    def __init__(self, N, cma=None, samples=None, oversample=0):
        self.N = N
        self.recommended_steps = range(1, floor(1e3 * N ** 2))

        self.samples = 4 + floor(3 * log(N)) * 2 if samples is None else samples
        self.oversample = oversample
        self.mu = self.samples // 2
        self.gen_count = 0
        self.cmu = self.mu / N ** 2 if cma is None else cma

        # variables
        self.mean = torch.zeros(N)
        self.b = torch.eye(N)
        self.d = torch.eye(N)
        self.c = torch.matmul(self.b.matmul(self.d), self.b.matmul(self.d).T)  # c = B D D B.T

    def step(self, objective_f, rank_order='max'):

        # sample parameters
        params, z = sample(self.samples + floor(self.oversample), 1.0, self.mean, self.b, self.d)
        self.oversample = self.oversample * 0.8

        # rank by fitness
        f = objective_f(params)
        results = [{'parameters': params[i], 'z': z[i], 'fitness': f.item()} for i, f in enumerate(f)]
        ranked_results = self._rank(results, rank_order)

        selected_results = ranked_results[0:self.mu]
        g = torch.stack([g['parameters'] for g in selected_results])
        z = torch.stack([g['z'] for g in selected_results])

        self.mean = g.mean(0)
        bdz = self.b.matmul(self.d).matmul(z.t())
        c_mu = torch.matmul(bdz, torch.eye(self.mu) / self.mu)
        c_mu = c_mu.matmul(bdz.t())

        self.c = (1 - self.cmu) * self.c + self.cmu * c_mu

        self.d, self.b = torch.symeig(self.c, eigenvectors=True)
        self.d = self.d.sqrt().diag_embed()

        info = {'fitness_max': f.max(), 'fitness_mean': f.mean(), 'c_norm': self.c.norm(), 'max_eigv': self.d.max()}
        self.gen_count += 1

        return ranked_results, info

    def __repr__(self):
        return f'N: {self.N}, samples: {self.samples}, mu: {self.mu}, cmu: {self.cmu}'


class FastCovarianceMatrixAdaptation(CMA):
    def __init__(self, N, step_mode='auto', step_decay=None, initial_step_size=None, samples=None, oversample=0.0):
        self.N = N
        self.recommended_steps = range(1, floor(1e3 * N ** 2))

        # selection settings
        self.samples = 4 + floor(3 * log(N)) if samples is None else samples
        self.oversample = oversample
        self.mu = self.samples / 2
        self.weights = torch.tensor([log(self.mu + 0.5)]) - torch.linspace(start=1, end=self.mu,
                                                                           steps=floor(self.mu)).log()
        self.weights = self.weights / self.weights.sum()
        self.weights = self.weights / self.weights.sum()
        self.mu = floor(self.mu)
        self.mueff = (self.weights.sum() ** 2 / (self.weights ** 2).sum()).item()

        # adaptation settings
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cmu = 2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + 2 * self.mueff / 2)
        self.damps = 1 + 2 * max(0.0, sqrt((self.mueff - 1.0) / (N + 1)) - 1) + self.cs
        self.chiN = expect_multivariate_norm(N)
        self.step_size = 0.5 if initial_step_size is None else initial_step_size
        self.step_mode = step_mode
        if step_mode == 'decay' and step_decay is None:
            raise Exception('decay mode requires you set a step decay')
        self.step_decay = (1.0 - step_decay) if step_decay is not None else None

        # variables
        self.mean = torch.zeros(N)
        self.b = torch.eye(N)
        self.d = torch.eye(N)
        self.c = torch.matmul(self.b.matmul(self.d), self.b.matmul(self.d).T)  # c = B D D B.T

        self.pc = torch.zeros(N)
        self.ps = torch.zeros(N)
        self.gen_count = 1

    def step(self, objective_f, rank_order='max'):

        # sample parameters
        s, z = sample(self.samples + floor(self.oversample), self.step_size, self.mean, self.b, self.d)
        self.oversample = self.oversample * 0.8

        # rank by fitness
        f = objective_f(s)
        results = [{'parameters': s[i], 'z': z[i], 'fitness': f.item()} for i, f in enumerate(f)]

        if rank_order == 'max':
            ranked_results = sorted(results, key=lambda x: x['fitness'], reverse=True)
        elif rank_order == 'min':
            ranked_results = sorted(results, key=lambda x: x['fitness'])
        else:
            raise Exception(f'invalid value for kwarg type {rank_order}, valid values are max or min')

        selected_results = ranked_results[0:self.mu]
        z = torch.stack([g['z'] for g in selected_results])
        g = torch.stack([g['parameters'] for g in selected_results])

        self.mean = (g * self.weights.unsqueeze(1)).sum(0)
        zmean = (z * self.weights.unsqueeze(1)).sum(0)

        # step size
        self.ps = (1 - self.cs) * self.ps + sqrt(self.cs * (2.0 - self.cs)) * self.b.matmul(zmean)

        correlation = self.ps.norm() / self.chiN

        # delay the introduction of the rank 1 update
        denominator = sqrt(1 - (1 - self.cs) ** (2 * self.gen_count / self.samples))
        threshold = 1.4e2 / self.N + 1
        hsig = correlation / denominator < threshold
        hsig = 1.0 if hsig else 0.0

        # adapt step size
        if self.step_mode == 'auto':
            self.step_size = self.step_size * ((self.cs / self.damps) * (correlation - 1.0)).exp()
        elif self.step_mode == 'nodamp':
            self.step_size = self.step_size * (correlation - 1.0).exp()
        elif self.step_mode == 'decay':
            self.step_size = self.step_size * self.step_decay
        elif self.step_mode == 'constant':
            pass
        else:
            raise Exception('step_mode must be auto | nodamp | decay | constant')

        # a mind bending way to write a exponential smoothed moving average
        # zmean does not contain step size or mean, so allows us to add together
        # updates of different step sizes
        self.pc = (1 - self.cc) * self.pc + hsig * sqrt(self.cc * (2.0 - self.cc) * self.mueff) * self.b.matmul(
            self.d).matmul(zmean)
        # which we then combine to make a covariance matrix, from 1 (mean) datapoint!
        # this is why it's called "rank 1" update
        pc_cov = self.pc.unsqueeze(1).matmul(self.pc.unsqueeze(1).t())
        # mix back in the old covariance if hsig == 0
        pc_cov = pc_cov + (1 - hsig) * self.cc * (2 - self.cc) * self.c

        # estimate cov for all selected samples (weighted by rank)
        bdz = self.b.matmul(self.d).matmul(z.t())
        cmu_cov = torch.matmul(bdz, self.weights.diag_embed())
        cmu_cov = cmu_cov.matmul(bdz.t())

        self.c = (1.0 - self.c1 - self.cmu) * self.c + (self.c1 * pc_cov) + (self.cmu * cmu_cov)

        # pull out the eigenthings and do the business
        self.d, self.b = torch.symeig(self.c, eigenvectors=True)
        self.d = self.d.sqrt().diag_embed()
        self.gen_count += 1

        info = {'step_size': self.step_size, 'correlation': correlation,
                'fitness_max': f.max(), 'fitness_mean': f.mean(), 'c_norm': self.c.norm(),
                'max_eigv': self.d.max()}
        return ranked_results, info

    def __repr__(self):
        return f'N: {self.N}, samples: {self.samples}, mu: {self.mu}, mueff: {self.mueff}, cc: {self.cc}, ' \
               f'cs: {self.cs}, c1: {self.c1}, cmu: {self.cmu}, damps: {self.damps}, chiN: {self.chiN}, ' \
               f'step_mode: {self.step_mode}, step_decay: {self.step_decay}, step_size: {self.step_size}'


