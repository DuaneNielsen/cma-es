import torch
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color


def rastrigin(x):
    n = 10
    a = 10
    from math import pi

    return a*n + x ** 2 - a * torch.cos(2 * pi * x)


def top_25_percent(scores, higher_is_better=True):
    """
    Calculates the top 25 best scores
    :param scores: a list of the scores
    :return: a longtensor with indices of the top 25 scores
    """
    indexed = [(i, s) for i, s in enumerate(scores)]
    indexed = sorted(indexed, key=lambda score: score[1], reverse=higher_is_better)
    best = [indexed[i][0] for i in range(len(indexed)//4)]
    rest = [indexed[i][0] for i in range(len(indexed)//4+1, len(indexed))]
    return torch.tensor(best), torch.tensor(rest)


n = 100
x = torch.linspace(-5.12, 5.12, n)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1 = ax.plot(x.cpu().numpy(), rastrigin(x).cpu().numpy())
line2 = None
line3 = None

mu = torch.Tensor(np.random.uniform(size=1) + 2.0)
sigma = torch.Tensor(np.random.uniform(size=1) * 5.0)

for epoch in range(500):
    space = dist.Normal(mu, sigma)
    parameters = space.sample((24,))
    scores = rastrigin(parameters)
    scores_l = scores.split(1, dim=0)

    best, rest = top_25_percent(scores_l, higher_is_better=False)
    best_individual = parameters[best[0]]
    best_score = rastrigin(best_individual[0].unsqueeze(0))
    print(best_score)

    mu = torch.index_select(parameters, 0, best).mean(0)
    sigma = torch.sqrt(parameters.var(0))

    if line2 is None:
        line2 = ax.scatter(parameters[best].cpu().numpy(), scores[best].cpu().numpy(), label='the best', c=color.to_rgb('r'))
        line3 = ax.scatter(parameters[rest].cpu().numpy(), scores[rest].cpu().numpy(), label='the rest', c=color.to_rgb('b'))
        ax.legend()
    else:
        line2.set_offsets(np.c_[parameters[best].cpu().numpy(), scores[best].cpu().numpy()])
        line3.set_offsets(np.c_[parameters[rest].cpu().numpy(), scores[rest].cpu().numpy()])
    fig.canvas.draw_idle()
    fig.savefig('images/cma%04d.png' % (epoch,), bbox_inches='tight')
    plt.pause(0.2)



