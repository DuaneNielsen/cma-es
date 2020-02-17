import torch
from torch import multiprocessing as mp


def nop(s_t):
    return s_t


class EvalPacket():
    def __init__(self, args, datapack, weights, features, depth, render):
        """
        Serializable arguments for eval function
        :param args:
        :param datapack:
        :param weights:
        :param render:
        """
        self.args = args
        self.datapack = datapack
        self.weights = weights
        self.render = render
        self.features = features
        self.depth = depth


def encode(args, datapack, weights, features, depth, render):
    weights = weights.cpu().numpy()
    return EvalPacket(args, datapack, weights, features, depth, render)


def decode(packet):
    packet.weights = torch.from_numpy(packet.weights)
    return packet


class AtariEval(object):
    def __init__(self, eval_func, args, datapack, policy_features, policy_actions, policy_depth, render=False, record=False):
        self.args = args
        self.datapack = datapack
        self.policy_features = policy_features
        self.policy_actions = policy_actions
        self.policy_depth = policy_depth
        self.render = render
        self.record = record
        self.eval_func = eval_func

    def call_evaluate(self, packet):
        packet = decode(packet)
        return self.eval_func(packet.args, packet.weights, packet.features, packet.depth, render=packet.render)


class AtariMpEvaluator(AtariEval):
    def __init__(self, eval_func, workers, args, datapack, policy_features, policy_actions, policy_depth, render=False, record=False):
        super().__init__(eval_func, args, datapack, policy_features, policy_actions, policy_depth, render=render, record=record)
        self.workers = workers

    def fitness(self, candidates):
        weights = torch.unbind(candidates, dim=0)

        worker_args = [encode(self.args, self.datapack, w, self.policy_features, self.policy_depth, self.render) for w in weights]

        with mp.Pool(processes=self.workers) as pool:
            results = pool.map(self.call_evaluate, worker_args)

        results = torch.tensor(results)
        return results


class AtariSpEvaluator(AtariEval):
    def __init__(self, eval_func, args, datapack, policy_features, policy_actions, policy_depth, render=False, record=False):
        super().__init__(eval_func, args, datapack, policy_features, policy_actions, policy_depth, render=render, record=record)

    def fitness(self, weights):
        return [self.eval_func(self.args, w, self.policy_features, self.policy_depth, self.render, self.record) for w in torch.unbind(weights, dim=0)]