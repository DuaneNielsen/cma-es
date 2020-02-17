import gym
import torch
import torch.nn as nn
from keypoints.models import knn
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import config
from torch.distributions import Categorical
import torchvision.transforms.functional as TVF

from keypoints.models import functional as KF
from keypoints.ds import datasets as ds
import gym_wrappers
import torch.multiprocessing as mp
import numpy as np
from torchvision.transforms import functional as TVF

from cma_es import NaiveCovarianceMatrixAdaptation, SimpleCovarianceMatrixAdaptation, FastCovarianceMatrixAdaptation, \
    get_policy
from eval import AtariMpEvaluator, AtariSpEvaluator, nop


class Keypoints(nn.Module):
    def __init__(self, transporter_net):
        super().__init__()
        self.transporter_net = transporter_net

    def forward(self, s_t):
        heatmap = self.transporter_net.keypoint(s_t)
        kp = KF.spacial_logsoftmax(heatmap)
        return kp


class KeypointsAndPatches(nn.Module):
    def __init__(self, transporter_net):
        super().__init__()
        self.transporter_net = transporter_net

    def forward(self, s_t):
        heatmap = self.transporter_net.keypoint(s_t)
        kp = KF.spacial_logsoftmax(heatmap)
        return kp


def evaluate(args, weights, features, depth, render=False, record=False):

    datapack = ds.datasets[args.dataset]
    env = gym.make(datapack.env)
    if args.gym_reward_count_limit is not None:
        env = gym_wrappers.RewardCountLimit(env, args.gym_reward_count_limit)

    actions = datapack.action_map.size
    policy = get_policy(features, actions, depth)
    policy = knn.load_weights(policy, weights)
    policy = policy.to(args.device)
    policy_dtype = next(policy.parameters()).dtype

    video = []

    with torch.no_grad():

        def get_action(s, prepro, transform, view, policy, action_map, device, action_select_mode='argmax'):
            s = prepro(s)
            s_t = transform(s).unsqueeze(0).type(policy_dtype).to(device)
            kp = view(s_t)
            p = policy(kp.flatten())
            if action_select_mode == 'argmax':
                a = torch.argmax(p)
            if action_select_mode == 'sample':
                a = Categorical(p).sample()
            a = action_map(a)
            return a, kp

        v = UniImageViewer()

        if args.model_type != 'nop':
            transporter_net = transporter.make(args, map_device='cpu')
            view = Keypoints(transporter_net).to(args.device)

        else:
            view = nop

        s = env.reset()
        a, kp = get_action(s, datapack.prepro, datapack.transforms, view, policy, datapack.action_map, args.device,
                           action_select_mode=args.policy_action_select_mode)

        done = False
        reward = 0.0

        while not done:
            s, r, done, i = env.step(a)
            reward += r

            a, kp = get_action(s, datapack.prepro, datapack.transforms, view, policy, datapack.action_map, args.device,
                               action_select_mode=args.policy_action_select_mode)

            if render or record:
                if args.model_keypoints:
                    s = datapack.prepro(s)
                    s = TVF.to_tensor(s).unsqueeze(0)
                    s = plot_keypoints_on_image(kp[0], s[0])
                    if render:
                        v.render(s)
                    if record:
                        video.append(s)
                else:
                    video.append(env.render(mode='rgb_array'))
                    if render:
                        env.render()

    if record and len(video) != 0:
        video = np.expand_dims(np.stack(video), axis=0)
        video = torch.from_numpy(video).permute(0, 1, 4, 2, 3)
        tb.add_video('action replay', video, global_step)

    return reward


if __name__ == '__main__':

    mp.set_start_method('spawn')

    args = config.config()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    datapack = ds.datasets[args.dataset]
    log_dir = f'data/cma_es/{datapack.env}/{args.run_id}/'
    tb = SummaryWriter(log_dir)
    global_step = 0
    best_reward = -1e8
    show = False

    if args.model_keypoints:
        policy_features = args.model_keypoints * 2
    else:
        policy_features = args.policy_inputs

    N = knn.parameter_count(get_policy(policy_features, args.policy_depth, datapack.action_map.size))

    evaluator = AtariMpEvaluator(args, datapack, policy_features, datapack.action_map.size, args.policy_depth)
    demo = AtariSpEvaluator(args, datapack, policy_features, datapack.action_map.size, args.policy_depth,
                            render=args.display, record=True)

    if args.cma_algo == 'fast':
        cma = FastCovarianceMatrixAdaptation(N=N,
                                             step_mode=args.cma_step_mode,
                                             step_decay=args.cma_step_decay,
                                             initial_step_size=args.cma_initial_step_size,
                                             samples=args.cma_samples,
                                             oversample=args.cma_oversample)
    elif args.cma_algo == 'naive':
        cma = NaiveCovarianceMatrixAdaptation(N=N,
                                              samples=args.cma_samples,
                                              oversample=args.cma_oversample)
    elif args.cma_algo == 'simple':
        cma = SimpleCovarianceMatrixAdaptation(N=N,
                                               samples=args.cma_samples,
                                               oversample=args.cma_oversample)
    else:
        raise Exception('--cma_algo fast | naive | simple')

    tb.add_text('args', str(args), global_step)
    tb.add_text('cma_params', str(cma), global_step)

    for step in cma.recommended_steps:

        ranked_results, info = cma.step(evaluator.fitness)

        print([result['fitness'] for result in ranked_results])

        for key, value in info.items():
            tb.add_scalar(key, value, global_step)

        if ranked_results[0]['fitness'] > best_reward:
            best_reward = ranked_results[0]['fitness']
            torch.save(ranked_results[0]['parameters'], log_dir + 'best_of_generation.pt')
            _p = get_policy(policy_features, datapack.action_map.size, args.policy_depth)
            _p = knn.load_weights(_p, weights=ranked_results[0]['parameters'])
            torch.save(_p.state_dict(), log_dir + 'best_policy.mdl')
            show = True

        if global_step % args.display_freq == 0 or show:
            demo.fitness(ranked_results[0]['parameters'].unsqueeze(0))

        show = False
        global_step += 1

        if args.epochs is not None and step >= args.epochs:
            break


