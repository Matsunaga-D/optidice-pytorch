import os
import time

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import util
from networks_th import TanhMixtureNormalPolicy, TanhNormalPolicy, ValueNetwork

np.set_printoptions(precision=3, suppress=True)


class NeuralDICERL(nn.Module):
    """Offline policy Optimization via Stationary DIstribution Correction Estimation (OptiDICE)"""

    def __init__(self, observation_spec, action_spec, config):
        super(NeuralDICERL, self).__init__()

        self._gamma = config['gamma']
        self._policy_extraction = config['policy_extraction']
        self._env_name = config['env_name']
        self._total_iterations = config['total_iterations']
        self._warmup_iterations = config['warmup_iterations']
        self._data_policy = config['data_policy']
        self._data_policy_num_mdn_components = config['data_policy_num_mdn_components']
        self._data_policy_mdn_temperature = config['data_policy_mdn_temperature']
        self._use_policy_entropy_constraint = config['use_policy_entropy_constraint']
        self._use_data_policy_entropy_constraint = config['use_data_policy_entropy_constraint']
        self._target_entropy = config['target_entropy']
        self._hidden_sizes = config['hidden_sizes']
        self._batch_size = config['batch_size']
        self._alpha = config['alpha']
        self._f = config['f']
        self._gendice_v = config['gendice_v']
        self._gendice_e = config['gendice_e']
        self._gendice_loss_type = config['gendice_loss_type']
        self._lr = config['lr']
        self._e_loss_type = config['e_loss_type']
        self._v_l2_reg = config['v_l2_reg']
        self._lamb_scale = config['lamb_scale']


def run(config):
    np.random.seed(config['seed'])
    th.manual_seed(config['seed'])

    # load dataset
    env = gym.make(config['env_name'])
    env.seed(config['seed'])
    initial_obs_dataset, dataset, dataset_statistics = util.dice_dataset(env, standardize_observation=True, absorbing_state=config['absorbing_state'], standardize_reward=config['standardize_reward'])
    if config['use_policy_entropy_constraint'] or config['use_data_policy_entropy_constraint']:
        if config['target_entropy'] is None:
            config['target_entropy'] = -np.prod(env.action_space.shape)

    print(f'observation space: {env.observation_space.shape}')
    print(f'- high: {env.observation_space.high}')
    print(f'- low: {env.observation_space.low}')
    print(f'action space: {env.action_space.shape}')
    print(f'- high: {env.action_space.high}')
    print(f'- low: {env.action_space.low}')

    def _sample_minibatch(batch_size, reward_scale):
        initial_indices = np.random.randint(0, dataset_statistics['N_initial_observations'], batch_size)
        indices = np.random.randint(0, dataset_statistics['N'], batch_size)
        sampled_dataset = (
            initial_obs_dataset['initial_observations'][initial_indices],
            dataset['observations'][indices],
            dataset['actions'][indices],
            dataset['rewards'][indices] * reward_scale,
            dataset['next_observations'][indices],
            dataset['terminals'][indices]
        )
        return tuple(map(tf.convert_to_tensor, sampled_dataset))

    # Create an agent
    agent = NeuralDICERL(
        observation_spec=TensorSpec(
            (dataset_statistics['observation_dim'] + 1) if config['absorbing_state'] else dataset_statistics['observation_dim']),
        action_spec=TensorSpec(dataset_statistics['action_dim']),
        config=config
    )

    result_logs = []
    start_iteration = 0

    # Start training
    start_time = time.time()
    last_start_time = time.time()
    for iteration in tqdm(range(start_iteration, config['total_iterations'] + 1), ncols=70, desc='DICE', initial=start_iteration, total=config['total_iterations'] + 1, ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
        # Sample mini-batch data from dataset
        initial_observation, observation, action, reward, next_observation, terminal = _sample_minibatch(config['batch_size'], config['reward_scale'])

        # Perform gradient descent
        train_result = agent.train_step(initial_observation, observation, action, reward, next_observation, terminal)
        if iteration % config['log_iterations'] == 0:
            train_result = {k: v.numpy() for k, v in train_result.items()}
            if iteration >= config['warmup_iterations']:
                # evaluation via real-env rollout
                eval = util.evaluate(env, agent, dataset_statistics, absorbing_state=config['absorbing_state'], pid=config.get('pid'))
                train_result.update({'iteration': iteration, 'eval': eval})
            train_result.update({'iter_per_sec': config['log_iterations'] / (time.time() - last_start_time)})

            result_logs.append({'log': train_result, 'step': iteration})
            if not int(os.environ.get('DISABLE_STDOUT', 0)):
                print(f'=======================================================')
                for k, v in sorted(train_result.items()):
                    print(f'- {k:23s}:{v:15.10f}')
                if train_result.get('eval'):
                    print(f'- {"eval":23s}:{train_result["eval"]:15.10f}')
                print(f'config={config}')
                print(f'iteration={iteration} (elapsed_time={time.time() - start_time:.2f}s, {train_result["iter_per_sec"]:.2f}it/s)')
                print(f'=======================================================', flush=True)

            last_start_time = time.time()


if __name__ == "__main__":
    from default_config import get_parser
    args = get_parser().parse_args()
    run(vars(args))
