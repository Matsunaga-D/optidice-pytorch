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
from networks_th import TanhNormalPolicy, ValueNetwork #, TanhMixtureNormalPolicy

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

        self._iteration = 0
        # create networks / variables for DICE-RL
        self._v_network = ValueNetwork((observation_spec,), hidden_sizes=self._hidden_sizes, output_activation_fn=None,
                                       name='v')
        self._optimizer_v_network = th.optim.Adam(self._v_network.parameters(), self._lr)

        self._e_network = ValueNetwork((observation_spec, action_spec), hidden_sizes=self._hidden_sizes, name='e')
        self._optimizer_e_network = th.optim.Adam(self._e_network.parameters(), self._lr)

        # GenDICE regularization, i.e., E[w] = 1.
        if self._gendice_v:
            self._lamb_v = th.autograd.Variable(0.0, dtype=th.float32, requires_grad=True, name='lamb_v')
            self._optimizer_lamb_v = th.optim.Adam(self._lamb_v.parameters(), self._lr)
        else:
            self._lamb_v = 0

        if self._gendice_e:
            self._lamb_e = th.autograd.Variable(0.0, dtype=th.float32, requires_grad=True, name='_lamb_e')
            self._optimizer_lamb_e = th.optim.Adam(self._lamb_e.parameters(), self._lr)
        else:
            self._lamb_e = 0

        if self._f == 'elu': # TODO: bug? should be -x? -1?
            self._f_fn = lambda x: th.where(x < 1, x * (th.log(x + 1e-10) - x) + 1, 0.5 * (x - 1)**2)
            self._f_prime_inv_fn = lambda x: th.where(x < 0, th.exp(th.minimum(x, th.Tensor(0))), x + 1) # TODO: why min?
            self._g_fn = lambda x: th.where
        else:
            raise NotImplementedError()

            # policy
        self._policy_network = TanhNormalPolicy((observation_spec,), action_spec.shape[0],
                                                hidden_sizes=self._hidden_sizes,
                                                mean_range=config['mean_range'], logstd_range=config['logstd_range'])
        self._optimizer_policy = th.optim.Adam(self._policy_network.parameters(), self._lr)

        if self._use_policy_entropy_constraint:
            self._log_ent_coeff = th.autograd.Variable(0.0, dtype=th.float32, requires_grad=True, name='ent_coeff')
            self._optimizer_ent_coeff = th.optim.Adam(self._log_ent_coeff.parameters(), self._lr)

        # TODO: data policy

    def _zero_grad(self):
        self._optimizer_e_network.zero_grad()
        self._optimizer_v_network.zero_grad()
        self._optimizer_policy.zero_grad()
        self._optimizer_ent_coeff.zero_grad()
        self._optimizer_lamb_e.zero_grad()
        self._optimizer_lamb_v.zero_grad()

    def v_loss(self, initial_v_values, e_v, w_v, f_w_v, result=None):
        # Compute v loss
        if result is None:
            result = {}

        v_loss0 = (1 - self._gamma) * th.mean(initial_v_values)
        v_loss1 = th.mean(- self._alpha * f_w_v)
        if self._gendice_loss_type == 'gendice':
            v_loss2 = th.mean(w_v * (e_v - self._lamb_scale * self._lamb_v))
            v_loss3 = self._lamb_scale * (self._lamb_v + self._lamb_v ** 2 / 2)
        elif self._gendice_loss_type == 'bestdice':
            v_loss2 = th.mean(w_v * (e_v - self._lamb_v))
            v_loss3 = self._lamb_v
        else:
            raise NotImplementedError
        v_loss = v_loss0 + v_loss1 + v_loss2 + v_loss3

        v_l2_norm = th.linalg.norm(self._v_network.variables) # TODO

        if self._v_l2_reg is not None:
            v_loss += self._v_l2_reg * v_l2_norm

        result.update({
            'v_loss0': v_loss0,
            'v_loss1': v_loss1,
            'v_loss2': v_loss2,
            'v_loss3': v_loss3,
            'v_loss': v_loss,
            'v_l2_norm': v_l2_norm
        })

        return result

    def e_loss(self, e_v, e_values, w_e, f_w_e, result=None):
        # Compute e loss
        if result is None:
            result = {}

        if self._e_loss_type == 'minimax':
            e_loss = th.mean(self._alpha * f_w_e - w_e * (e_v - self._lamb_scale * self._lamb_e))
        elif self._e_loss_type == 'mse':
            e_loss = th.mean((e_v - e_values) ** 2)
        else:
            raise NotImplementedError

        e_l2_norm = th.linalg.norm(self._e_network.variables) # TODO

        result.update({
            'e_loss': e_loss,
            'e_l2_norm': e_l2_norm,
        })

        return result

    def policy_loss(self, observation, action, w_e, result=None):
        # Compute policy loss
        if result is None:
            result = {}
        (sampled_action, sampled_pretanh_action, sampled_action_log_prob, sampled_pretanh_action_log_prob, pretanh_action_dist), _ \
            = self._policy_network((observation,))
        # Entropy is estimated on newly sampled action.
        negative_entropy_loss = th.mean(sampled_action_log_prob)

        policy_l2_norm = th.linalg.norm(self._policy_network.variables)

        if self._policy_extraction == 'wbc':
            # Weighted BC
            action_log_prob, _ = self._policy_network.log_prob(pretanh_action_dist, action, is_pretanh_action=False)
            policy_loss = - th.mean(w_e * action_log_prob)

        elif self._policy_extraction == 'iproj':
            # Information projection
            (_, _, _, _, data_pretanh_action_dist), _ = self._data_policy_network((observation,))

            sampled_e_values, _ = self._e_network((observation, sampled_action))
            if self._gendice_loss_type == 'gendice':
                sampled_log_w_e = self._log_r_fn((sampled_e_values - self._lamb_scale * self._lamb_e) / self._alpha)
            elif self._gendice_loss_type == 'bestdice':
                sampled_log_w_e = self._log_r_fn((sampled_e_values - self._lamb_e) / self._alpha)
            else:
                raise NotImplementedError()

            _, sampled_pretanh_action_data_log_prob = self._data_policy_network.log_prob(data_pretanh_action_dist, sampled_pretanh_action)
            kl = sampled_pretanh_action_log_prob - sampled_pretanh_action_data_log_prob
            policy_loss = - th.mean(sampled_log_w_e - kl)

            result.update({'kl': th.mean(kl)})

        else:
            raise NotImplementedError()

        if self._use_policy_entropy_constraint:
            ent_coeff = th.exp(self._log_ent_coeff)
            policy_loss += ent_coeff * negative_entropy_loss

            ent_coeff_loss = - self._log_ent_coeff * (sampled_action_log_prob + self._target_entropy)

            result.update({
                'ent_coeff_loss': th.mean(ent_coeff_loss),
                'ent_coeff': ent_coeff,
            })

        result.update({
            'policy_loss': policy_loss,
            'policy_l2_norm': policy_l2_norm,
            'negative_entropy_loss': negative_entropy_loss
        })

        return result

    def train_step(self, initial_observation, observation, action, reward, next_observation, terminal):
        # Shared network values
        initial_v_values, _ = self._v_network((initial_observation,))
        v_values, _ = self._v_network((observation,))
        next_v_values, _ = self._v_network((next_observation,))

        e_v = reward + (1 - terminal) * self._gamma * next_v_values - v_values
        preactivation_v = (e_v - self._lamb_scale * self._lamb_v) / self._alpha
        w_v = self._r_fn(preactivation_v)
        f_w_v = self._g_fn(preactivation_v)

        e_values, _ = self._e_network((observation, action))
        preactivation_e = (e_values - self._lamb_scale * self._lamb_e) / self._alpha
        w_e = self._r_fn(preactivation_e)
        f_w_e = self._g_fn(preactivation_e)

        # compute loss within GradientTape context manager
        loss_result = self.v_loss(initial_v_values, e_v, w_v, f_w_v, result={})
        if self._gendice_v:
            loss_result = self.lamb_v_loss(e_v, w_v, f_w_v, result=loss_result)

        loss_result = self.e_loss(e_v.detach(), e_values, w_e, f_w_e, result=loss_result)
        if self._gendice_e:
            loss_result = self.lamb_e_loss(e_v.detach(), w_e, f_w_e, result=loss_result)

        loss_result = self.policy_loss(observation, action, w_e.detach(), result=loss_result)
        if self._policy_extraction == 'iproj':
            loss_result = self.data_policy_loss(observation, action, result=loss_result)

        self._zero_grad()

        loss_result['v_loss'].backward()
        self._optimizer_v_network.step()

        if self._gendice_v:
            loss_result['lamb_v_loss'].backward()
            self._optimizer_lamb_v.step()

        loss_result['e_loss'].backward()
        self._optimizer_e_network.step()

        if self._gendice_e:
            loss_result['lamb_e_loss'].backward()
            self._optimizer_lamb_e.step()

        if self._iteration >= self._warmup_iterations:
            loss_result['policy_loss'].backward()
            self._optimizer_policy.step()

            if self._use_policy_entropy_constraint:
                loss_result['ent_coeff_loss'].backward()
                self._optimizer_ent_coeff.step()

        if self._policy_extraction == 'iproj':
            # TODO
            raise NotImplementedError

        self._iteration += 1

        return loss_result


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
