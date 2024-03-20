# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from amp.utils.torch_utils import *
from amp.utils.torch_jit_utils import quat_to_tan_norm, quat_diff_rad
import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn

import amp.learning.replay_buffer as replay_buffer
import amp.learning.common_agent as common_agent 
import wandb
from tensorboardX import SummaryWriter
# SMPL
from amp.tasks.smpl_rig_amp import build_amp_observations
from amp.tasks.amp.smpl_rig_amp_base import dof_to_obs,dof_to_diff

# from util import r2rpy, quat2r

class AMPAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        return

    def init_tensors(self):
        super().init_tensors()
        self._build_amp_buffers()
        return
    
    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self.normalize_value:
            self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
        return

    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list
        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done()
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
            if (self.vec_env.env.viewer and (n == (self.horizon_length - 1))):
                self._amp_debug(infos)

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict
    
    def env_step(self, test=False):
        obs, rewards, dones, infos = self.vec_env.step(self.model, self.running_mean_std, self.value_mean_std, test=test)

        if self.value_size == 1:
            rewards = rewards.unsqueeze(2)
        return self.obs_to_tensors(obs), rewards.to(self.ppo_device).float(), dones.to(self.ppo_device), infos

    def play_steps_ray(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list
        # ____ start for ________
        self.obs, rewards, self.dones, infos = self.env_step()

        for n, (obs, reward, done, amp_obs, terminate, prev_obs, motion_time) in enumerate(zip(self.obs['obs'].transpose(0,1), rewards.transpose(0,1), self.dones.transpose(0,1), infos['amp_obs'].transpose(0,1), infos['terminate'].transpose(0,1), infos['prev_obses'].transpose(0,1), infos['motion_times'].transpose(1,0))):
            
            # self.experience_buffer.update_data('obses', n, torch.cat((prev_obs.unsqueeze(dim=0), obs[1:,:])))

            for k in update_list:
                self.experience_buffer.update_data(k, n, infos['res_dicts'][k].transpose(0,1)[n])

            # if self.has_central_value:
            #     self.experience_buffer.update_data('states', n, self.obs['states'])

            shaped_rewards = self.rewards_shaper(reward)
            self.experience_buffer.update_data('obses', n, prev_obs)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, obs)
            self.experience_buffer.update_data('dones', n, done)
            self.experience_buffer.update_data('amp_obs', n, amp_obs)

            terminated = terminate.float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(obs).to(self.device)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += reward
            self.current_lengths += 1
            all_done_indices = done.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - done.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
            if (self.vec_env.env.viewer and (n == (self.horizon_length - 1))):
                self._amp_debug(infos)
        # ____ end for ________
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        # y0-0n: AMP
        # mb_obs = self.experience_buffer.tensor_dict['obses']
        # mb_motion_time = self.experience_buffer.tensor_dict['motion_times']
        # amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        deepmimic_rewards = self._calc_deepmimic_rewards(self.obs['obs'], infos['motion_times'].transpose(0,1)) # TODO: Check next observation is right
        mb_rewards = deepmimic_rewards[0]
        # mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        # y0-0n: AMP
        # for k, v in amp_rewards.items():
        #     batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict


    def prepare_dataset(self, batch_dict):
        # batch_dict=self.concat_MPC_dataset(batch_dict)
        super().prepare_dataset(batch_dict)
        # y0-0n: AMP
        # self.dataset.values_dict['amp_obs'] = batch_dict['amp_obs']
        # self.dataset.values_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        # self.dataset.values_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']
        return

    def prepare_dataset_MPC(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset_MPC.update_values_dict(dataset_dict)
    
    # def concat_MPC_dataset(self, batch_dict):
    def build_MPC_dataset(self):
        MPC_obs = self._MPC_dataset_buffer._data_buf['obs'][:self._MPC_dataset_buffer._total_count]
        MPC_action = self._MPC_dataset_buffer._data_buf['action'][:self._MPC_dataset_buffer._total_count]
        # actions=torch.cat((batch_dict['actions'], MPC_actions[:-1]))
        # obses=torch.cat((batch_dict['obses'], MPC_obses[:-1]))
        # next_obses=torch.cat((batch_dict['obses'], MPC_obses[1:]))
        actions = MPC_action[:-1].clone()
        prev_obses = MPC_obs[:-1].clone()
        next_obses = MPC_obs[1:].clone()
        MPC_motion_times = torch.zeros((MPC_obs.shape[0],), device=self.ppo_device)
        for L in range(315): # TODO y0-0n: hard coding
            for H in range(50):
                MPC_motion_times[50*L+H] = 0.00833 * (L + H)
        motion_times = MPC_motion_times[1:].clone()
        input_dict = {
            'is_train': True,
            'prev_actions': actions, #torch.cat((batch_dict['actions'], self._MPC_dataset_buffer.sample(self._MPC_dataset_buffer._head)['action'])),
            'obs': prev_obses #torch.cat((batch_dict['obses'], self._MPC_dataset_buffer.sample(self._MPC_dataset_buffer._head)['obs'])),
        }
        res_dict = self.model(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        for key in res_dict:
            if type(res_dict[key]) == type(None):
                continue
            res_dict[key] = res_dict[key].detach()
        del res_dict['entropy']
        res_dict['neglogpacs'] = res_dict['prev_neglogp']
        res_dict['obses'] = prev_obses
        next_obses[:, 3:7] = next_obses[:, [4,5,6,3]] # TODO: need to be changed
        res_dict['next_obses'] = next_obses
        res_dict['actions'] = actions
        res_dict['played_frames'] = actions.shape[0]
        del res_dict['prev_neglogp']
        # res_dict['played_frames']=batch_dict['actions'].shape[0]+self._MPC_dataset_buffer._head
        MPC_deepmimic_rewards, reward_info = self._calc_deepmimic_rewards(next_obses.unsqueeze(0), motion_times.cpu().numpy(), deepmimic=True)

        MPC_dones=torch.zeros((self._MPC_dataset_buffer._head-1), device=self.ppo_device, dtype=torch.float32)
        MPC_dones[49::50]=1 # horizon = 50
        MPC_terminates = MPC_dones.clone()
        MPC_terminates = MPC_terminates.unsqueeze(-1)
        # dones=torch.cat((batch_dict['dones'], MPC_dones))
        res_dict['dones']=MPC_dones

        MPC_next_values = self._eval_critic(next_obses).to(self.device).detach()
        MPC_next_values *= (1.0 - MPC_terminates)
        MPC_values=res_dict['values'].unsqueeze(1)

        # values=res_dict['values'].unsqueeze(1)
        MPC_advs = self.discount_values(MPC_dones.unsqueeze(1), MPC_values, MPC_deepmimic_rewards, MPC_next_values.unsqueeze(1))
        MPC_returns = MPC_advs + MPC_values
        res_dict['returns'] = MPC_returns.reshape(-1, 1)

        return res_dict

        # batch_dict['amp_obs'] = torch.zeros((batch_dict['returns'].shape[0], batch_dict['amp_obs'].shape[-1]))

        # return batch_dict

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                # batch_dict = self.play_steps() 
                batch_dict = self.play_steps_ray() 

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        # y0-0n: AMP
        # self._update_amp_demos()
        # num_obs_samples = batch_dict['amp_obs'].shape[0]
        # amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)['amp_obs']
        # batch_dict['amp_obs_demo'] = amp_obs_demo

        # if (self._amp_replay_buffer.get_total_count() == 0):
        #     batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
        # else:
        #     batch_dict['amp_obs_replay'] = self._amp_replay_buffer.sample(num_obs_samples)['amp_obs']

        # y0-0n: Concat MPC Dataset
        # MPC_batch_dict = self.build_MPC_dataset()
        # for key in batch_dict.keys() - ['amp_obs', 'played_frames']:
        #     batch_dict[key] = torch.cat((batch_dict[key],MPC_batch_dict[key]))

        # batch_dict['played_frames'] = batch_dict['played_frames'] + MPC_batch_dict['played_frames']
        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        # self.prepare_dataset_MPC(MPC_batch_dict) # TODO y0-0n: don't need to prepare at every epoch
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])
                
                if self.schedule_type == 'legacy':  
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            # y0-0n: MPC dataset loop
            # for i in range(len(self.dataset_MPC)):
            #     curr_train_info = self.train_actor_critic(self.dataset_MPC[i])
                
                # if self.schedule_type == 'legacy':  
                #     if self.multi_gpu:
                #         curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                #     self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                #     self.update_lr(self.last_lr)

                # if (train_info is None):
                #     train_info = dict()
                #     for k, v in curr_train_info.items():
                #         train_info[k] = [v]
                # else:
                #     for k, v in curr_train_info.items():
                #         train_info[k].append(v)

            
            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        # self._store_replay_amp_obs(batch_dict['amp_obs'])

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time

        # y0-0n: evaluation
        if self.epoch_num % 1 == 0:
            self.obs, rewards, self.dones, infos = self.env_step(test=True)
            mb_rewards = self.experience_buffer.tensor_dict['rewards']
            # y0-0n: AMP
            # mb_obs = self.experience_buffer.tensor_dict['obses']
            # mb_motion_time = self.experience_buffer.tensor_dict['motion_times']
            # amp_rewards = self._calc_amp_rewards(mb_amp_obs)
            deepmimic_rewards = self._calc_deepmimic_rewards(self.obs['obs'], infos['motion_times'].transpose(0,1)) # TODO: Check next observation is right
            wandb.log(
                {
                    "deepmimic_reward": torch.mean(deepmimic_rewards[0]),
                    "qpos_reward": torch.mean(deepmimic_rewards[1]['qpos']),
                    "qvel_reward": torch.mean(deepmimic_rewards[1]['qvel']),
                    "key_pos_reward": torch.mean(deepmimic_rewards[1]['key_pos']),
                    "root_position": torch.mean(deepmimic_rewards[1]['root_position']),
                    # 'bound_loss': torch_ext.mean_list(train_info['b_loss']).item(),
                    'actor_loss': torch_ext.mean_list(train_info['actor_loss']).item(),
                    'critic_loss': torch_ext.mean_list(train_info['critic_loss']).item(),
                })

        # y0-0n: AMP
        # self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        # y0-0n: AMP
        # amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        # amp_obs = self._preproc_amp_obs(amp_obs)
        # amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        # amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        # amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        # amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        # amp_obs_demo.requires_grad_(True)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            # y0-0n: AMP
            # 'amp_obs' : amp_obs,
            # 'amp_obs_replay' : amp_obs_replay,
            # 'amp_obs_demo' : amp_obs_demo
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            # y0-0n: clamp neglogp
            # res_dict['prev_neglogp'] = torch.min(res_dict['prev_neglogp'], torch.ones_like(res_dict['prev_neglogp'])*10)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            # y0-0n: AMP
            # disc_agent_logit = res_dict['disc_agent_logit']
            # disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            # disc_demo_logit = res_dict['disc_demo_logit']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]
            
            # y0-0n: AMP
            # disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            # disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            # disc_loss = disc_info['disc_loss']

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                #  + self._disc_coef * disc_loss
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        # y0-0n: AMP
        # self.train_result.update(disc_info)

        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert(self._amp_minibatch_size <= self.minibatch_size)

        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        return config

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()
        # y0-0n: MPC dataset
        # self._init_MPC_dataset_buf()
        # self._fetch_MPC_experience_buf()
        return

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty,
            'disc_logit_loss': disc_logit_loss,
            'disc_agent_acc': disc_agent_acc,
            'disc_demo_acc': disc_demo_acc,
            'disc_agent_logit': disc_agent_logit,
            'disc_demo_logit': disc_demo_logit
        }
        return disc_info

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros(batch_shape + self._amp_observation_space.shape,
                                                                    device=self.ppo_device)
        
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        # y0-0n: MPC dataset
        self._MPC_dataset_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['amp_obs']
        return

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return
    
    def _init_MPC_dataset_buf(self):
        import pickle
        with open(file='asset/smpl_rig/motion/MPC_dataset_240315_ctrl_minimize.pkl', mode='rb') as f:
            dataset = pickle.load(f)
        
        N = dataset['root_pos'].shape[0]
        
        local_key_pos_flat = dataset['local_key_pos'].reshape(N, 12)
        obs_np = np.concatenate((dataset['root_pos'], dataset['root_rot'], dataset['root_vel'], dataset['root_ang_vel'], dataset['dof_pos'], dataset['dof_vel'], local_key_pos_flat), axis=1)
        obs_tensor = torch.tensor(obs_np)
        act_tensor = torch.tensor(dataset['action'])
        self._MPC_dataset_buffer.store({'obs': obs_tensor, 'action': act_tensor})

    # def _fetch_MPC_experience_buf(self):
    #     self._MPC_dataset_buffer
    
    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output
    
    def _calc_deepmimic_rewards(self, deepmimic_obs, motion_times, deepmimic=False):
        # from amp.poselib.poselib.core import quat_diff
        motion_lib = self.vec_env.env._motion_lib_demo
        motion_times = motion_times.flatten()
        motion_ids = motion_lib.sample_motions(motion_times.shape[0]) # TODO : fix this hard code line
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = motion_lib.get_motion_state(motion_ids, motion_times)
        
        reward_shape = deepmimic_obs.shape[:-1]

        local_key_pos = key_pos - root_pos.unsqueeze(-2) # local
        local_key_pos = local_key_pos.view(key_pos.shape[0], key_pos.shape[1] * key_pos.shape[2])
        # root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        # deepmimic_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
        #                               self.vec_env.env.cfg['env']['localRootObs'])

        # deepmimic_obs = [root_h (1), root_rot (4), root_vel (3), root_ang_vel (3), dof_pos (37), dof_vel (37), key_body_pos (12)]

        # rotation of a quaternion error
        # root_pos, root_rot_obs, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos
        deepmimic_obs = deepmimic_obs.reshape(-1, self.vec_env.env.num_obs)
        root_pos_sample = deepmimic_obs[:, 0:3]
        root_rot_sample = deepmimic_obs[:, 3:7]
        root_vel_sample = deepmimic_obs[:, 7:10]
        root_ang_vel_sample = deepmimic_obs[:, 10:13]
        dof_pos_sample = deepmimic_obs[:, 13:50]
        dof_vel_sample = deepmimic_obs[:, 50:87]
        key_pos_sample = deepmimic_obs[:, 87:99]

        dof_diff = dof_pos_sample - dof_pos
        # root_rot_diff = torch.stack(get_euler_xyz(root_rot_sample),axis=1) - torch.stack(get_euler_xyz(root_rot),axis=1) #quat_to_tan_norm(root_rot_sample) - quat_to_tan_norm(root_rot)
        root_rot_diff = quat_diff_rad(root_rot_sample, root_rot).reshape(-1, 1)
        # diff = quat_diff(root_rot_sample, root_rot)
        qpos_reward = torch.cat((dof_diff, root_rot_diff), dim=1)
        qpos_reward = torch.sum(torch.abs(qpos_reward),axis=1)
        qpos_reward = torch.exp(-1.5*qpos_reward)

        # angular velocity error
        # dof_vel = torch.cat((root_vel, root_ang_vel, dof_vel), dim=1)
        # dof_vel_sample = torch.cat((root_vel_sample, root_ang_vel_sample, dof_vel_sample), dim=1)
        qvel_reward = dof_vel_sample - dof_vel
        qvel_reward = torch.sum(torch.abs(qvel_reward),axis=1)
        qvel_reward = torch.exp(-1e-1*qvel_reward)

        # key point task position error
        key_pos_diff = key_pos_sample - local_key_pos
        key_pos_reward = torch.sum(torch.abs(key_pos_diff),axis=1)
        key_pos_reward = torch.exp(-40*key_pos_reward)

        # COM error
        root_position_diff = root_pos_sample[...,:3] - root_pos[...,:3]
        root_position_reward = torch.sum(torch.abs(root_position_diff),axis=1)
        root_position_reward = torch.exp(-10*root_position_reward)

        reward = (0.1*root_position_reward + 0.65*qpos_reward + 0.1*qvel_reward + 0.15*key_pos_reward)
        # if not deepmimic:
        reward = reward.view(reward_shape).transpose(0, 1).unsqueeze(2)


        return reward, {"qpos": qpos_reward, "qvel": qvel_reward, "key_pos": key_pos_reward, "root_position": root_position_reward}




    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale
        return disc_r

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        self._amp_replay_buffer.store({'amp_obs': amp_obs})
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        train_info['deepmimic_rewards'] = batch_dict['deepmimic_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        self.writer.add_scalar('losses/disc_loss', torch_ext.mean_list(train_info['disc_loss']).item(), frame)

        self.writer.add_scalar('info/disc_agent_acc', torch_ext.mean_list(train_info['disc_agent_acc']).item(), frame)
        self.writer.add_scalar('info/disc_demo_acc', torch_ext.mean_list(train_info['disc_demo_acc']).item(), frame)
        self.writer.add_scalar('info/disc_agent_logit', torch_ext.mean_list(train_info['disc_agent_logit']).item(), frame)
        self.writer.add_scalar('info/disc_demo_logit', torch_ext.mean_list(train_info['disc_demo_logit']).item(), frame)
        self.writer.add_scalar('info/disc_grad_penalty', torch_ext.mean_list(train_info['disc_grad_penalty']).item(), frame)
        self.writer.add_scalar('info/disc_logit_loss', torch_ext.mean_list(train_info['disc_logit_loss']).item(), frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)
        return

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)
        return