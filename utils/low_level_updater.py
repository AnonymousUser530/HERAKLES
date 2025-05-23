import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

from utils.awr_buffer import AWRBuffer

class AWR_low_level_separated_actor_critic:
    def __init__(self, low_level_actor_model, low_level_critic_model, buffer_path, config_args, device):

        self._ll_actor_model = low_level_actor_model
        self._ll_critic_model = low_level_critic_model
        self.buffer_ll_path = buffer_path
        self.memory = config_args.rl_script_args.memory_ll

        self._minibatch_size_ll = config_args.rl_script_args.minibatch_size_awr_low_level
        self.nbr_usage_transition_awr_critic = config_args.rl_script_args.nbr_usage_transition_awr_critic
        self.nbr_usage_transition_awr_policy = config_args.rl_script_args.nbr_usage_transition_awr_policy

        self._beta_awr = config_args.rl_script_args.beta_awr
        self._entro_coef_awr = config_args.rl_script_args.entro_coef_awr
        self._max_size_awr_buffer = config_args.rl_script_args.max_size_awr_buffer
        self._gamma_awr = config_args.rl_script_args.gamma_awr
        self._lam_awr = config_args.rl_script_args.lam_awr

        self._normalisation_coef_awr = config_args.rl_script_args.normalisation_coef_awr
        self._tsallis_reg = config_args.rl_script_args.tsallis_reg
        self._q = config_args.rl_script_args.tsallis_reg_q
        self._device = device

    def perform_update(self, **kwargs):

        if not (hasattr(self, 'optimizer_ll_policy') and hasattr(self, 'optimizer_ll_critic')):
            self.optimizer_ll_policy = torch.optim.Adam(self._ll_actor_model.parameters(), lr=kwargs["lr_low_level_actor"])
            self.optimizer_ll_critic = torch.optim.Adam(self._ll_critic_model.parameters(), lr=kwargs["lr_low_level_critic"])
            if os.path.exists(kwargs["saving_path_model"] + "/last/actor/optimizer_ll_policy.checkpoint"):
                try:
                    print("Loading optimizer low level")
                    self.optimizer_ll_policy.load_state_dict(
                        torch.load(kwargs["saving_path_model"] + "/last/actor/optimizer_ll_policy.checkpoint"))
                    self.optimizer_ll_critic.load_state_dict(
                        torch.load(kwargs["saving_path_model"] + "/last/critic/optimizer_ll_critic.checkpoint"))
                except:
                    print("Loading optimizer low level from backup")
                    self.optimizer_ll_policy.load_state_dict(
                        torch.load(kwargs["saving_path_model"] + "/backup/actor/optimizer_ll_policy.checkpoint"))
                    self.optimizer_ll_critic.load_state_dict(
                        torch.load(kwargs["saving_path_model"] + "/backup/critic/optimizer_ll_critic.checkpoint"))
                    src = kwargs["saving_path_model"] + "/backup"
                    dest = kwargs["saving_path_model"] + "/last"
                    shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                print("Creating optimizer low level actor")
                self.optimizer_ll_policy = torch.optim.Adam(self._ll_actor_model.parameters(), lr=kwargs["lr_low_level_actor"])
                print("Creating optimizer low level critic")
                self.optimizer_ll_critic = torch.optim.Adam(self._ll_critic_model.parameters(), lr=kwargs["lr_low_level_critic"])
                torch.save(self._ll_actor_model.state_dict(), kwargs["saving_path_model"] + "/last/actor/model.checkpoint")
                torch.save(self._ll_critic_model.state_dict(), kwargs["saving_path_model"] + "/last/critic/model.checkpoint")
                torch.save(self.optimizer_ll_policy.state_dict(),
                           kwargs["saving_path_model"] + "/last/actor/optimizer_ll_policy.checkpoint")
                torch.save(self.optimizer_ll_critic.state_dict(),
                           kwargs["saving_path_model"] + "/last/critic/optimizer_ll_critic.checkpoint")

            return {}

        elif "reinitialize_optimizer" in kwargs and kwargs["reinitialize_optimizer"]:
            print("Reinitializing optimizer low level ")
            self.optimizer_ll_policy = torch.optim.Adam(self._ll_actor_model.parameters(), lr=kwargs["lr_low_level_actor"])
            self.optimizer_ll_critic = torch.optim.Adam(self._ll_critic_model.parameters(), lr=kwargs["lr_low_level_critic"])
            torch.save(self._ll_actor_model.state_dict(), kwargs["saving_path_model"] + "/last/actor/model.checkpoint")
            torch.save(self._ll_critic_model.state_dict(), kwargs["saving_path_model"] + "/last/critic/model.checkpoint")
            torch.save(self.optimizer_ll_policy.state_dict(),
                       kwargs["saving_path_model"] + "/last/actor/optimizer_ll_policy.checkpoint")
            torch.save(self.optimizer_ll_critic.state_dict(),
                       kwargs["saving_path_model"] + "/last/critic/optimizer_ll_critic.checkpoint")

            return {}

        elif "save_models_at_epoch_k" in kwargs:
            print("Saving low level at epoch {}".format(kwargs["save_models_at_epoch_k"]))
            path_model_epoch_k_actor = kwargs["saving_path_model"] + "/save_epoch_{}/actor/model.checkpoint".format(kwargs["save_models_at_epoch_k"])
            path_model_epoch_k_critic = kwargs["saving_path_model"] + "/save_epoch_{}/critic/model.checkpoint".format(kwargs["save_models_at_epoch_k"])
            if os.path.exists(path_model_epoch_k_actor):
                torch.save(self._ll_actor_model.state_dict(), path_model_epoch_k_actor)
                torch.save(self._ll_critic_model.state_dict(), path_model_epoch_k_critic)

            else:
                os.makedirs(os.path.join(kwargs["saving_path_model"], "save_epoch_{}/actor".format(kwargs["save_models_at_epoch_k"])), exist_ok=True)
                os.makedirs(os.path.join(kwargs["saving_path_model"], "save_epoch_{}/critic".format(kwargs["save_models_at_epoch_k"])), exist_ok=True)
                torch.save(self._ll_actor_model.state_dict(), path_model_epoch_k_actor)
                torch.save(self._ll_critic_model.state_dict(), path_model_epoch_k_critic)

            return {}


        if self._normalisation_coef_awr:
            epochs_losses = {
            "value": [],
            "policy": [],
            "entropy": [],
            "max_weight": [],
            "min_weight": []
        }
        else:
            epochs_losses = {
            "value": [],
            "policy": [],
            "entropy": [],
            "exp_w": [],
            "nbr_saturated_weights": []
            }

        # load buffer
        print("Loading buffer low level")
        buffer_awr = AWRBuffer(actor_critic_shared_encoder=False,
                               use_memory=self.memory,
                               size=self._max_size_awr_buffer,
                               gamma=self._gamma_awr,
                               lam=self._lam_awr,
                               device=self._device)

        if os.path.exists(self.buffer_ll_path):
            buffer_awr.load(self.buffer_ll_path)

        # update r_lambda, the discounted reward using TD(lambda) for the low level policy
        buffer_awr.compute_r_lambda(self._ll_critic_model)

        # update the critic
        # we fix the number of time a transition is used during an update and calculate the number of
        # epochs depending on the size of the minibatch and size of the buffer
        # awr_critic_epochs_low_level = nbr_usage_transition_awr_critic * size_buffer/mini_batch_size
        awr_critic_epochs_low_level = int(np.ceil(self.nbr_usage_transition_awr_critic * buffer_awr.len_buffer/self._minibatch_size_ll))
        for b in tqdm(range(awr_critic_epochs_low_level), ascii=" " * 9 + ">", ncols=100):
            if b % (int(np.ceil(buffer_awr.len_buffer / self._minibatch_size_ll))) == 0:
                buffer_awr.shuffle()
            batch_idx_beg = b * self._minibatch_size_ll
            batch_idx_end = batch_idx_beg + self._minibatch_size_ll
            critic_batch_idx = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
            critic_batch_idx = np.mod(critic_batch_idx, buffer_awr.len_buffer)

            self.optimizer_ll_critic.zero_grad()
            critic_loss = buffer_awr.compute_critic_loss(self._ll_critic_model, critic_batch_idx)
            epochs_losses["value"].append(critic_loss.detach().cpu().numpy())
            critic_loss.backward()
            self.optimizer_ll_critic.step()

        # update r_lambda, the discounted reward using TD(lambda) for the low level policy
        buffer_awr.compute_r_lambda(self._ll_critic_model)

        # update the policy
        # we fix the number of time a transition is used during an update and calculate the number of
        # epochs depending on the size of the minibatch and size of the buffer
        # policy_epochs_low_level = nbr_usage_transition_awr_policy * size_buffer/mini_batch_size
        awr_policy_epochs_low_level = int(np.ceil(self.nbr_usage_transition_awr_policy * buffer_awr.len_buffer/self._minibatch_size_ll))
        for b in tqdm(range(awr_policy_epochs_low_level), ascii=" " * 9 + ">", ncols=100):
            if b % (int(np.ceil(buffer_awr.len_buffer / self._minibatch_size_ll))) == 0:
                buffer_awr.shuffle()
            batch_idx_beg = b * self._minibatch_size_ll
            batch_idx_end = batch_idx_beg + self._minibatch_size_ll
            policy_batch_idx = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
            policy_batch_idx = np.mod(policy_batch_idx, buffer_awr.len_buffer)

            self.optimizer_ll_policy.zero_grad()

            if self._normalisation_coef_awr:
                if self._tsallis_reg:
                    policy_loss, entropy, max_weight, min_weight = buffer_awr.compute_policy_loss(self._ll_actor_model,
                                                                                                        policy_batch_idx,
                                                                                                        self._beta_awr,
                                                                                                        self._normalisation_coef_awr,
                                                                                                        self._tsallis_reg,
                                                                                                        self._q)
                else:
                    policy_loss, entropy, max_weight, min_weight = buffer_awr.compute_policy_loss(self._ll_actor_model,
                                                                                                        policy_batch_idx,
                                                                                                        self._beta_awr,
                                                                                                        self._normalisation_coef_awr)
                epochs_losses["max_weight"].append(max_weight.detach().cpu().numpy())
                epochs_losses["min_weight"].append(min_weight.detach().cpu().numpy())
            else:
                policy_loss, entropy, exp_w, nbr_saturated_weights = buffer_awr.compute_policy_loss(self._ll_actor_model,
                                                                                                    policy_batch_idx,
                                                                                                    self._beta_awr,
                                                                                                    self._normalisation_coef_awr)
                epochs_losses["exp_w"].append(exp_w.detach().cpu().numpy())
                epochs_losses["nbr_saturated_weights"].append(nbr_saturated_weights)

            epochs_losses["policy"].append(policy_loss.detach().cpu().numpy())
            epochs_losses["entropy"].append(entropy.detach().cpu().numpy())


            total_loss = policy_loss - self._entro_coef_awr * entropy
            total_loss.backward()
            self.optimizer_ll_policy.step()

        del buffer_awr

        if kwargs["save_after_update"]:
            print("Saving  low level after training")
            src = kwargs["saving_path_model"] + "/last"
            dest = kwargs["saving_path_model"] + "/backup"
            shutil.copytree(src, dest, dirs_exist_ok=True)
            torch.save(self._ll_critic_model.state_dict(), kwargs["saving_path_model"] + "/last/critic/model.checkpoint")
            torch.save(self._ll_actor_model.state_dict(), kwargs["saving_path_model"] + "/last/actor/model.checkpoint")
            torch.save(self.optimizer_ll_policy.state_dict(),
                       kwargs["saving_path_model"] + "/last/actor/optimizer_ll_policy.checkpoint")
            torch.save(self.optimizer_ll_critic.state_dict(),
                       kwargs["saving_path_model"] + "/last/critic/optimizer_ll_critic.checkpoint")

        if self._normalisation_coef_awr:
            return {'value_loss': np.mean(epochs_losses["value"]), 'policy_loss': np.mean(epochs_losses["policy"]),
                'entropy': np.mean(epochs_losses["entropy"]), 'max_weight': np.mean(epochs_losses["max_weight"]),
                'min_weight': np.mean(epochs_losses["min_weight"])}
        else:
            return {'value_loss': np.mean(epochs_losses["value"]), 'policy_loss': np.mean(epochs_losses["policy"]),
                'entropy': np.mean(epochs_losses["entropy"]), 'exp_w': np.mean(epochs_losses["exp_w"]),
                'nbr_saturated_weights': np.mean(epochs_losses["nbr_saturated_weights"])}

class AWR_low_level_shared_encoder:
    def __init__(self, low_level_model, buffer_path, config_args, device):

        self._ll_model = low_level_model
        self.buffer_ll_path = buffer_path
        self.memory = config_args.rl_script_args.memory_ll

        self._minibatch_size_ll = config_args.rl_script_args.minibatch_size_awr_low_level
        self.nbr_usage_transition_awr_critic = config_args.rl_script_args.nbr_usage_transition_awr_critic
        self.nbr_usage_transition_awr_policy = config_args.rl_script_args.nbr_usage_transition_awr_policy

        self._beta_awr = config_args.rl_script_args.beta_awr
        self._entro_coef_awr = config_args.rl_script_args.entro_coef_awr
        self._max_size_awr_buffer = config_args.rl_script_args.max_size_awr_buffer
        self._gamma_awr = config_args.rl_script_args.gamma_awr
        self._lam_awr = config_args.rl_script_args.lam_awr

        self._normalisation_coef_awr = config_args.rl_script_args.normalisation_coef_awr
        self._tsallis_reg = config_args.rl_script_args.tsallis_reg
        self._q = config_args.rl_script_args.tsallis_reg_q
        self._device = device

    def perform_update(self, **kwargs):

        if not (hasattr(self, 'optimizer_ll_policy')):
            self.optimizer_ll_policy = torch.optim.Adam(self._ll_model.parameters(), lr=kwargs["lr_low_level_actor"])
            self.optimizer_ll_critic = torch.optim.Adam(self._ll_model.parameters(), lr=kwargs["lr_low_level_critic"])
            if os.path.exists(kwargs["saving_path_model"] + "/last/optimizer_ll_policy.checkpoint"):
                try:
                    print("Loading optimizer low level")
                    self.optimizer_ll_policy.load_state_dict(
                        torch.load(kwargs["saving_path_model"] + "/last/optimizer_ll_policy.checkpoint"))
                    self.optimizer_ll_critic.load_state_dict(
                        torch.load(kwargs["saving_path_model"] + "/last/optimizer_ll_critic.checkpoint"))
                except:
                    print("Loading optimizer low level from backup")
                    self.optimizer_ll_policy.load_state_dict(
                        torch.load(kwargs["saving_path_model"] + "/backup/optimizer_ll_policy.checkpoint"))
                    self.optimizer_ll_critic.load_state_dict(
                        torch.load(kwargs["saving_path_model"] + "/backup/optimizer_ll_critic.checkpoint"))
                    src = kwargs["saving_path_model"] + "/backup"
                    dest = kwargs["saving_path_model"] + "/last"
                    shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                print("Creating optimizer low level actor")
                self.optimizer_ll_policy = torch.optim.Adam(self._ll_model.parameters(), lr=kwargs["lr_low_level_actor"])
                print("Creating optimizer low level critic")
                self.optimizer_ll_critic = torch.optim.Adam(self._ll_model.parameters(), lr=kwargs["lr_low_level_critic"])
                torch.save(self._ll_model.state_dict(), kwargs["saving_path_model"] + "/last/model.checkpoint")
                torch.save(self.optimizer_ll_policy.state_dict(),
                           kwargs["saving_path_model"] + "/last/optimizer_ll_policy.checkpoint")
                torch.save(self.optimizer_ll_critic.state_dict(),
                           kwargs["saving_path_model"] + "/last/optimizer_ll_critic.checkpoint")

            return {}

        elif "reinitialize_optimizer" in kwargs and kwargs["reinitialize_optimizer"]:
            print("Reinitializing optimizer low level ")
            self.optimizer_ll_policy = torch.optim.Adam(self._ll_model.parameters(), lr=kwargs["lr_low_level_actor"])
            self.optimizer_ll_critic = torch.optim.Adam(self._ll_model.parameters(), lr=kwargs["lr_low_level_critic"])
            torch.save(self._ll_model.state_dict(), kwargs["saving_path_model"] + "/last/model.checkpoint")
            torch.save(self.optimizer_ll_policy.state_dict(),
                       kwargs["saving_path_model"] + "/last/optimizer_ll_policy.checkpoint")
            torch.save(self.optimizer_ll_critic.state_dict(),
                       kwargs["saving_path_model"] + "/last/optimizer_ll_critic.checkpoint")

            return {}

        elif "save_models_at_epoch_k" in kwargs:
            print("Saving low level at epoch {}".format(kwargs["save_models_at_epoch_k"]))
            path_model_epoch_k = kwargs["saving_path_model"] + "/save_epoch_{}/model.checkpoint".format(kwargs["save_models_at_epoch_k"])
            if os.path.exists(path_model_epoch_k):
                torch.save(self._ll_model.state_dict(), path_model_epoch_k)
            else:
                os.makedirs(os.path.join(kwargs["saving_path_model"], "save_epoch_{}".format(kwargs["save_models_at_epoch_k"])), exist_ok=True)
                torch.save(self._ll_model.state_dict(), path_model_epoch_k)

            return {}

        elif "load_models_at_epoch_k" in kwargs:
            print("Loading low level at epoch {}".format(kwargs["load_models_at_epoch_k"]))
            path_model_epoch_k = kwargs["saving_path_model"] + "/save_epoch_{}/model.checkpoint".format(kwargs["load_models_at_epoch_k"])
            if os.path.exists(path_model_epoch_k):
                self._ll_model.load_state_dict(torch.load(path_model_epoch_k))
            else:
                print("Model not found")
            return {}

        if self._normalisation_coef_awr:
            epochs_losses = {
            "value": [],
            "policy": [],
            "entropy": [],
            "max_weight": [],
            "min_weight": []
        }
        else:
            epochs_losses = {
            "value": [],
            "policy": [],
            "entropy": [],
            "exp_w": [],
            "nbr_saturated_weights": []
            }

        # load buffer
        print("Loading buffer low level")
        buffer_awr = AWRBuffer(actor_critic_shared_encoder=True,
                               use_memory=self.memory,
                               size=self._max_size_awr_buffer,
                               gamma=self._gamma_awr,
                               lam=self._lam_awr,
                               device=self._device)

        if os.path.exists(self.buffer_ll_path):
            buffer_awr.load(self.buffer_ll_path)

        # update r_lambda, the discounted reward using TD(lambda) for the low level policy
        buffer_awr.compute_r_lambda(self._ll_model)

        # update the critic
        # we fix the number of time a transition is used during an update and calculate the number of
        # epochs depending on the size of the minibatch and size of the buffer
        # awr_critic_epochs_low_level = nbr_usage_transition_awr_critic * size_buffer/mini_batch_size
        awr_critic_epochs_low_level = int(np.ceil(self.nbr_usage_transition_awr_critic * buffer_awr.len_buffer/self._minibatch_size_ll))
        for b in tqdm(range(awr_critic_epochs_low_level), ascii=" " * 9 + ">", ncols=100):
            if b % (int(np.ceil(buffer_awr.len_buffer / self._minibatch_size_ll))) == 0:
                buffer_awr.shuffle()
            batch_idx_beg = b * self._minibatch_size_ll
            batch_idx_end = batch_idx_beg + self._minibatch_size_ll
            critic_batch_idx = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
            critic_batch_idx = np.mod(critic_batch_idx, buffer_awr.len_buffer)

            self.optimizer_ll_critic.zero_grad()
            critic_loss = buffer_awr.compute_critic_loss(self._ll_model, critic_batch_idx)
            epochs_losses["value"].append(critic_loss.detach().cpu().numpy())
            critic_loss.backward()
            self.optimizer_ll_critic.step()

        # update r_lambda, the discounted reward using TD(lambda) for the low level policy
        buffer_awr.compute_r_lambda(self._ll_model)

        # update the policy
        # we fix the number of time a transition is used during an update and calculate the number of
        # epochs depending on the size of the minibatch and size of the buffer
        # policy_epochs_low_level = nbr_usage_transition_awr_policy * size_buffer/mini_batch_size
        awr_policy_epochs_low_level = int(np.ceil(self.nbr_usage_transition_awr_policy * buffer_awr.len_buffer/self._minibatch_size_ll))
        for b in tqdm(range(awr_policy_epochs_low_level), ascii=" " * 9 + ">", ncols=100):
            if b % (int(np.ceil(buffer_awr.len_buffer / self._minibatch_size_ll))) == 0:
                buffer_awr.shuffle()
            batch_idx_beg = b * self._minibatch_size_ll
            batch_idx_end = batch_idx_beg + self._minibatch_size_ll
            policy_batch_idx = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
            policy_batch_idx = np.mod(policy_batch_idx, buffer_awr.len_buffer)

            self.optimizer_ll_policy.zero_grad()

            if self._normalisation_coef_awr:
                if self._tsallis_reg:
                    policy_loss, entropy, max_weight, min_weight = buffer_awr.compute_policy_loss(self._ll_model,
                                                                                                        policy_batch_idx,
                                                                                                        self._beta_awr,
                                                                                                        self._normalisation_coef_awr,
                                                                                                        self._tsallis_reg,
                                                                                                        self._q)
                else:
                    policy_loss, entropy, max_weight, min_weight = buffer_awr.compute_policy_loss(self._ll_model,
                                                                                                        policy_batch_idx,
                                                                                                        self._beta_awr,
                                                                                                        self._normalisation_coef_awr)
                epochs_losses["max_weight"].append(max_weight.detach().cpu().numpy())
                epochs_losses["min_weight"].append(min_weight.detach().cpu().numpy())
            else:
                policy_loss, entropy, exp_w, nbr_saturated_weights = buffer_awr.compute_policy_loss(self._ll_model,
                                                                                                    policy_batch_idx,
                                                                                                    self._beta_awr,
                                                                                                    self._normalisation_coef_awr)
                epochs_losses["exp_w"].append(exp_w.detach().cpu().numpy())
                epochs_losses["nbr_saturated_weights"].append(nbr_saturated_weights)

            epochs_losses["policy"].append(policy_loss.detach().cpu().numpy())
            epochs_losses["entropy"].append(entropy.detach().cpu().numpy())


            total_loss = policy_loss - self._entro_coef_awr * entropy
            total_loss.backward()
            self.optimizer_ll_policy.step()

        del buffer_awr

        if kwargs["save_after_update"]:
            print("Saving  low level after training")
            src = kwargs["saving_path_model"] + "/last"
            dest = kwargs["saving_path_model"] + "/backup"
            shutil.copytree(src, dest, dirs_exist_ok=True)
            torch.save(self._ll_model.state_dict(), kwargs["saving_path_model"] + "/last/model.checkpoint")
            torch.save(self.optimizer_ll_policy.state_dict(),
                       kwargs["saving_path_model"] + "/last/optimizer_ll_policy.checkpoint")
            torch.save(self.optimizer_ll_critic.state_dict(),
                       kwargs["saving_path_model"] + "/last/optimizer_ll_critic.checkpoint")

        if self._normalisation_coef_awr:
            return {'value_loss': np.mean(epochs_losses["value"]), 'policy_loss': np.mean(epochs_losses["policy"]),
                'entropy': np.mean(epochs_losses["entropy"]), 'max_weight': np.mean(epochs_losses["max_weight"]),
                'min_weight': np.mean(epochs_losses["min_weight"])}
        else:
            return {'value_loss': np.mean(epochs_losses["value"]), 'policy_loss': np.mean(epochs_losses["policy"]),
                'entropy': np.mean(epochs_losses["entropy"]), 'exp_w': np.mean(epochs_losses["exp_w"]),
                'nbr_saturated_weights': np.mean(epochs_losses["nbr_saturated_weights"])}