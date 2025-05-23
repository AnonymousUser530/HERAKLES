import os
import shutil
import logging

import math
import numpy as np
import torch
import pickle

from torch.nn import functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from collections import OrderedDict, deque

from lamorel import BaseUpdater

class HLUpdater(BaseUpdater):
    def __init__(self, model_type, minibatch_size, gradient_batch_size, use_unsloth=False, gradient_minibatch_size=None):
        super(HLUpdater, self).__init__()
        self._model_type = model_type
        self._minibatch_size = minibatch_size
        self._gradient_batch_size = gradient_batch_size
        self._gradient_minibatch_size = gradient_minibatch_size
        self.use_unsloth = use_unsloth
        # print("updater use_unsloth: {}".format(self.use_unsloth))

    def _get_trainable_params(self, model, return_with_names=False):
        if return_with_names:
            return filter(lambda p: p[1].requires_grad, model.named_parameters())
        else:
            return filter(lambda p: p.requires_grad, model.parameters())

    def _get_filtered_params(self, model, name_filter, return_with_names=False):
        if return_with_names:
            return filter(lambda p: name_filter(p[0]), model.named_parameters())
        else:
            return filter(lambda p: name_filter(p[0]), model.parameters())

    def get_parameters_name_filter(self):
        return lambda n: '.default.' in n or '.value.' in n

    def gradient_pass_actor_critic(self, contexts, candidates, current_process_buffer,
                      n_minibatches, epochs_losses, kwargs):

        base_log_probs = []
        for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):

            if isinstance(base_log_probs, list) and len(base_log_probs) > 0:
                # print("epoch: {} base_log_probs: {}".format(i, base_log_probs))
                base_log_probs = torch.cat(base_log_probs, dim=0)

            ratio_accumulation = []
            policy_loss_accumulation = []
            value_loss_accumulation = []
            entropy_accumulation = []
            loss_accumulation = []
            kl_penalty_accumulation = []

            for step in range(n_minibatches):
                _minibatch_start_idx = step * self._minibatch_size
                _minibatch_end_idx = min(
                    (step + 1) * self._minibatch_size,
                    len(contexts))

                self.ac_optimizer.zero_grad()
                gradient_accumulation_steps = math.ceil(
                    (_minibatch_end_idx - _minibatch_start_idx) / self._gradient_batch_size)
                for accumulated_batch in range(gradient_accumulation_steps):
                    _start_idx = _minibatch_start_idx + accumulated_batch * self._gradient_batch_size
                    _stop_idx = _minibatch_start_idx + min(
                        (accumulated_batch + 1) * self._gradient_batch_size, _minibatch_end_idx)
                    # print("accumulated_batch: {}, _start_idx: {}, _stop_idx: {}, process_index: {}".format(accumulated_batch, _start_idx, _stop_idx, int(os.environ.get("RANK"))))
                    _contexts = contexts[_start_idx:_stop_idx]
                    _candidates = candidates[_start_idx:_stop_idx]
                    if len(_contexts) == 0: break
                    if self._gradient_minibatch_size is None:
                        _batch_size = min(10, sum(len(_c) for _c in _candidates))
                    else:
                        _batch_size = self._gradient_minibatch_size
                    # Use LLM to compute again action probabilities and value
                    dict_tokens = {"{}".format(
                        kwargs["tokenizer"]("{}".format(idx_c), add_special_tokens=False, return_token_type_ids=False)[
                            "input_ids"]):
                                       _cands for idx_c, _cands in enumerate(_candidates)}
                    candidates_keys = [["{}".format(i)] for i in range(len(_candidates))]
                    # dict_tokens = {'[15]': [2651], '[16]': [50256]}
                    output = self._llm_module(['score', 'value'], contexts=_contexts, candidates=candidates_keys,
                                              require_grad=True, minibatch_size=_batch_size,
                                              add_eos_on_candidates=False,
                                              peft_adapter="default",
                                              dict_tokens=dict_tokens,
                                              pad_contexts=False)

                    scores = torch.stack([_o['score']["tokens_logprobs"] for _o in output])
                    scores_over_voc = torch.stack([_o['score']["log_prob_full_distrib"] for _o in output]).squeeze()
                    if scores.dim() < 3:
                        scores = scores.unsqueeze(1)
                    distrib = Categorical(logits=scores)
                    distrib_over_voc = Categorical(logits=scores_over_voc)
                    values = torch.stack([_o["value"]["value"][0][0] for _o in output]).squeeze()
                    # Compute policy loss
                    entropy = distrib.entropy().mean()
                    log_prob = distrib.log_prob(current_process_buffer['actions'][
                                               _start_idx:_stop_idx].unsqueeze(
                        -1)).squeeze()  # Use logprobs from dist as they were normalized
                    ratio = torch.exp(log_prob - current_process_buffer['logprobs'][_start_idx:_stop_idx])
                    # assert not (i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)))
                    if i == 0 and step == 0 and (torch.any(ratio < 0.9) or torch.any(ratio > 1.1)):
                        logging.warning("PPO ratio != 1 !!")
                        if (torch.any(ratio < 0.8) or torch.any(ratio > 1.2)):
                            print(
                                "===== issue in ratio for process_index {}====  \nratio: {} \n_contexts: {} \ndict_tokens: {} \ncurrent_process_buffer['actions'][_start_idx:_stop_idx]: {} \nlog_prob: {} \ncurrent_process_buffer['logprobs'][_start_idx:_stop_idx]: {} \n========".format(
                                    int(os.environ.get("RANK")),
                                    ratio, _contexts, dict_tokens,
                                    current_process_buffer['actions'][_start_idx:_stop_idx],
                                    log_prob, current_process_buffer['logprobs'][_start_idx:_stop_idx]))

                    clip_adv = torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]) * \
                               current_process_buffer['advantages'][_start_idx:_stop_idx]
                    policy_loss = -(
                        torch.min(ratio * current_process_buffer['advantages'][_start_idx:_stop_idx], clip_adv)).mean()

                    # Compute value loss
                    unclipped_value_error = ((values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                    clipped_values = current_process_buffer['values'][_start_idx:_stop_idx] + \
                                     torch.clamp(values - current_process_buffer['values'][_start_idx:_stop_idx],
                                                 -kwargs["clip_eps"], kwargs["clip_eps"])
                    clipped_value_error = (
                            (clipped_values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                    value_loss = torch.max(unclipped_value_error, clipped_value_error).mean()

                    # compute kl_divergence with original model if necessary
                    if "compute_kl_with_original_model" in kwargs and kwargs["compute_kl_with_original_model"]:
                        if i==0:
                            with self._llm_module.module._modules['_LLM_model'].disable_adapter():
                                with self._llm_module.no_sync():
                                    base_output = self._llm_module(['score'], contexts=_contexts, candidates=candidates_keys,
                                                  require_grad=False, minibatch_size=_batch_size,
                                                  add_eos_on_candidates=False,
                                                  peft_adapter="default",
                                                  dict_tokens=dict_tokens,
                                                  pad_contexts=False)
                                    _base_log_probs = Categorical(logits=torch.stack([_o['score']['log_prob_full_distrib'] for _o in base_output]).squeeze()).logits

                                    if _base_log_probs.dim() < 2:
                                        _base_log_probs = _base_log_probs.unsqueeze(0)
                                    try:
                                        base_log_probs.append(_base_log_probs)
                                    except:
                                        print("epoch: {} step:{} accumulated_batch: {} base_log_probs: {}".format(i, step, accumulated_batch, base_log_probs))
                                        base_log_probs.append(_base_log_probs)
                        else:
                            _base_log_probs = base_log_probs[_start_idx:_stop_idx]

                        if _base_log_probs.dim() < 2:
                            kl_penalty = (distrib_over_voc.probs * (distrib_over_voc.logits - _base_log_probs)).unsqueeze(0).sum(dim=1)  # true KL divergencee
                        else:
                            try :
                                kl_penalty = (distrib_over_voc.probs * (distrib_over_voc.logits - _base_log_probs)).sum(dim=1)  # true KL divergencee
                            except:
                                print("Epoch {} _base_log_probs: {} dim {}, distrib_over_voc.logits: {}".format(i, _base_log_probs, _base_log_probs.dim(), distrib_over_voc.logits))
                                kl_penalty = (distrib_over_voc.probs * (distrib_over_voc.logits - _base_log_probs)).sum(dim=1)

                        if not torch.all(kl_penalty >= -0.1):
                            # kl_penalty should be >= 0 but calculation error could error on gpu thus we allow a small margin
                            print(f"Epoch {i} - minibatch {step} - accumulated_batch {accumulated_batch}: \n kl_penalty {kl_penalty} - \nscores_over_voc {scores_over_voc} - \n_base_log_probs {_base_log_probs}")

                        kl_penalty_accumulation.append(kl_penalty.mean().float().detach().cpu().numpy())

                        # Compute final loss
                        loss = (policy_loss - kwargs["entropy_coef"] * entropy + kwargs["value_loss_coef"] * value_loss +
                                kwargs["kl_coef_with_original_model"] * kl_penalty.mean()) # kl penalty

                    else:
                        # Compute final loss
                        loss = policy_loss - kwargs["entropy_coef"] * entropy + kwargs["value_loss_coef"] * value_loss

                    loss_accumulation.append(loss.detach().cpu().numpy())
                    policy_loss_accumulation.append(policy_loss.detach().cpu().numpy())
                    ratio_accumulation.extend(ratio.detach().cpu().numpy())
                    value_loss_accumulation.append(value_loss.detach().cpu().numpy())
                    entropy_accumulation.append(entropy.detach().cpu().numpy())
                    loss = loss / gradient_accumulation_steps
                    # Backward
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self._iterator_ac_params, kwargs["max_grad_norm"])
                self.ac_optimizer.step()
                # print("process: {}, step: {}, minibatch: {}".format(int(os.environ.get("RANK")), step, i))

            """if os.environ["RANK"] == "1":
                print("process: {}, loss_accumulation: {}".format(int(os.environ.get("RANK")), loss_accumulation))
                params_updated = np.sum([torch.sum(p.detach().cpu()) for n, p in self._iterator_named_filtered_params(self.get_parameters_name_filter()) if "lora_b" in n.lower()])
                print(f"LoRA_B params: {params_updated}")"""
            epochs_losses["ratio"].append(np.mean(ratio_accumulation))
            epochs_losses["entropy"].append(np.mean(entropy_accumulation))
            epochs_losses["loss"].append(np.mean(loss_accumulation))
            epochs_losses["policy"].append(np.mean(policy_loss_accumulation))
            epochs_losses["value"].append(np.mean(value_loss_accumulation))
            epochs_losses["kl_penalty"].append(np.mean(kl_penalty_accumulation))

    def gradient_pass_sr_estimator(self, contexts, candidates, epochs_loss, _current_batch_ids, kwargs):
        # here the context is the goal_buffer and the candidates are the success_buffer
        loss_accumulation = []
        # linearly weighting the contexts
        p = kwargs["weights"][_current_batch_ids["contexts"]] / kwargs["weights"][_current_batch_ids["contexts"]].sum()
        for _ in tqdm(range(kwargs["sr_estimator_epochs"]), ascii=" " * 9 + ">", ncols=100):
            idx = np.random.choice(len(contexts), kwargs["batch_size"], p=p)
            goals = [contexts[i] for i in idx]
            success = torch.tensor([candidates[i] for i in idx], dtype=torch.float32).squeeze()
            gradient_accumulation_steps = math.ceil(kwargs["batch_size"] / self._gradient_batch_size)

            if kwargs["network_type"] == "LL":
                self.sr_ll_estimator_optimizer.zero_grad()
            elif kwargs["network_type"] == "HL":
                self.sr_hl_estimator_optimizer.zero_grad()
            else:
                raise ValueError("Unknown network type: {}".format(kwargs["network_type"]))
            for accumulated_batch in tqdm(range(gradient_accumulation_steps)):
                _start_idx = accumulated_batch * self._gradient_batch_size
                _stop_idx = (accumulated_batch + 1) * self._gradient_batch_size

                _goals = goals[_start_idx:_stop_idx]
                _success = success[_start_idx:_stop_idx]

                _batch_size = len(_goals)

                if _batch_size <= 1:
                    continue

                # Use LLM to compute again action probabilities and value
                if kwargs["network_type"] == "LL":
                    output = self._llm_module(['sr_ll_estimator'], contexts=_goals,
                                                require_grad=True, minibatch_size=_batch_size,
                                                peft_adapter='sr_LL_adapters',
                                           add_eos_on_candidates=False,
                                              pad_contexts=False)
                    logits_sr = torch.stack([_o['sr_ll_estimator']["sr_ll_estimated"] for _o in output]).squeeze()

                elif kwargs["network_type"] == "HL":
                    output = self._llm_module(['sr_hl_estimator'], contexts=_goals,
                                                require_grad=True, minibatch_size=_batch_size,
                                                peft_adapter='sr_HL_adapters',
                                           add_eos_on_candidates=False,
                                              pad_contexts=False)
                    logits_sr = torch.stack([_o['sr_hl_estimator']["sr_hl_estimated"] for _o in output]).squeeze()

                # Compute sr loss
                sr_loss = F.binary_cross_entropy_with_logits(logits_sr, _success)

                # Compute final loss
                loss = sr_loss / gradient_accumulation_steps
                loss_accumulation.append(loss.detach().cpu().numpy())

                # Backward
                loss.backward()

            if kwargs["network_type"] == "LL":
                self.sr_ll_estimator_optimizer.step()
                """if os.environ["RANK"] == "1":
                    params_updated = np.sum([torch.sum(p.detach().cpu()) for n, p in self._iterator_named_filtered_params(self._sr_ll_estimator_parameters_filter) if "lora_b" in n.lower()])
                    print(f"LoRA_B params: {params_updated}")"""
            elif kwargs["network_type"] == "HL":
                self.sr_hl_estimator_optimizer.step()

                """if os.environ["RANK"] == "1":
                    params_updated = np.sum([torch.sum(p.detach().cpu()) for n, p in self._iterator_named_filtered_params(self._sr_hl_estimator_parameters_filter) if "lora_b" in n.lower()])
                    print(f"LoRA_B params: {params_updated}")"""
        epochs_loss["loss"].append(np.mean(loss_accumulation))

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'ac_optimizer') or not hasattr(self, 'sr_ll_estimator_optimizer') or not hasattr(self, 'sr_hl_estimator_optimizer'):
            if "load_fine_tuned_version" not in kwargs:
                self._iterator_named_trainable_params = lambda: self._get_trainable_params(self._llm_module, True)
                self._iterator_named_filtered_params = lambda name_filter: self._get_filtered_params(self._llm_module, name_filter, True)

                self._sr_ll_estimator_parameters_filter = self._llm_module.module._module_functions['sr_ll_estimator'].get_parameters_name_filter()
                self._iterator_ac_params = (p for n, p in self._iterator_named_filtered_params(self.get_parameters_name_filter()))
                self._iterator_sr_ll_estimator_params = (p for n, p in self._iterator_named_filtered_params(self._sr_ll_estimator_parameters_filter))

                if not hasattr(self, 'ac_optimizer'):
                    self.ac_optimizer = torch.optim.Adam(self._iterator_ac_params, lr=kwargs["lr"])

                if not hasattr(self, 'sr_ll_estimator_optimizer'):
                    self.sr_ll_estimator_optimizer = torch.optim.Adam(self._iterator_sr_ll_estimator_params, lr=kwargs["lr_ll_sr_estimator"])
                if not hasattr(self, 'sr_hl_estimator_optimizer') and "sr_hl_estimator" in kwargs and kwargs["sr_hl_estimator"]:
                    self._sr_hl_estimator_parameters_filter = self._llm_module.module._module_functions['sr_hl_estimator'].get_parameters_name_filter()
                    self._sr_hl_estimator_delayed_parameters_filter = self._llm_module.module._module_functions['sr_hl_estimator_delayed'].get_parameters_name_filter()
                    self._iterator_sr_hl_estimator_params = (p for n, p in self._iterator_named_filtered_params(self._sr_hl_estimator_parameters_filter))
                    self.sr_hl_estimator_optimizer = torch.optim.Adam(self._iterator_sr_hl_estimator_params, lr=kwargs["lr_hl_sr_estimator"])
                    if not hasattr(self, 'weights_buffer'):
                        if os.path.exists(kwargs["saving_path_model"] + "/buffer_sr_hl_estimator_weights.pkl"):
                            with open(kwargs["saving_path_model"] + "/buffer_sr_hl_estimator_weights.pkl", 'rb') as f:
                                self.weights_buffer = pickle.load(f)
                                for (n, p), w in zip(self._iterator_named_filtered_params(self._sr_hl_estimator_delayed_parameters_filter), self.weights_buffer[0]):
                                    p.data.copy_(w)
                        else:
                            self.weights_buffer = deque(maxlen=kwargs["buffer_size"])

        if "load_fine_tuned_version" in kwargs or "save_first_last" in kwargs or "reinitialize" in kwargs or "reinitialize_optimizer" in kwargs:
            self._iterator_named_trainable_params = lambda: self._get_trainable_params(self._llm_module, True)
            self._iterator_named_filtered_params = lambda name_filter: self._get_filtered_params(self._llm_module, name_filter, True)

            self._sr_ll_estimator_parameters_filter = self._llm_module.module._module_functions['sr_ll_estimator'].get_parameters_name_filter()

            self._iterator_ac_params = (p for n, p in self._iterator_named_filtered_params(self.get_parameters_name_filter()))
            self._iterator_sr_ll_estimator_params = (p for n, p in self._iterator_named_filtered_params(self._sr_ll_estimator_parameters_filter))
            if "sr_hl_estimator" in kwargs and kwargs["sr_hl_estimator"]:
                self._sr_hl_estimator_parameters_filter = self._llm_module.module._module_functions['sr_hl_estimator'].get_parameters_name_filter()
                self._sr_hl_estimator_delayed_parameters_filter = self._llm_module.module._module_functions['sr_hl_estimator_delayed'].get_parameters_name_filter()
                self._iterator_sr_hl_estimator_params = (p for n, p in self._iterator_named_filtered_params(self._sr_hl_estimator_parameters_filter))
                if not hasattr(self, 'weights_buffer'):
                        if os.path.exists(kwargs["saving_path_model"] + "/buffer_sr_hl_estimator_weights.pkl"):
                            with open(kwargs["saving_path_model"] + "/buffer_sr_hl_estimator_weights.pkl", 'rb') as f:
                                self.weights_buffer = pickle.load(f)
                                for (n, p), w in zip(self._iterator_named_filtered_params(self._sr_hl_estimator_delayed_parameters_filter), self.weights_buffer[0]):
                                    p.data.copy_(w)
                        else:
                            self.weights_buffer = deque(maxlen=kwargs["buffer_size"])

            if "load_fine_tuned_version" in kwargs and kwargs["load_fine_tuned_version"] \
                    and not hasattr(self, "is_loaded"):
                try:
                    print("Loading model high level")
                    self._llm_module.load_state_dict(torch.load(kwargs["saving_path_model"] + "/last/model.checkpoint"),
                                                     strict=False)
                    self.ac_optimizer = torch.optim.Adam(self._iterator_ac_params)
                    self.ac_optimizer.load_state_dict(torch.load(
                        kwargs["saving_path_model"] + "/last/ac_optimizer.checkpoint"))
                    self.sr_ll_estimator_optimizer = torch.optim.Adam(self._iterator_sr_ll_estimator_params)
                    self.sr_ll_estimator_optimizer.load_state_dict(torch.load(
                        kwargs["saving_path_model"] + "/last/sr_ll_estimator_optimizer.checkpoint"))
                    if "sr_hl_estimator" in kwargs and kwargs["sr_hl_estimator"]:
                        self.sr_hl_estimator_optimizer = torch.optim.Adam(self._iterator_sr_hl_estimator_params)
                        self.sr_hl_estimator_optimizer.load_state_dict(torch.load(
                            kwargs["saving_path_model"] + "/last/sr_hl_estimator_optimizer.checkpoint"))
                    self.is_loaded = True
                except:
                    print("Loading model high level from backup")
                    # The last save has been corrupted for whatever reason, possibly the program has been forced
                    # to close during the saving => we use the backup
                    self._llm_module.load_state_dict(
                        torch.load(kwargs["saving_path_model"] + "/backup/model.checkpoint"),
                        strict=False)
                    self.ac_optimizer = torch.optim.Adam(self._iterator_ac_params)
                    self.ac_optimizer.load_state_dict(torch.load(
                        kwargs["saving_path_model"] + "/backup/ac_optimizer.checkpoint"))
                    self.sr_ll_estimator_optimizer = torch.optim.Adam(self._iterator_sr_ll_estimator_params)
                    self.sr_ll_estimator_optimizer.load_state_dict(torch.load(
                        kwargs["saving_path_model"] + "/backup/sr_ll_estimator_optimizer.checkpoint"))
                    if "sr_hl_estimator" in kwargs and kwargs["sr_hl_estimator"]:
                        self.sr_hl_estimator_optimizer = torch.optim.Adam(self._iterator_sr_hl_estimator_params)
                        self.sr_hl_estimator_optimizer.load_state_dict(torch.load(
                            kwargs["saving_path_model"] + "/backup/sr_hl_estimator_optimizer.checkpoint"))
                    self.is_loaded = True

                    src = kwargs["saving_path_model"] + "/backup"
                    dest = kwargs["saving_path_model"] + "/last"
                    shutil.copytree(src, dest, dirs_exist_ok=True)

            elif "save_first_last" in kwargs and kwargs["save_first_last"] \
                    and not hasattr(self, "save_first_last"):
                print("Saving model high level before training")
                full_state_dict = OrderedDict({
                    k: v for d in (
                        self._iterator_named_filtered_params(self.get_parameters_name_filter()),
                        self._iterator_named_filtered_params(self._sr_ll_estimator_parameters_filter),
                        self._iterator_named_filtered_params(self._sr_hl_estimator_parameters_filter)
                    ) for k, v in d
                })
                self.ac_optimizer = torch.optim.Adam(self._iterator_ac_params, lr=kwargs["lr"])
                self.sr_ll_estimator_optimizer = torch.optim.Adam(self._iterator_sr_ll_estimator_params, lr=kwargs["lr_ll_sr_estimator"])
                torch.save(full_state_dict, kwargs["saving_path_model"] + "/last/model.checkpoint")
                torch.save(self.ac_optimizer.state_dict(), kwargs["saving_path_model"] + "/last/ac_optimizer.checkpoint")
                torch.save(self.sr_ll_estimator_optimizer.state_dict(), kwargs["saving_path_model"] + "/last/sr_ll_estimator_optimizer.checkpoint")
                if "sr_hl_estimator" in kwargs and kwargs["sr_hl_estimator"]:
                    self.sr_hl_estimator_optimizer = torch.optim.Adam(self._iterator_sr_hl_estimator_params, lr=kwargs["lr_hl_sr_estimator"])
                    torch.save(self.sr_hl_estimator_optimizer.state_dict(), kwargs["saving_path_model"] + "/last/sr_hl_estimator_optimizer.checkpoint")

                self.save_first_last = True

            elif "reinitialize" in kwargs and kwargs["reinitialize"]:
                print("Reinitializing model high level")
                model_state_dict_to_reinitialise = torch.load(
                    kwargs["saving_path_model"] + "/last/model_zero.checkpoint")
                self._llm_module.load_state_dict(model_state_dict_to_reinitialise, strict=False)

            elif "reinitialize_optimizer" in kwargs and kwargs["reinitialize_optimizer"]:
                print("Reinitializing optimizer high level")
                self.ac_optimizer = torch.optim.Adam(self._iterator_ac_params, lr=kwargs["lr"])
                torch.save(self.ac_optimizer.state_dict(), kwargs["saving_path_model"] + "/last/ac_optimizer.checkpoint")
                self.sr_ll_estimator_optimizer = torch.optim.Adam(self._iterator_sr_ll_estimator_params, lr=kwargs["lr_ll_sr_estimator"])
                torch.save(self.sr_ll_estimator_optimizer.state_dict(), kwargs["saving_path_model"] + "/last/sr_ll_estimator_optimizer.checkpoint")
                if "sr_hl_estimator" in kwargs and kwargs["sr_hl_estimator"]:
                    self.sr_hl_estimator_optimizer = torch.optim.Adam(self._iterator_sr_hl_estimator_params, lr=kwargs["lr_hl_sr_estimator"])
                    torch.save(self.sr_hl_estimator_optimizer.state_dict(), kwargs["saving_path_model"] + "/last/sr_hl_estimator_optimizer.checkpoint")

            return {}


        elif "save_model_at_epoch_k" in kwargs:
            print("saving model high level at epoch {}".format(kwargs["save_model_at_epoch_k"]))
            full_state_dict = OrderedDict({
                    k: v for d in (
                        self._iterator_named_filtered_params(self.get_parameters_name_filter()),
                        self._iterator_named_filtered_params(self._sr_ll_estimator_parameters_filter),
                        self._iterator_named_filtered_params(self._sr_hl_estimator_parameters_filter)
                    ) for k, v in d
                })
            path_model_epoch_k = kwargs["saving_path_model"] + "/save_epoch_{}/model.checkpoint".format(kwargs["save_model_at_epoch_k"])
            if os.path.exists(path_model_epoch_k):
                torch.save(full_state_dict, path_model_epoch_k)
            else:
                os.makedirs(os.path.join(kwargs["saving_path_model"], "save_epoch_{}".format(kwargs["save_model_at_epoch_k"])), exist_ok=True)
                torch.save(full_state_dict, path_model_epoch_k)
            return {}

        elif "load_version_at_step_k" in kwargs:
            print("Loading model high level at epoch {}".format(kwargs["load_version_at_step_k"]))
            self._llm_module.load_state_dict(torch.load(kwargs["saving_path_model"] + "/save_epoch_{}/model.checkpoint".format(kwargs["load_version_at_step_k"])),
                                                     strict=False)
            return {}

        # if self.use_unsloth:
        #     # Unsloth activates gradient checkpointing after generating, so we properly reset its value
        #     self._llm_module.module._LLM_model.for_training(use_gradient_checkpointing=False)

        if "retrieve_weights_buffer" in kwargs and kwargs["retrieve_weights_buffer"]:
            return self.weights_buffer

        # Update the SR LL estimator
        if "sr_ll_estimator_update" in kwargs and kwargs["sr_ll_estimator_update"]:
            epochs_loss = {"loss": []}
            kwargs["network_type"] = "LL"
            self.gradient_pass_sr_estimator(contexts, candidates, epochs_loss, _current_batch_ids, kwargs)
            return epochs_loss


        # Updates related to the SR HL estimator
        # Update the buffer of weights (old SR HL estimator used to compute LP)
        if "update_buffer_hl_sr_estimator" in kwargs and kwargs["update_buffer_hl_sr_estimator"]:
            self.update_buffer(kwargs["saving_path_buffer_sr_hl_estimator_weights"])
            return {}
        # Set the weights of the old SR HL estimator used to compute LP
        if "set_weights_hl_sr_estimator" in kwargs and kwargs["set_weights_hl_sr_estimator"]:
            self.set_weights()
            return {}
        # Update the SR HL estimator
        if "sr_hl_estimator_update" in kwargs and kwargs["sr_hl_estimator_update"]:
            epochs_loss = {"loss": []}
            kwargs["network_type"] = "HL"
            self.gradient_pass_sr_estimator(contexts, candidates, epochs_loss, _current_batch_ids, kwargs)
            return epochs_loss

        # Update the Actor Critic HL agent
        current_process_buffer = {}
        for k in ['actions', 'advantages', 'returns', 'logprobs', 'values']:
            current_process_buffer[k] = kwargs[k][_current_batch_ids["contexts"]]
        epochs_losses = {
            "value": [],
            "policy": [],
            "loss": [],
            "ratio": [],
            "entropy": [],
            "kl_penalty": []
        }
        modify_minibatch_size = False
        old_minibatch_size = self._minibatch_size
        if len(contexts) <= self._minibatch_size:
            while len(contexts) <= self._minibatch_size / 2:
                self._minibatch_size = int(self._minibatch_size / 2)
        else:
            while len(contexts) > self._minibatch_size:
                self._minibatch_size = int(self._minibatch_size * 2)
        n_minibatches = math.ceil(len(contexts) / self._minibatch_size)
        self.gradient_pass_actor_critic(contexts, candidates, current_process_buffer,
                           n_minibatches, epochs_losses, kwargs)
        if modify_minibatch_size:
            # restore the value of the minibatch size
            self._minibatch_size = old_minibatch_size

        if kwargs["save_after_update"] and int(os.environ.get("RANK")) == 1:
            print("Saving model high level after training")
            src = kwargs["saving_path_model"] + "/last"
            dest = kwargs["saving_path_model"] + "/backup"
            shutil.copytree(src, dest, dirs_exist_ok=True)
            full_state_dict = OrderedDict({
                    k: v for d in (
                        self._iterator_named_filtered_params(self.get_parameters_name_filter()),
                        self._iterator_named_filtered_params(self._sr_ll_estimator_parameters_filter),
                        self._iterator_named_filtered_params(self._sr_hl_estimator_parameters_filter)
                    ) for k, v in d
                })
            torch.save(full_state_dict, kwargs["saving_path_model"] + "/last/model.checkpoint")
            torch.save(self.ac_optimizer.state_dict(), kwargs["saving_path_model"] + "/last/ac_optimizer.checkpoint")
            print("Model saved")
        return {'loss': epochs_losses["loss"], 'value_loss': epochs_losses["value"],
                'policy_loss': epochs_losses["policy"], "ratio": epochs_losses["ratio"],
                "entropy": epochs_losses["entropy"], "kl_penalty": epochs_losses["kl_penalty"]}

    def update_buffer(self, saving_path):
        weights = [p.data.detach().clone() for n, p in self._iterator_named_filtered_params(self._sr_hl_estimator_parameters_filter)]
        self.weights_buffer.append(weights)
        with open(saving_path + "/buffer_sr_hl_estimator_weights.pkl", 'wb') as f:
            pickle.dump(self.weights_buffer, f)

    def set_weights(self):
        for (n, p), w in zip(self._iterator_named_filtered_params(self._sr_hl_estimator_delayed_parameters_filter), self.weights_buffer[0]):
            p.data.copy_(w)