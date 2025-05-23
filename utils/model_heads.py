try:
    from unsloth import FastLanguageModel
    print(f"Successfully imported unsloth!")
except Exception as err:
    print("Failed to import unsloth")

import os
from typing import List

import torch
from torch.nn import functional as F
import math
from lamorel import BaseModuleFunction, BaseModelInitializer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING, PeftType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class LogScoringModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":
            if self._pre_encoded_input:
                logits = forward_outputs["logits"][:, :-1, :]
                output_tokens = minibatch["input_ids"][:, 1:]
            else:  # hence input should be removed from result
                end_of_context_positions = [len(_context["input_ids"]) for _context in tokenized_contexts]
                raw_logits, raw_output_tokens = [], []
                max_len = 0
                for i in range(len(tokenized_contexts)):
                    raw_logits.append(forward_outputs["logits"][i, end_of_context_positions[i]:-1, :])
                    raw_output_tokens.append(minibatch["input_ids"][i, end_of_context_positions[i]+1:])
                    if len(raw_output_tokens[-1]) > max_len:
                        max_len = len(raw_output_tokens[-1])

                logits = torch.stack([
                    torch.nn.functional.pad(_logits, (0, 0, 0, max_len - len(_logits)), value=0)
                    for _logits in raw_logits
                ])
                output_tokens = torch.stack([
                    torch.nn.functional.pad(_tokens, (0, max_len - len(_tokens)), value=self._pad_token)
                    for _tokens in raw_output_tokens
                ])

        else:
            logits = forward_outputs["logits"][:, :-1, :]  # skip </s> token appended by tokenizer
            output_tokens = minibatch["decoder_input_ids"][:, 1:]  # skip pad token

        # we calculate the log_probs of the output tokens
        logits_full = logits[:, 0, :]
        # print("cuda:{} output_tokens: {} logits_full: {} {}".format(logits_full.get_device(), output_tokens, logits_full.size(), logits_full))
        mask = torch.full_like(logits_full, -math.inf)
        gather_idx = [torch.tensor(kwargs['dict_tokens']['{}'.format([_o.item() for _o in output_t[output_t > 0].cpu().numpy()])])
                      for
                      output_t in output_tokens]
        """if os.environ["RANK"]=="4":
            print("gather_idx: {}".format(gather_idx))"""
        for i_gather, g_idx in enumerate(gather_idx):
            mask[i_gather, g_idx] = 0
        logits_masked = logits_full + mask
        logits_masked = F.log_softmax(logits_masked, dim=-1)

        # TODO replace "cuda:{}".format(logits_masked.get_device()) by a direct call to the device
        if logits_masked.get_device() == -1:
            gather_idx_2 = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(kwargs['dict_tokens']['{}'.format([_o.item() for _o in output_t[output_t > 0].cpu().numpy()])]) for
                 output_t in
                 output_tokens], batch_first=True, padding_value=0)
        else:

            gather_idx_2 = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(kwargs['dict_tokens']['{}'.format([_o.item() for _o in output_t[output_t > 0].cpu().numpy()])],
                              device="cuda:{}".format(logits_masked.get_device())) for output_t in
                 output_tokens], batch_first=True, padding_value=0).squeeze()
        """if os.environ["RANK"]=="4":
            print("kwargs['dict_tokens']: {}\ngather_idx_2: {}".format(kwargs['dict_tokens'], gather_idx_2))"""
        while gather_idx_2.dim() != logits_masked.dim():
            # print("gather_idx_2 before unsqueeze: {}".format(gather_idx_2))
            if len(kwargs['dict_tokens']) == 1:
                # print("before gather_idx_2: {}, logits_masked: {}".format(gather_idx_2, logits_masked))
                gather_idx_2 = gather_idx_2.unsqueeze(0)
                # print("after gather_idx_2: {}".format(gather_idx_2))

            elif all([len(v) == 1 for v in kwargs['dict_tokens'].values()]):
                # print("before gather_idx_2: {}, logits_masked: {}".format(gather_idx_2, logits_masked))
                gather_idx_2 = gather_idx_2.unsqueeze(-1)
                # print("after gather_idx_2: {}".format(gather_idx_2))
            elif gather_idx_2.dim()<=1 and logits_masked.size()[0]>1:
                gather_idx_2 = gather_idx_2.unsqueeze(-1)
            elif gather_idx_2.dim()<=1 and logits_masked.size()[0]==1:
                gather_idx_2 = gather_idx_2.unsqueeze(0)
            else:
                print("issue with kwargs['dict_tokens']: {}, gather_idx_2: {}, logits_masked: {} \nnew_gather_idx_2: {} ".format(kwargs['dict_tokens'], gather_idx_2, logits_masked, gather_idx_2.unsqueeze(-1)))
                """print("issue with kwargs['dict_tokens']: {}, gather_idx_2: {}, logits_masked: {} \nnew_gather_idx_2: {} \ntokens_logprobs: {}".format(kwargs['dict_tokens'], gather_idx_2, logits_masked, gather_idx_2.unsqueeze(-1), torch.gather(logits_masked, 1, gather_idx_2.unsqueeze(0)).squeeze(-1).to(
                torch.float32)))"""
            # print("gather_idx_2 after unsqueeze: {}".format(gather_idx_2))

        try:
            tokens_logprobs = \
            torch.gather(logits_masked, 1, gather_idx_2).squeeze(-1).to(
                torch.float32)
        except:
            print("original gather_idx_2: {}".format(torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(kwargs['dict_tokens']['{}'.format([_o.item() for _o in output_t[output_t > 0].cpu().numpy()])],
                              device="cuda:{}".format(logits_masked.get_device())) for output_t in
                 output_tokens], batch_first=True, padding_value=0).squeeze()))
            print("gather_idx_2: {}".format(gather_idx_2))
            print("logits_masked: {}".format(logits_masked))
            raise Exception("Error in gather operation")
        if tokens_logprobs.dim() < 2:
            # print("tokens_logprobs before unsqueeze: {}".format(tokens_logprobs))
            tokens_logprobs = tokens_logprobs.unsqueeze(-1)
            # print("tokens_logprobs after unsqueeze: {}".format(tokens_logprobs))

        """if os.environ["RANK"]=="4":
            print("tokens_logprobs: {}".format(tokens_logprobs))
            print("log_prob_full_distrib: {}".format(logits_full))"""
        return {"tokens_logprobs": tokens_logprobs.cpu(),
                "log_prob_full_distrib": logits_full.cpu()}

    def get_parameters_name_filter(self):
        return lambda n: '.default.' in n


class ValueHeadModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        if 'hidden_size' in self.model_config.attribute_map:
            _hidden_size_key = self.model_config.attribute_map['hidden_size']
        else:
            if "word_embed_proj_dim" in self.model_config.to_dict():
                _hidden_size_key = "word_embed_proj_dim"
            elif "hidden_size" in self.model_config.to_dict():
                _hidden_size_key = "hidden_size"
            else:
                print(self.model_config.to_dict())
                raise NotImplementedError("Unknown hidden size key")

        self._llm_hidden_size = self.model_config.to_dict()[_hidden_size_key]
        self.output_value = torch.nn.Linear(1024, 1).to(self.device)
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's hidden from last token in context
        if self._model_type == "causal":
            if self._pre_encoded_input:
                model_head = forward_outputs['hidden_states'][-1][:, :-1, :]
            else:  # hence input should be removed from result
                end_of_context_positions = [len(_context["input_ids"]) for _context in tokenized_contexts]
                if "print_all" in kwargs and kwargs["print_all"]:
                    print("end_of_context_positions: {}".format(end_of_context_positions))
                raw_hidden_states = []
                max_len = 0
                for i in range(len(tokenized_contexts)):
                    raw_hidden_states.append(forward_outputs["hidden_states"][-1][i, end_of_context_positions[i]:-1, :])
                    if "print_all" in kwargs and kwargs["print_all"]:
                        print("minibatch[input_ids][{}] size {}: {}".format(i, minibatch["input_ids"][i].size(),  minibatch["input_ids"][i]))
                    if len(minibatch["input_ids"][i, end_of_context_positions[i]+1:]) > max_len:
                        max_len = len(minibatch["input_ids"][i, end_of_context_positions[i]+1:])
                if "print_all" in kwargs and kwargs["print_all"]:
                    print("raw_hidden_states: {}".format(raw_hidden_states))
                    print("max_len: {}".format(max_len))

                model_head = torch.stack([
                    torch.nn.functional.pad(_hidden_states, (0, 0, 0, max_len - len(_hidden_states)), value=0)
                    for _hidden_states in raw_hidden_states
                ])

        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, :-1, :]

        value_representation = self.value_head_op(model_head.to(torch.float32).to(self.device))
        value = self.output_value(value_representation)
        return {"value": value.cpu(), "model_head": model_head.cpu(), "value_rep": value_representation.cpu()}

    def get_parameters_name_filter(self):
        raise NotImplementedError()
        # return lambda n: f'.value.' in n or f'.{self._adapters}.' in n

class SR_LL_Estimator_ModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input, name, adapters):
        super().__init__()
        self._model_type = model_type
        self._pre_encoded_input = pre_encoded_input
        self._name = name
        self._adapters = adapters

    def initialize(self):
        if 'hidden_size' in self.model_config.attribute_map:
            _hidden_size_key = self.model_config.attribute_map['hidden_size']
        else:
            if "word_embed_proj_dim" in self.model_config.to_dict():
                _hidden_size_key = "word_embed_proj_dim"
            elif "hidden_size" in self.model_config.to_dict():
                _hidden_size_key = "hidden_size"
            else:
                print(self.model_config.to_dict())
                raise NotImplementedError("Unknown hidden size key")

        self._llm_hidden_size = self.model_config.to_dict()[_hidden_size_key]
        self.output_sr_ll_estimation = torch.nn.Linear(1024, 1).to(self.device)
        self.sr_ll_estimator_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 1024),
            torch.nn.SiLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.SiLU(),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's hidden from last token in context
        if self._model_type == "causal":
            if self._pre_encoded_input:
                model_head = forward_outputs['hidden_states'][-1][:, 0, :]
            else:  # hence input should be removed from result
                end_of_context_positions = [len(_context["input_ids"]) for _context in tokenized_contexts]
                raw_hidden_states = []
                max_len = 0
                for i in range(len(tokenized_contexts)):
                    raw_hidden_states.append(forward_outputs["hidden_states"][-1][i, end_of_context_positions[i], :])
                    if len(minibatch["input_ids"][i, end_of_context_positions[i]+1:]) > max_len:
                        max_len = len(minibatch["input_ids"][i, end_of_context_positions[i]+1:])

                model_head = torch.stack(raw_hidden_states)

        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, -1, :]

        sr_ll_estimated_representation = self.sr_ll_estimator_head_op(model_head.to(torch.float32).to(self.device))
        sr_ll_estimated = self.output_sr_ll_estimation(sr_ll_estimated_representation)
        return {"sr_ll_estimated": sr_ll_estimated.cpu()}

    def get_parameters_name_filter(self):
        # LoRA default params + MLP head
        return lambda n: f'.{self._name}.' in n or f'.{self._adapters}.' in n


class SR_HL_Estimator_ModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input, name, adapters):
        super().__init__()
        self._model_type = model_type
        self._pre_encoded_input = pre_encoded_input
        self._name = name
        self._adapters = adapters

    def initialize(self):
        if 'hidden_size' in self.model_config.attribute_map:
            _hidden_size_key = self.model_config.attribute_map['hidden_size']
        else:
            if "word_embed_proj_dim" in self.model_config.to_dict():
                _hidden_size_key = "word_embed_proj_dim"
            elif "hidden_size" in self.model_config.to_dict():
                _hidden_size_key = "hidden_size"
            else:
                print(self.model_config.to_dict())
                raise NotImplementedError("Unknown hidden size key")

        self._llm_hidden_size = self.model_config.to_dict()[_hidden_size_key]
        self.output_sr_hl_estimation = torch.nn.Linear(1024, 1).to(self.device)
        self.sr_hl_estimator_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 1024),
            torch.nn.SiLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.SiLU(),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's hidden from last token in context
        if self._model_type == "causal":
            if self._pre_encoded_input:
                model_head = forward_outputs['hidden_states'][-1][:, 0, :]
            else:  # hence input should be removed from result
                end_of_context_positions = [len(_context["input_ids"]) for _context in tokenized_contexts]
                raw_hidden_states = []
                max_len = 0
                for i in range(len(tokenized_contexts)):
                    raw_hidden_states.append(forward_outputs["hidden_states"][-1][i, end_of_context_positions[i], :])
                    if len(minibatch["input_ids"][i, end_of_context_positions[i]+1:]) > max_len:
                        max_len = len(minibatch["input_ids"][i, end_of_context_positions[i]+1:])

                model_head = torch.stack(raw_hidden_states)
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, -1, :]

        sr_hl_estimated_representation = self.sr_hl_estimator_head_op(model_head.to(torch.float32).to(self.device))
        sr_hl_estimated = self.output_sr_hl_estimation(sr_hl_estimated_representation)
        return {"sr_hl_estimated": sr_hl_estimated.cpu()}

    def get_parameters_name_filter(self):
        # LoRA default params + MLP head
        return lambda n: f'.{self._name}.' in n or f'.{self._adapters}.' in n


class SequentialInitializer(BaseModelInitializer):
    def __init__(self, initializers: List[BaseModelInitializer]):
        super().__init__()
        self._initializers = initializers

    def initialize_model(self, model):
        for _initializer in self._initializers:
            model = _initializer.initialize_model(model)

        return model


class WeightsLoaderInitializer(BaseModelInitializer):
    def __init__(self, weights_path, load_state_dict_strict):
        super().__init__()
        self._weights_path = weights_path
        self._load_state_dict_strict = load_state_dict_strict

    def initialize_model(self, model):
        if self._weights_path is not None:
            loaded_ddp_dict = torch.load(self._weights_path + "/model.checkpoint")
            hf_llm_module_dict = {_k.replace('module.', ''): _v for _k, _v in loaded_ddp_dict.items()}
            model.load_state_dict(state_dict=hf_llm_module_dict, strict=self._load_state_dict_strict)

        return model


class PeftInitializer(BaseModelInitializer):
    def __init__(self, model_type, model_name, use_unsloth, use_lora, use_4bit, r, alpha, use_cache=True,
                 additional_target_modules=None,
                 add_target_adapters=False, override_target_modules=None):
        super().__init__()
        self._model_type = model_type
        self._model_name = model_name
        self._use_unsloth = use_unsloth
        # print("peft use_unsloth: {}".format(use_unsloth))
        self._use_lora = use_lora
        self._use_4bit = use_4bit
        self._r = r
        self._alpha = alpha
        self._additional_target_modules = additional_target_modules
        self._use_cache = use_cache
        self._add_target_adapters = add_target_adapters
        self._override_target_modules = override_target_modules

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _get_model_config(self, config_model_type):
        if self._override_target_modules is not None:
            target_modules = self._override_target_modules
        else:
            target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[config_model_type]
            if self._additional_target_modules is not None:
                target_modules.extend(self._additional_target_modules)

        if self._model_type == "seq2seq":
            task_type = "SEQ_2_SEQ_LM"
        else:
            task_type = "CAUSAL_LM"

        return LoraConfig(
            r=self._r,
            lora_alpha=self._alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=task_type
        )

    def initialize_model(self, model):
        if self._use_lora:
            llm_module = model._modules['_LLM_model']
            """if self._model_type == "seq2seq" or not self._use_cache:
                llm_module.gradient_checkpointing_enable()"""  # reduce number of stored activations



            # Init adapters
            config = self._get_model_config(llm_module.config.model_type)
            if self._use_unsloth:
                 unsloth_peft_config = config.to_dict()
                 del unsloth_peft_config["task_type"]
                 peft_model = FastLanguageModel.get_peft_model(
                    llm_module,
                    **unsloth_peft_config,
                    use_gradient_checkpointing="unsloth" if not self._use_cache else False
                )
            else:
                if self._use_4bit:
                    llm_module = prepare_model_for_kbit_training(llm_module)
                peft_model = get_peft_model(llm_module, config)

            # Add sr_LL adapters
            peft_model.add_adapter("sr_LL_adapters", config)

            # Add sr_HL adapters
            peft_model.add_adapter("sr_HL_adapters", config)

            # Add sr_HL adapters delayed
            peft_model.add_adapter("sr_HL_adapters_delayed", config)

            parent_module_device = None
            for name, param in peft_model.named_modules():
                if name.split(".")[-1].startswith("lora_"):
                    if hasattr(param, "weight"):
                        param.to(parent_module_device)
                else:
                    if hasattr(param, "weight"):
                        parent_module_device = param.weight.device
                    else:
                        parent_module_device = None

                if (name.split('.')[-1] == "default" or
                    name.split('.')[-1] in ["sr_LL_adapters","sr_HL_adapters", "sr_HL_adapters_delayed"]) and hasattr(param, "weight"):
                    param.weight.requires_grad = True

            model._modules['_LLM_model'] = peft_model

        model.eval()  # Important to ensure ratios are 1 in first minibatch of PPO (i.e. no dropout)
        model._modules['_LLM_model'].config.use_cache = self._use_cache
        self._print_trainable_parameters(model)
        return model