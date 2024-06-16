from third_party.models.t5 import T5LayerNorm
from adapters import (AutoAdapterConfig, AdapterController, Adapter)
import os
import regex as re
import logging
from dataclasses import fields
import torch.nn as nn
import json
import torch 

import sys
sys.path.append('..')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_adapter_config(adapter_args, data_args, training_args, config):
    if adapter_args.train_task_adapters or adapter_args.prefix_tuning or adapter_args.bitfit or adapter_args.apply_mixda:
        adapter_config = AutoAdapterConfig.get(
            adapter_args.adapter_config_name)
        adapter_config.input_dim = config.d_model
        adapter_config.output_dim = config.d_model
        if adapter_args.train_task_adapters:
            data_args.tasks = [data_args.task_name]
            adapter_config.tasks = data_args.tasks
        adapter_params = [field.name for field in fields(adapter_args)]
        for p in adapter_params:
            if hasattr(adapter_args, p) and hasattr(adapter_config, p) and\
                    getattr(adapter_args, p) is not None:
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                logger.warning(
                    f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        adapter_config.device = training_args.device
        adapter_config.output_dir = training_args.output_dir
        adapter_config.attn_method = config.attn_method
        adapter_config.ignore_target = config.ignore_target
        adapter_config.attn_prefix = config.attn_prefix_tuning
        adapter_config.fix_attention = config.fix_attention
    else:
        adapter_config = None
    return adapter_config


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_model_params(model, adapter_args, adapter_config):
    """
    Freezes the model parameters based on the given setting in the arguments.
    Args:
      model: the given model.
      adapter_args: defines the adapters arguments.
    """
    # If we are training adapters, we freeze all parameters except the
    # adapter parameters like adapter controllers.
    if adapter_args.train_task_adapters:
        freeze_params(model)
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (AdapterController, Adapter)):
                for param_name, param in sub_module.named_parameters():
                    if param_name.find(adapter_config.tasks[0]) >= 0:
                        param.requires_grad = True
        for n, p in model.named_parameters():
            if 'gate' in n:
                p.requires_grad = True
        if adapter_args.add_task_embedding:
            for n, p in model.named_parameters():
                if 'task_shared' in n:
                    p.requires_grad = True
                if 'temperature' in n:
                    p.requires_grad = True

    # Unfreezes adamix
    if adapter_args.num_experts is not None and adapter_args.num_experts > 1:
        freeze_params(model)
        for n, p in model.named_parameters():
            if 'ExpertSoup' in n and 'expert_score_weight' not in n:
                p.requires_grad = True
    # Unfreezes mixda
    if adapter_args.apply_mixda:
        freeze_params(model)
        for n, p in model.named_parameters():
            if 'gating' in n:
                p.requires_grad = True
            # if 'kas' in n:
            #     p.requires_grad = True
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (AdapterController, Adapter)):
                for param_name, param in sub_module.named_parameters():
                    if param_name.find(adapter_config.tasks[0]) >= 0:
                        param.requires_grad = True
    
    
    # Unfreezes LoRA
    if adapter_args.train_lora:
        freeze_params(model)
        for n, p in model.named_parameters():
            if 'lora_' in n:
                p.requires_grad = True
            if adapter_args.add_task_embedding:
                if 'task_shared' in n:
                    p.requires_grad = True

    # Unfreezes last linear layer of decoder.
    if adapter_args.unfreeze_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # Unfreezes layer norms.
    if adapter_args.unfreeze_layer_norms:
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                # this will not consider layer norms inside adapters then.
                if len(name.split(".")) < 7:
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

    if adapter_args.prefix_tuning:
        freeze_params(model)
        if adapter_config.attn_prefix is False:
            for n, m in model.named_parameters():
                if "prefix_shared" == n:
                    m.requires_grad = True
                # update grad
                if "W_weighting" == n:
                    m.requires_grad = True
        else:
            # update attention related weights.
            if adapter_config.attn_method == "dot":
                for n, m in model.named_parameters():
                    if "mul_prefix_emb" == n:
                        m.requires_grad = True

            elif adapter_config.attn_method == "linear":
                for n, m in model.named_parameters():

                    if "encoder.attn_Wa.weight" == n:
                        m.requires_grad = True
                    if "prefix_shared" == n and adapter_config.ignore_target is False:
                        m.requires_grad = True

            elif adapter_config.attn_method == "sub":
                for n, m in model.named_parameters():
                    if "encoder.attn_W_down.weight" == n and adapter_config.fix_attention is False:
                        m.requires_grad = True
                    if "encoder.attn_W_up.weight" == n and adapter_config.fix_attention is False:
                        m.requires_grad = True
                    if "prefix_shared" == n and adapter_config.ignore_target is False:
                        m.requires_grad = True
            elif adapter_config.attn_method == "constant":
                for n, m in model.named_parameters():
                    if "prefix_shared" == n and adapter_config.ignore_target is False:
                        m.requires_grad = True
            elif adapter_config.attn_method == "concat":
                for n, m in model.named_parameters():
                    if "encoder.attn_Wa.weight" == n or "attn_va" == n:
                        m.requires_grad = True

    # For bitfit we freeze the whole model except for the biases and the final classifier layer.
    if adapter_args.bitfit:
        freeze_params(model)
        # unfreeze bias terms.
        for n, p in model.named_parameters():
            if ".bias" in n:
                p.requires_grad = True

        # unfreeze the classifier.
        for param in model.lm_head.parameters():
            param.requires_grad = True
        if adapter_args.freeze_bitfit_lm_head:
            for n, param in model.lm_head.named_parameters():
                if "bias" in n:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if adapter_args.freeze_bitfit_lm_head_all:
            for n, param in model.lm_head.named_parameters():
                param.requires_grad = False


def get_adapter_params_names(model):
    """
    Returns adapter related parameters names.
    Args:
      model: the given model.
    """
    params_names = []
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, (AdapterController, Adapter)):
            for param_name, param in sub_module.named_parameters():
                params_names.append(name+"."+param_name)
    return params_names


def get_layer_norm_params_names(model):
    """Returns the layer norms parameters.
    Args:
        model: the given model.
    """
    params_names = []
    for name, sub_module in model.named_modules():
        if isinstance(sub_module,  (T5LayerNorm, nn.LayerNorm)):
            for param_name, param in sub_module.named_parameters():
                params_names.append(name+"."+param_name)
    return params_names


def get_last_checkpoint(output_dir):
    if os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')):
        return output_dir
    return None


def pad_punctuation(text):
    """Re-implementation of _pad_punctuation in t5. This function adds spaces
    around punctuation. While this pads punctuation as expected, it has the 
    unexpected effected of padding certain unicode characters with accents, with
    spaces as well. For instance: "François" becomes "Fran ç ois"""
    # Pad everything except for: underscores (_), whitespace (\s),
    # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
    text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', str(text))
    # Collapse consecutive whitespace into one space.
    text = re.sub(r'\s+', ' ', text)
    return text


def init_task_param(config, tokenizer):
    logger.info('init task param')
    task = config.target_task[0]
    task_desc_map = {
        "qnli": "Given a question and a context sentence, the task is to determine whether the context sentence contains the answer to the question.",
        "mnli": "Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis, contradicts the hypothesis, or neither.",
        "qqp": "Given a pair of sentences, the task is to determine if the two sentences are semantically equivalent or not.",
        "cola": "Given a sentence, the task is to judge the grammatical acceptability of the sentence.",
        "rte": "Given a premise sentence and a hypothesis sentence, the task is to determine whether the hypothesis can be inferred from the premise.",
        "qqp": "Given two questions, the task is to determine whether the two questions have the same intent or meaning.",
        "mrpc": "Given a pair of sentences, the task is to determine whether the two sentences are semantically equivalent or not.",
        "stsb": "Given a pair of sentences, the task is to measure the degree of semantic similarity or relatedness between pairs of sentences.",
        "sst2": "Given a sentence, the task is to predict whether a given sentence expresses a positive or negative sentiment.",
        "superglue-cb": "Given a premise and a hypothesis, the task is to determine the type and strength of the commitment being expressed.",
        "superglue-wic": "Given a target word and a pair of sentences, the task is to determine if the target word has the same meaning in two different contexts.",
        "superglue-wsc-fixed": "Given a set of sentences that contain an ambiguous pronoun, the task is to determine the referent of the ambiguous pronoun based on the context provided.",
        "superglue-boolq": "Given a question and a paragraph, the task is to determine if the question can be answered with a simple 'true' or 'false' based on the given passage of text.",
        "superglue-multirc": "Given a passage of text and a set of multiple-choice questions, the task is to select the correct answer choice for each question based on the information provided in the passage.",
        "superglue-record": "Given a passage and a cloze-style question about the article in which one entity is masked out, the task is to predict the masked out entity from a list of possible entities in the provided passage.",
        'squad': "Given an article and a corresponding question about the article, the task is to answer the question accurately based on the provided context in the articles.",
        'nq': "Given an article and a corresponding question about the article, the task is to answer the question accurately based on the provided context in the articles.",
        'newsqa': "Given an article and a corresponding question about the article, the task is to answer the question accurately based on the provided context in the articles.",
        'hotpotqa': "Given an article and a corresponding question about the article, the task is to answer the question accurately based on the provided context in the articles.",
        'searchqa': "Given an article and a corresponding question about the article, the task is to answer the question accurately based on the provided context in the articles.",
        'scitail': "Given a premise and a hypothesis, the task is to classify the relationship between the premise and the hypothesis as entail or neutral.",
        'yelp_polarity': "Given a Yelp sentence, the task is to predict the sentiment polarity (positive or negative) of customer reviews from the Yelp dataset.",
        'winogrande': "Given a sentence and two options, the task is to choose the right option for a given sentence which requires commonsense reasoning.",
        'paws': "Given a pair of sentence, where one sentence is a paraphrase of the other. The task is to determine if the given sentence pair is a paraphrase or not."
    }
    task_token_id = tokenizer(task_desc_map[task])['input_ids']
    config.task_embedding_len = len(task_token_id)
    config.task_embedding_init_token = task_token_id
    logger.info(f"task embedding len={config.task_embedding_len}, token={task_token_id}, desc={task_desc_map[task]}")


def modify_model_after_init(model:nn.Module, training_args, adapter_args, adapter_config):
    # Freezes model parameters.
    freeze_model_params(model, adapter_args, adapter_config)
    # load lora
    if adapter_args.add_lora and adapter_args.load_lora_path is not None:
        lora_list = adapter_args.load_lora_path.split(',')
        if len(lora_list) == 1:
            logger.info(f'load lora weight from:{adapter_args.load_lora_path}')
            lora_dict = torch.load(adapter_args.load_lora_path)
            model.load_state_dict(lora_dict, strict=False)
        else:
            for i,path in enumerate(lora_list):
                logger.info(f'load lora weight from:{path}')
                lora_dict = torch.load(path)
                for key in list(lora_dict.keys()):
                    # 创建新的键名
                    new_key = key + f'.{i}'
                    # 将原键对应的值赋给新键
                    lora_dict[new_key] = lora_dict.pop(key)
                model.load_state_dict(lora_dict, strict=False)

    if adapter_args.load_adapter_path is not None:
        adapter_list = adapter_args.load_adapter_path.split(',')
        logger.info(adapter_list)
        if len(adapter_list) == 1:
            logger.info(f'load adapter weight from:{adapter_args.load_adapter_path}')
            adapter_dict = torch.load(adapter_args.load_adapter_path)            
            model.load_state_dict(adapter_dict, strict=False)
        else:
            for adapter in adapter_list:
                logger.info(f'load many adapter weight from:{adapter}')
                adapter_dict = torch.load(adapter)
                model.load_state_dict(adapter_dict, strict=False)

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.info(
        "***** Model Trainable Parameters {} *****".format(trainable_params))
    if training_args.print_num_parameters:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("##### Parameter name %s", name)
        total_lm_head_params = sum(p.numel()
                                   for p in model.lm_head.parameters())
        total_trainable_params = sum(p.numel()
                                     for p in model.parameters() if p.requires_grad)
        total_trainable_bias_params = sum(p.numel(
        ) for n, p in model.named_parameters() if p.requires_grad and n.endswith(".b"))
        total_trainable_layernorm_params = sum(p.numel() for n, p in model.named_parameters(
        ) if p.requires_grad and ".layer_norm.weight" in n)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total trainable parameters %s", total_trainable_params)
        logger.info("Total trainable bias parameters %s",
                    total_trainable_bias_params)
        logger.info("Total trainable layer norm parameters %s",
                    total_trainable_layernorm_params)
        logger.info("Total parameters %s", total_params)
        t5_base_params = 222882048
        # total params since we have 8 task, it is Y = 1*BERT + 8*ADAPTERS, and final number is Y/BERT ("1.3x")
        total_params_ratio = ((total_params-t5_base_params)
                              * 8+t5_base_params)/t5_base_params
        total_trainable_params_percent = (
            total_trainable_params/t5_base_params)*100
        total_trainable_bias_params_percent = (
            total_trainable_bias_params/total_trainable_params)*100
        total_trainable_layernorm_params_percent = (
            total_trainable_layernorm_params/total_trainable_params)*100
        total_trainable_lm_head_params_percent = (
            total_lm_head_params/t5_base_params)*100
        logger.info("For adapters/prompt-tuning, total params %s",
                    total_params_ratio)
        logger.info("For intrinsic, total params %s",
                    total_params/t5_base_params)
        logger.info("Total trainable params %s",
                    total_trainable_params_percent)
        logger.info("Total trainable bias params %s",
                    total_trainable_bias_params_percent)
        logger.info("Total trainable layer norm params %s",
                    total_trainable_layernorm_params_percent)
        logger.info("Total lm_head params %s",
                    total_trainable_lm_head_params_percent)
    return model


def save_json(filepath, dictionary):
    with open(filepath, "w") as outfile:
        json.dump(dictionary, outfile)


def read_json(filepath):
    f = open(filepath,)
    return json.load(f)


def save_training_config(config_file, output_dir):
    json_data = read_json(config_file)
    save_json(os.path.join(output_dir, "training_config.json"), json_data)

def save_prompts(model, output_dir, attn_prefix_tuning, shared_attn, num_target, task_name):
    for name, param in model.named_parameters():
        # Save prompt weights.
        if attn_prefix_tuning is False and ("prefix_shared" in name or "prefix" in name):
            shared_params = param
            torch.save(shared_params, os.path.join(
                output_dir, "prefix_embeddings.pt"))
        elif attn_prefix_tuning is True and name == "prefix_shared":
            shared_params = param
            if shared_attn is True:
                for i in range(num_target):
                    torch.save(shared_params[i], os.path.join(
                        output_dir, "prefix_embeddings_{}.pt".format(task_name[i])))
            else:
                torch.save(shared_params, os.path.join(
                    output_dir, "prefix_embeddings.pt"))

        # Save attention and layer norm weights.
        if attn_prefix_tuning is True and "encoder.attn_Wa.weight" == name:
            attn_weights_params = param
            torch.save(attn_weights_params, os.path.join(
                output_dir, "attn_Wa_weights.pt"))
        if attn_prefix_tuning is True and "encoder.attn_W_down.weight" == name:
            attn_weights_params = param
            torch.save(attn_weights_params, os.path.join(
                output_dir, "attn_W_down.pt"))
        if attn_prefix_tuning is True and "encoder.attn_W_up.weight" == name:
            attn_weights_params = param
            torch.save(attn_weights_params, os.path.join(
                output_dir, "attn_W_up.pt"))
        if attn_prefix_tuning is True and "encoder.layer_norm.weight" == name:
            attn_weights_params = param
            torch.save(attn_weights_params, os.path.join(
                output_dir, "layer_norm_weight.pt"))
        if attn_prefix_tuning is True and "encoder.layer_norm.bias" == name:
            attn_weights_params = param
            torch.save(attn_weights_params, os.path.join(
                output_dir, "layer_norm_bias.pt"))