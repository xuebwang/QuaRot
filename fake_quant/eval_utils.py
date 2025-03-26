import utils
import model_utils
import quant_utils
import torch
import os
import logging
from tqdm import tqdm
import random
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import numpy as np

class DatasetToDevice(torch.utils.data.Dataset):
    def __init__(self, data: List, device: Optional[Union[str, torch.device]]):
        super().__init__()
        self.data = data
        self.device = device

    def __getitem__(self, idx):
        if self.device is not None:
            return {name: recursive_to_device(val, self.device) for name, val in self.data[idx].items()}
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

@torch.no_grad()
def recursive_to_device(tensor_or_iterable: Union[Iterable, torch.Tensor], device) -> None:
    if isinstance(tensor_or_iterable, torch.Tensor):
        return tensor_or_iterable.to(device)
    elif isinstance(tensor_or_iterable, tuple):  # Special handling of tuples, since they are immutable
        tmp_list = []
        for i in tensor_or_iterable:
            tmp_list.append(recursive_to_device(i, device))
        return tuple(tmp_list)
    elif isinstance(tensor_or_iterable, Iterable):
        for i in tensor_or_iterable:
            tensor_or_iterable[i] = recursive_to_device(i, device)
        return tensor_or_iterable
    else:
        raise ValueError(f"Cannot move {type(tensor_or_iterable)} to {device}")

@torch.no_grad()
def compute_perplexity(model: torch.nn.Module, data: List[Dict], context_length: int, tokenizer: Any, seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    model = model.to('cuda')
    model = model.eval()

    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    data = DatasetToDevice(data, device='cuda')

    nlls = []
    for sample in tqdm(data, desc="Computing perplexity..."):
        sample_length = sample["input_ids"].shape[1]
        for start_index in range(0, sample_length, context_length * 2):
            end_index = min(start_index + sample_length, sample_length - 1)

            subsample = {
                "input_ids": sample["input_ids"][:, start_index : end_index + 1],
                "attention_mask": sample["attention_mask"][:, start_index : end_index + 1],
            }

            # In case we are using torch.fx, we can not have optional inputs, and we have traced the model with past_key_values inputs, thus we need them here as well.
            if "past_key_values" in sample and isinstance(model, torch.fx.GraphModule):
                subsample["past_key_values"] = sample["past_key_values"]

            # Add BOS token.
            if tokenizer.bos_token_id is not None:
                subsample["input_ids"][:, 0] = tokenizer.bos_token_id

            use_accelerate = hasattr(model, "hf_device_map")
            if not use_accelerate or (use_accelerate and not hasattr(model, "_hf_hook")):
                device = next(model.parameters()).device
                for name, val in subsample.items():
                    subsample[name] = recursive_to_device(val, device)
            else:
                # In accelerate by default `io_same_device=True`, and here we want the of the model output on device.
                device = model._hf_hook.execution_device
                for name, val in subsample.items():
                    subsample[name] = recursive_to_device(val, device)

            lm_logits = model(**subsample)["logits"]

            reference_labels = subsample["input_ids"][:, context_length:]

            shift_logits = lm_logits[:, context_length - 1 : -1]

            # Fuse batch and sequence length dimensions.
            reference_labels = reference_labels.view(reference_labels.shape[-1])
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])

            loss = cross_entropy_loss(shift_logits, reference_labels)

            nlls.append(loss)

    ppl = torch.exp(torch.stack(nlls).mean())

    return ppl

@torch.no_grad()
def evaluator(model, testenc, dev, args):

    model.eval()

    # if 'opt' in args.model:
    #     opt_type = True
    #     llama_type = False
    # elif 'meta' in args.model:
    #     llama_type = True
    #     opt_type = False
    # else:
    #     raise ValueError(f'Unknown model {args.model}')
    llama_type = True
    opt_type = False

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if opt_type:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)

    elif llama_type:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        if hasattr(model.model, 'rotary_emb'):
            # for llama-3.1
            model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)  # (nsamples, seqlen)

    batch_size = args.bsz
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    inps = [0] * nbatches
    cache = {'i': 0, 'attention_mask': None}
    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if llama_type:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
   
    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    if opt_type:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif llama_type:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        position_ids = cache['position_ids']

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)

        # Dump the layer input and output
        if args.capture_layer_io and args.layer_idx == i:
            captured_io = model_utils.capture_layer_io(model_utils.get_model_type(model), layer, inps)
            save_path = model_utils.get_layer_io_save_path(args)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(captured_io, save_path)
            logging.info(f'Dumped layer input and output to: {save_path}')

        for j in range(nbatches):
            if opt_type:
                outs[j] = layer(inps[j], attention_mask=attention_mask)[0]
            elif llama_type:
                outs[j] = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if opt_type:
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)

    elif llama_type:
        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, 'rotary_emb'):
            # for llama-3.1
            model.model.rotary_emb = model.model.rotary_emb.to(dev)

    model.lm_head = model.lm_head.to(dev)
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
    for i in range(nbatches):
        hidden_states = inps[i]
        if opt_type:
            if model.model.decoder.final_layer_norm is not None:
                hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            if model.model.decoder.project_out is not None:
                hidden_states = model.model.decoder.project_out(hidden_states)
        elif llama_type:
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache
    logging.info(f'\n{args.eval_dataset.upper()} PPL: {ppl.item():.3f}')
    return ppl.item()
