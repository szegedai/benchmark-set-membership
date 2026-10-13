###=============================================================================
### Imports
###=============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

###=============================================================================
### Implementation
###=============================================================================

def load_model_and_tokenizer(model_path, return_pipeline=False, device='cpu'):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=False
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    if return_pipeline:
        pipe = pipeline(
            task='text-generation',
            model=model,
            tokenizer=tokenizer
        )

        return model, tokenizer, pipe

    return model, tokenizer


def evaluate_pipeline(pipe, prompt, generation_args):
    messages = [
        {'role': 'system', 'content': 'You are a helpful AI assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    outputs = pipe(
        messages,
        pad_token_id=pipe.tokenizer.eos_token_id,
        **generation_args
    )
    return outputs[0]['generated_text']


def evaluate_model(model, tokenizer, prompt, generation_args):
    messages = [
        {'role': 'system', 'content': 'You are a helpful AI assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors='pt').to(model.device)

    generated_ids = model.generate(**model_inputs, **generation_args)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip('\n')

    return output


def clear_gpu(var_name, local_vars):
    if var_name in local_vars:
        local_vars[var_name].to('cpu')
        torch.cuda.empty_cache()
