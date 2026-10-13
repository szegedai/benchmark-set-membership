###=============================================================================
### Imports
###=============================================================================

import argparse
import json
import os
import sys
from itertools import product
from os import path

import yaml
from tqdm import tqdm
from transformers import set_seed

from src.model import *
from src.prompts import *
from src.sets import *

###=============================================================================
### Arguments
###=============================================================================

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--setting',
                        type=str, default=None, required=True)
    parser.add_argument('--model',
                        type=str, default=None, required=True)
    parser.add_argument('--set',
                        type=str, default=None, required=True)
    parser.add_argument('--gpu',
                        type=int, default=0)
    parser.add_argument('--seed',
                        type=int, default=0)
    parser.add_argument('--force',
                        action='store_true')

    return parser.parse_args(args)

###=============================================================================
### Functions
###=============================================================================

def get_filepath(save_dir, model_path, template, set_name, feature):
    model = model_path.replace('/', '--')
    features = '_'.join(map(str, get_feature_ids(feature, template).values()))
    fname = f'{template["id"]}__{features}.json'
    return path.join(save_dir, model, set_name, fname)

###=============================================================================
### Functionality
###=============================================================================

def run(config):
    # set seed
    set_seed(config.seed)

    # set device
    device = f'cuda:{config.gpu}'
    print(f'Using device: {device}')

    # load model
    generation_args = {
        'max_new_tokens': 128,
        'return_full_text': False,
        'do_sample': False,
        'temperature': None,
        'top_p': None
    }

    _, _, pipe = load_model_and_tokenizer(
        model_path=config.model,
        return_pipeline=True,
        device=device
    )

    # load data
    fp_sets = './data/config/sets.yaml'
    fp_prompts = f'./data/config/prompts_{config.setting}.yaml'
    save_dir = f'./data/results/results_{config.setting}'

    temp_save_dir = path.join(save_dir, config.model.replace('/', '--'))
    temp_save_dir = path.join(temp_save_dir, config.set)
    os.makedirs(temp_save_dir, exist_ok=True)

    with open(fp_sets, mode='rb') as f:
        set_data = yaml.safe_load(f)
        set_data = set_data[config.set]

    with open(fp_prompts, mode='rb') as f:
        prompt_data = yaml.safe_load(f)

    prefix = prompt_data['prefix']

    templates = prompt_data['templates']

    features = prompt_data['features']
    features_grid = get_feature_grid(features)


    # grid search
    feature_grid = product(templates, features_grid)

    i = 0
    n_feature_grid = len(templates) * len(features_grid)

    for template, feature in feature_grid:
        i += 1

        fp = get_filepath(save_dir, config.model, template, config.set, feature)

        # skip if evaluated
        if path.exists(fp) and not config.force:
            continue

        # evaluate
        print(fp)

        all_output = list()
        set_perms = generate_setinclusion_permutations(
            elements=set_data['set'],
            intruder=set_data['intruder']
        )
        pbar = tqdm(set_perms, desc=f'Prompt {i}/{n_feature_grid}', ncols=80)
        for set_perm in pbar:
            if config.setting.startswith('natural'):
                create_prompt_fn = create_prompt_natural
            elif config.setting.startswith('coding_simple'):
                create_prompt_fn = create_prompt_coding_simple
            elif config.setting.startswith('coding_algorithmic'):
                create_prompt_fn = create_prompt_coding_algorithmic

            prompt = create_prompt_fn(
                template=template,
                feature=feature,
                elements=set_perm['elements'],
                element=set_perm['element']
            )

            prompt = prefix + ' ' + prompt

            output = evaluate_pipeline(pipe, prompt, generation_args)

            all_output.append({
                'prompt': prompt,
                'answer': output,
                'features': get_feature_ids(feature, template),
                **set_perm
            })

        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(all_output, f, ensure_ascii=False, indent=4)

###=============================================================================
### Run
###=============================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    config = parse_args(args)
    run(config)
