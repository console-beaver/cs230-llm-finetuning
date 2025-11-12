#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from single_inference import evaluate_problem
from tqdm import tqdm
import numpy as np
import random

if __name__ == '__main__':
    print('loading dataset...')

    dataset = load_dataset('newfacade/LeetCodeDataset')

    print('finished loading dataset')
    print('loading model...')

    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")
    model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-350M")
    model = model.to('cuda')

    print('loaded model')
    print('iterating through data...')

    combined_dataset = concatenate_datasets([dataset['train'], dataset['test']])

    n = len(combined_dataset)
    correct = np.zeros((n), dtype=int)
    errors = np.zeros((n), dtype=int)
    total = np.zeros((n), dtype=int)

    # for i, problem in tqdm(enumerate(combined_dataset), total=n):
    i = 0
    while True:
        print(i)
        if i == 10: break
        problem = combined_dataset[random.randint(0, n - 1)]
        model_out, _, times, stdout = evaluate_problem(model, tokenizer, problem)
        if len(stdout) > len('stdout= '):
            processed = [int(x) for x in stdout.split()[1:]]
            if len(processed) == 3:
                correct[i] = processed[0]
                errors[i] = processed[1]
                total[i] = processed[2]
        if total[i] > 0: i += 1  # valid out

    valid_mask = total != 0
    pass_percentage = correct[valid_mask] / total[valid_mask]
    error_percentage = errors[valid_mask] / total[valid_mask]
    print('mean % passed: ', np.mean(pass_percentage) * 100, '%')
    print('std % passed:  ', np.std(pass_percentage))
    print('mean % errored:', np.mean(error_percentage) * 100, '%')
    print('std % errored: ', np.std(error_percentage))
    print(correct[:10])
    print(errors[:10])
    print(total[:10])
