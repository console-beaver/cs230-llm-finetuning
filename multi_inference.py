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

    for i, problem in tqdm(enumerate(combined_dataset), total=n):
        _, _, times, stdout = evaluate_problem(model, tokenizer, problem)
        if type(stdout) == int: errors[i] = total[i] = stdout  # syntax error
        else: correct[i], errors[i], total[i] = [int(x) for x in stdout.split()

    pass_percentage = correct / total
    error_percentage = errors / total
    print('mean % passed: ', np.mean(pass_percentage) * 100, '%')
    print('std % passed:  ', np.std(pass_percentage))
    print('mean % errored:', np.mean(error_percentage) * 100, '%')
    print('std % errored: ', np.std(error_percentage))
