#!/usr/bin/env python3

# THIS IS A TESTING SCRIPT
# which loads the Liquid SLM, and iterates over the whole leetcode dataset
# to get statistics on success rate, runtime, syntax/runtime/TLE errors

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from single_inference import clean_code_str, clean_test_str, TIMEOUT
from tqdm import tqdm
import numpy as np
import torch
import random
import tempfile
import time
import subprocess
import sys

def evaluate_problem_eval(problem, code):
    times = dict()
    contents = code + '\n\n'
    cleaned_test, num_tests = clean_test_str(problem['test'])
    contents += cleaned_test
    contents += '\n\nif __name__ == \'__main__\': check(' + problem['entry_point'] + ')'

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
        f.write(contents)
        f.flush()

        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=TIMEOUT
            )
            times['evaluate'] = time.time() - start
            stdout = result.stdout
        except subprocess.TimeoutExpired as e:
            times['evaluate'] = time.time() - start
            stdout = num_tests

    if stdout == '':  # SLM generated code with syntax error
        stdout = num_tests
    return contents, times, stdout

if __name__ == '__main__':
    print('loading dataset...')

    dataset = load_dataset('newfacade/LeetCodeDataset')

    print('finished loading dataset')
    print('loading model...')

    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-350M").to('cuda')
    model.eval()

    print('loaded model')
    print('iterating through data...')

    # combined_dataset = concatenate_datasets([dataset['train'], dataset['test']])
    dataset = dataset['test']

    n = len(combined_dataset)
    correct = np.zeros((n), dtype=int)
    errors = np.zeros((n), dtype=int)
    timelimit = np.zeros((n), dtype=int)
    total = np.zeros((n), dtype=int)

    batch_size = 64  # TODO: may need to tweak this, fully utilize local GPU

    ####  batch_size=1 implementation below (slow)  ####
    # for i, problem in tqdm(enumerate(combined_dataset), total=n):
    #     _, _, times, stdout = evaluate_problem(model, tokenizer, problem)
    #     if times['evaluate'] >= 10: timelimit[i] = total[i] = stdout  # TLE
    #     elif type(stdout) == int: errors[i] = total[i] = stdout  # syntax error
    #     else: correct[i], errors[i], total[i] = [int(x) for x in stdout.split()]

    ####  larger batch_size for better GPU utilization  ####
    for i in tqdm(range(0, n, batch_size), total=(n + batch_size - 1) // batch_size):
        batch = combined_dataset.select(range(i, min(i+batch_size, n))).to_list()
        prompts = [problem['query'] for problem in batch]

        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')
        with torch.no_grad(): outputs = model.generate(**inputs, max_new_tokens=1024)

        for j, (problem, output) in enumerate(zip(batch, outputs)):
            code = tokenizer.decode(output[inputs['input_ids'][j].shape[-1]:], skip_special_tokens=True)
            code = clean_code_str(code)
            _, times, stdout = evaluate_problem_eval(problem, code)

            idx = i + j
            if times['evaluate'] >= 10: timelimit[idx] = total[idx] = stdout  # TLE
            elif type(stdout) == int: errors[idx] = total[idx] = stdout  # syntax error
            else: correct[idx], errors[idx], total[idx] = [int(x) for x in stdout.split()]

    pass_percentage = correct / total
    error_percentage = errors / total
    problems_lost_to_TLE = np.sum(timelimit != 0)
    print('mean % passed: ', np.mean(pass_percentage) * 100, '%')
    print('std % passed:  ', np.std(pass_percentage))
    print('mean % errored:', np.mean(error_percentage) * 100, '%')
    print('std % errored: ', np.std(error_percentage))
    print('problems lost to TLE: ', problems_lost_to_TLE)
