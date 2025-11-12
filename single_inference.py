#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import subprocess
import sys
import time
import tempfile

TIMEOUT = 3  # seconds

def clean_code_str(code):
    lines = code.splitlines()
    lines = lines[1:]
    cutoff = next((i for i, line in enumerate(lines) if '```' in line), None)
    if cutoff is not None:
        lines = lines[:cutoff]
    return '\n'.join(lines)

def clean_test_str(test):
    lines = test.splitlines()
    first_line = lines[0]
    lines = lines[1:]
    result = []
    for line in lines:
        line = line.lstrip()
        if line.startswith('assert'):
            line = line[len('assert'):].lstrip()
        idx = line.find('==')
        gold = ''
        if idx != -1:
            gold = line[idx+3:].rstrip()
            line = '\"' + line[:idx].rstrip() + '\",'
        if line:
            result.append('        (' + line + ' ' + gold + '),')
    output = f"""{first_line}
    test_cases = (
{'\n'.join(result)}
    )
    correct = 0
    error = 0
    for this_call, gold in test_cases:
        try:
            res = exec(this_call)
            correct += int(res == gold)
        except:
            error += 1
    print('stdout=', correct, error, len(test_cases), flush=True)
"""
    return output

def evaluate_problem(model, tokenizer, problem):
    times = dict()

    messages = [
        {
            'role': 'user',
            'content': problem['query']
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    start = time.time()
    output = model.generate(**inputs, max_new_tokens=1024)
    times['generate'] = time.time() - start

    code = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:])
    code = clean_code_str(code)

    stdout = None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
        f.write(code)
        f.write('\n\n')
        f.write(clean_test_str(problem['test']))
        f.write('\n\nif __name__ == \'__main__\':')
        f.write('\tcheck(' + problem['entry_point'] + ')\n')
        f.flush()

        start = time.time()
        result = subprocess.run(
            [sys.executable, f.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        times['evaluate'] = time.time() - start
        stdout = result.stdout

    return output, inputs, times, stdout

if __name__ == '__main__':
    print('loading dataset...')

    dataset = load_dataset('newfacade/LeetCodeDataset')

    print('finished loading dataset')
    print('loading model...')

    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")
    model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-350M")
    model = model.to('cuda')

    print('loaded model')
    print('generating output...')

    idx = random.randint(0, len(dataset['train']) - 1)
    problem = dataset['train'][idx]
    model_out, _, times, stdout = evaluate_problem(model, tokenizer, problem)
    stdout = [int(x) for x in stdout.split()[1:]]

    print('testing train at idx', idx)
    print('evaluated problem', problem['difficulty'], problem['task_id'], problem['question_id'])
    print('finished generation in', times['generate'], 'seconds, eval took', times['evaluate'], 'seconds')
    print(f'correct = {stdout[0]}/{stdout[2]}, raised errors = {stdout[1]}')
