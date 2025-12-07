#!/usr/bin/env python3

# this script runs GRPO on the Liquid SLM, the reward model depends on args

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from single_inference import clean_code_str
from multi_inference import evaluate_problem_eval
import torch
import torch.nn.functional as F
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logits = logits[:, -logits_to_keep:, :]
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, -1, input_ids[:, -logits_to_keep:].unsqueeze(-1)).squeeze(-1)
    return per_token_logps

def run_grpo(model, dataset, tokenizer, group_size=4, kl_beta=0.1, eps=0.2, num_epochs=1, batch_size=8):
    model.train()
    ref_model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-350M").to('cuda')
    ref_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset['train']
    n = len(dataset)

    for epoch in range(num_epochs):
        for i in tqdm(range(0, n, batch_size), total=(n + batch_size - 1) // batch_size):
            batch = dataset.select(range(i, min(i+batch_size, n))).to_list()

            prompts = [problem['query'] for problem in batch for _ in range(group_size)]
            inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')
            with torch.no_grad(): outputs = model.generate(**inputs, max_new_tokens=1024)

            generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
            completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            eval_args = []
            for idx, code in enumerate(completions):
                problem = batch[idx // group_size]
                eval_args.append((problem, clean_code_str(code)))

            rewards = torch.zeros(len(completions), device='cuda')
            final_codes = []

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda p : evaluate_problem_eval(*p), eval_args))

            for idx, (contents, times, stdout) in enumerate(results):
                final_codes.append(eval_args[idx][1])
                if type(stdout) == int: error_rate = 1.0
                else:
                    correct, errors, total = [int(x) for x in stdout.split()]
                    error_rate = errors / total if total > 0 else 1.0
                rewards[idx] = -error_rate

            # training begins here
            train_inputs = tokenizer(final_codes, return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')
            input_ids = train_inputs['input_ids']
            attention_mask = train_inputs['attention_mask']
            logits_to_keep = input_ids.shape[1] - 1

            per_token_logps = get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
            with torch.no_grad(): ref_per_token_logps = get_per_token_logps(ref_model, input_ids, attention_mask, logits_to_keep)

            kl_div = per_token_logps - ref_per_token_logps

            # 1. FIX ADVANTAGE SHAPE
            rewards = rewards.view(-1, group_size)
            reward_mean = rewards.mean(dim=1, keepdim=True)
            reward_std = rewards.std(dim=1, keepdim=True)
            advantage = (rewards - reward_mean) / (reward_std + 1e-8)
            advantage = advantage.view(-1) # Flatten back to [total_sequences]

            policy_ratio = torch.exp(per_token_logps - per_token_logps.detach())
            clip_policy_ratio = torch.clamp(policy_ratio, min=1.0 - eps, max=1.0 + eps)

            advantage_expanded = advantage.unsqueeze(1)

            loss = torch.min(advantage_expanded * policy_ratio, advantage_expanded * clip_policy_ratio)
            loss = -loss + kl_beta * kl_div

            # Mask out padding tokens so they don't affect the loss
            loss_mask = attention_mask[:, -logits_to_keep:]
            loss = (loss * loss_mask).sum() / loss_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


if __name__ == '__main__':
    if os.path.exists('output_model_weights.pth'):
        print('output_model_weights.pth is already on this filesystem! please rename the file')
        exit(1)

    print('loading dataset...')

    dataset = load_dataset('newfacade/LeetCodeDataset')

    print('finished loading dataset')
    print('loading model...')

    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-350M").to('cuda')

    print('loaded model')
    print('running grpo...')

    model = run_grpo(model, dataset, tokenizer, batch_size=1)

    print('finished grpo')
    print('saving model weights...')

    torch.save(model.state_dict(), 'output_model_weights.pth')

    print('model saved')
