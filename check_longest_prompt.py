#!/usr/bin/env python3

# THIS IS A TESTING SCRIPT
# iterate over the dataset to find the max token length of input
# this is used for setting params for evaluation

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from matplotlib import pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")
dataset = load_dataset('newfacade/LeetCodeDataset')
combined_dataset = concatenate_datasets([dataset['train'], dataset['test']])

prompt_lengths = [len(tokenizer.encode(problem['query'])) for problem in combined_dataset]
print("Max prompt length:", max(prompt_lengths))

plt.hist(prompt_lengths, bins=50, edgecolor='black')
plt.xlabel('Prompt Length (tokens)')
plt.ylabel('Frequency')
plt.title('Distribution of Prompt Lengths')
plt.show()
