#!/usr/bin/env python
# -*- coding: utf-8 -*-

DATA_PATH = '../../data/poetry.txt'
# DATA_PATH = './temp/input.txt'

import numpy as np

THRESHOLD_FREQ = 2

# read the data
poems = []
with open(DATA_PATH) as fp:
    for line in fp.readlines():
        poem = line.strip().split(":")[-1]
        poem = poem.replace(' ', '')
        if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '{' in poem:
            continue
        if len(poem) < 10 or len(poem) > 128:
            continue
        poem = '[' + poem + ']'
        poems.append(poem)

# data preprocess
all_words = {}
for poem in poems:
    for word in poem:
        if word not in all_words:
            all_words[word] = 1
        else:
            all_words[word] += 1
            
# discard the words with very low frequence to be used
erase = []
for key in all_words:
    if all_words[key] < THRESHOLD_FREQ:
        erase.append(key)

for key in erase:
    del all_words[key]

# sort the words
all_words = sorted(all_words.items(), key=lambda x: -x[1])
print(all_words)

data = open(DATA_PATH, 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

print('data has %d characters, %d unique.' % (data_size, vocab_size))
# print(poems[0:2])
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}
words_to_ix = {ch:i for i, ch in enumerate(all_words)}
ix_to_words = {i:ch for i, ch in enumerate(all_words)}
# hyper-parameters

def build_model():
    pass

def infer():
    pass

# test running
