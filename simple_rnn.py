#!/usr/bin/env python
# -*- coding: utf-8 -*-

DATA_PATH = '../../data/novel.txt'
# DATA_PATH = './temp/input.txt'

import numpy as np

# read the data
data = open(DATA_PATH, 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

print('data has %d characters, %d unique.' % (data_size, vocab_size))

char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# model hyperparameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01
Whh = np.random.randn(hidden_size, hidden_size)*0.01
Why = np.random.randn(vocab_size, hidden_size)*0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

def loss_rnn(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
        ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
        loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = [seed_ix]
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
if seq_length > len(data):
    seq_length = len(data) - 1
smooth_loss = -np.log(1.0/vocab_size)*seq_length
while True:
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))
        p = 0
    inputs = [char_to_ix[ch] for ch in data[p : p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1 : p+seq_length+1]]

    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('--------\n %s \n--------' % (txt, ))
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_rnn(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))

    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                    [dWxh, dWhh, dWhy, dbh, dby],
                                    [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam*dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1

# with open(DATA_PATH, 'r') as f:
#     counter = 0
#     seg = f.readline()
#
# import jieba
#
# par = jieba.cut(seg)
#
# res = dict()
#
# for word in par:
#     word = word.strip()
#     if len(word) > 0:
#         if word in res.keys():
#             res[word] += 1
#         else:
#             res[word] = 1
#
# import operator
#
# sorted_res = sorted(res.items(), key=operator.itemgetter(1), reverse=True)
# print(len(sorted_res))
# print(sorted_res[0:10])
#
# erase = []
# for key in res:
#     if res[key] < 2:
#         erase.append(key)
#
# for key in erase:
#     del res[key]
