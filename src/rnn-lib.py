#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 19:59:15 2023

@author: rockerzega
"""

from clases import SimpleRNN, STData, RNN
from torch.utils.data import DataLoader
from funciones import fit, generador, RSME, predict, plot_series

# preparacion de la data simulada
n_steps = 50
series = generador(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

# Infomracion de la data
print(X_train.shape, y_train.shape)

# y_pred = X_test[:,-1]

dataset = {
  'train': STData(X_train, y_train),
  'eval': STData(X_valid, y_valid),
  'test': STData(X_test, y_test, train=False)
}

dataloader = {
  'train': DataLoader(dataset['train'], shuffle=True, batch_size=64),
  'eval': DataLoader(dataset['eval'], shuffle=False, batch_size=64),
  'test': DataLoader(dataset['test'], shuffle=False, batch_size=64)
}

rnn = SimpleRNN()

fit(rnn, dataloader)

y_pred = predict(rnn, dataloader['test'])
plot_series(X_test, y_test, y_pred.cpu().numpy())
print(RSME(y_test, y_pred.cpu()))
# Parametros de la RNN Simple
print(rnn.rnn.weight_hh_l0.shape, 
      rnn.rnn.weight_ih_l0.shape, 
      rnn.rnn.bias_hh_l0.shape, 
      rnn.rnn.bias_ih_l0.shape)

rnn = RNN()
# Parametros de la RNN completa
print(rnn.rnn.weight_hh_l0.shape, 
      rnn.rnn.weight_ih_l0.shape, 
      rnn.rnn.bias_hh_l0.shape, 
      rnn.rnn.bias_ih_l0.shape, 
      rnn.fc.weight.shape, 
      rnn.fc.bias.shape)
fit(rnn, dataloader)
print(RSME(y_test, y_pred.cpu()))