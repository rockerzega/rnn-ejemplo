#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 18:12:06 2023

@author: rockerzega
"""

from funciones import generador, plot_series, RSME, fit, predict
from clases import STData, MLP
from torch.utils.data import DataLoader

n_steps = 50
series = generador(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
X_train.shape, y_train.shape


plot_series(X_test, y_test)

# Determinamos el Error cuadratico medio entre el valor real y el predicho 
print (RSME(X_test, y_test))

y_pred = X_test[:,-1]
# Mostramos las graficas y denotamos el valor predicho y el real en cada epoca
plot_series(X_test, y_test, y_pred)

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

# Creamos un perceptron multicapa para determinar su poder de prediccion
# Multi Layer Perceptron MLP
mlp = MLP()
print(mlp)

fit(mlp, dataloader)

y_pred = predict(mlp, dataloader['test'])
plot_series(X_test, y_test, y_pred.cpu().numpy())

# Prediccion de la MLP
print(RSME(y_test, y_pred.cpu()))

# Valores del perceptron multicapa
print(mlp.fc.weight.shape, mlp.fc.bias.shape)
