#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 18:55:20 2023

@author: rockerzega
"""

import torch
from torch.utils.data import Dataset

class STData(Dataset):
  def __init__(self, X, y=None, train=True):
    self.X = X
    self.y = y
    self.train = train

  def __len__(self):
    return len(self.X)

  def __getitem__(self, ix):
    if self.train:
      return torch.from_numpy(self.X[ix]), torch.from_numpy(self.y[ix])
    return torch.from_numpy(self.X[ix])


class MLP(torch.nn.Module):
  def __init__(self, n_in=50, n_out=1):
    super().__init__()
    self.fc = torch.nn.Linear(n_in, n_out)

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x

class SimpleRNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = torch.nn.RNN(input_size=1, hidden_size=1, num_layers=1, batch_first=True)

  def forward(self, x):
    x, h = self.rnn(x) 
    # solo queremos la última salida
    return x[:,-1]

"""
En la RNN tenemos una entrada conecta a una neurona en la capa oculta (que al tener solo una capa oculta también es la salida), un peso más 
el bias, 2. Además, tenemos una conexión de la neurona en la capa oculta en un instante a la capa oculta en el instante siguiente, otro peso 
más el bias, 2 más. En total 4 parámetros contra los 51 del MLP. Necesitamos una RNN con más capacidad, por ejemplo aumentando el 
número de neuronas en la capa oculta. Para este caso, además, necesitaremos una capa lineal que nos de el último valor a partir de los 20 
valores en la capa oculta.
"""

class RNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = torch.nn.RNN(input_size=1, hidden_size=20, num_layers=1, batch_first=True)
    self.fc = torch.nn.Linear(20, 1)

  def forward(self, x):
    x, h = self.rnn(x) 
    # get the last output and apply linear layer
    y = self.fc(x[:,-1])
    return y
