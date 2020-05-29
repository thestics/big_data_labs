#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


DATASAET_PART_SIZE = 2_000


def build_reducer(reducer, dump_file_path, dataset, row_size):
    (x_train, y_train), (_, _) = dataset.load_data()
    x_train = x_train[:DATASAET_PART_SIZE].reshape((-1, row_size))
    reducer.fit(x_train)

    dir_name, _ = os.path.split(dump_file_path)
    os.makedirs(dir_name, exist_ok=True)

    with open(dump_file_path, 'wb') as f:
        pickle.dump(reducer, f)


def plot_reducer_x_y(X, Y, title):
    plt.scatter(X[:, 0], X[:, 1], c=Y[:DATASAET_PART_SIZE], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title(title)
    plt.show()


def plot_reducer(dump_file_path, dataset, row_size, title):
    with open(dump_file_path, 'rb') as f:
        reducer = pickle.load(f)

    (x_train, y_train), (_, _) = dataset.load_data()
    x_train = x_train[:DATASAET_PART_SIZE].reshape(-1, row_size)
    embedding = reducer.transform(x_train)
    plot_reducer_x_y(embedding, y_train, title)
