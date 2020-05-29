#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko

import pickle
import logging as log

import numpy as np

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist, cifar10

from utils import build_reducer, plot_reducer, plot_reducer_x_y, DATASAET_PART_SIZE


log.basicConfig(format="[%(asctime)s] - %(message)s", level=log.DEBUG)


mnist_args_raw = {
    'dataset': mnist,
    'row_size': 28 * 28,
    'dump_file_path': 'embeddings/{}/reducer_mnist.pkl'
}


cifar_args_raw = {
    'dataset': cifar10,
    'row_size': 32 * 32 * 3,
    'dump_file_path': 'embeddings/{}/reducer_cifar.pkl'
}


def get_cifar_args(reducer):
    args = cifar_args_raw.copy()
    args['reducer'] = reducer
    args['dump_file_path'] = args['dump_file_path'].format(
        reducer.__class__.__name__
    )
    return args


def get_mnist_args(reducer):
    args = mnist_args_raw.copy()
    args['reducer'] = reducer
    args['dump_file_path'] = args['dump_file_path'].format(
        reducer.__class__.__name__
    )
    return args


def build_reducer_for_datasets(reducer):
    log.info(f'Building {reducer.__class__.__name__} MNIST reducer')
    build_reducer(**get_mnist_args(reducer))

    log.info(f'Building {reducer.__class__.__name__} CIFAR10 reducer')
    build_reducer(**get_cifar_args(reducer))


def build_umap():
    reducer = UMAP(random_state=42)
    build_reducer_for_datasets(reducer)


def build_pca():
    reducer = PCA(n_components=2)
    build_reducer_for_datasets(reducer)


def build_tsne():
    reducer = TSNE(n_components=2)
    build_reducer_for_datasets(reducer)


def plot_tsne():
    with open('embeddings/TSNE/reducer_cifar.pkl', 'rb') as f:
        reducer = pickle.load(f)
    data = reducer.embedding_
    (_, y_train), _ = cifar10.load_data()
    plot_reducer_x_y(data, y_train[:DATASAET_PART_SIZE], 'TSNE CIFAR')

    with open('embeddings/TSNE/reducer_mnist.pkl', 'rb') as f:
        reducer = pickle.load(f)
    data = reducer.embedding_
    (_, y_train), _ = mnist.load_data()
    plot_reducer_x_y(data, y_train[:DATASAET_PART_SIZE], 'TSNE MNIST')


def plot_umap():
    args = cifar_args_raw.copy()
    args['title'] = 'UMAP CIFAR'
    args['dump_file_path'] = 'embeddings/UMAP/reducer_cifar.pkl'
    plot_reducer(**args)

    args = mnist_args_raw.copy()
    args['title'] = 'UMAP MNIST'
    args['dump_file_path'] = 'embeddings/UMAP/reducer_mnist.pkl'
    plot_reducer(**args)


def plot_pca():
    args = cifar_args_raw.copy()
    args['title'] = 'PCA CIFAR'
    args['dump_file_path'] = 'embeddings/PCA/reducer_cifar.pkl'
    plot_reducer(**args)

    args = mnist_args_raw.copy()
    args['title'] = 'PCA MNIST'
    args['dump_file_path'] = 'embeddings/PCA/reducer_mnist.pkl'
    plot_reducer(**args)


if __name__ == '__main__':
    # build_umap()
    # build_tsne()
    # build_pca()
    plot_umap()
    plot_pca()
    plot_tsne()