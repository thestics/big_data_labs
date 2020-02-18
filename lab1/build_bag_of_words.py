#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko

import typing
from collections import Counter

import numpy as np


def clear(data: str):
    punctuation = '.,?!;:-\n\t'
    for sign in punctuation:
        data = data.replace(sign, '')
    return data


def build_mapping(src: str) -> dict:
    with open(src) as f:
        data = f.read()
    data = clear(data)
    words = set(data.split())
    word2index = {char: num for num, char in enumerate(words)}
    return word2index


def map_word(mapping: typing.Dict[str, int], word:str) -> np.ndarray:
    size = len(mapping)
    return np.array([0 if i != mapping[word] else 1 for i in range(size)])


def map(mapping: typing.Dict[str, int],  targets: typing.List[str]):
    res = []
    for sentence in targets:
        sentence = clear(sentence)
        cur = np.zeros(len(mapping))
        for word in sentence.split():
            cur += map_word(mapping, word)
        res += cur
    return res
