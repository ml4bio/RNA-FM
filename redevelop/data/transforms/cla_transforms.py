# encoding: utf-8
"""
@author:  chenjiayang
@contact: chenjiayang@163.com
"""

import sys
import collections
import random
import torch
import numpy as np


if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, sequence):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return torch.Tensor(sequence)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Encode(object):
    """
    "DNA Sequence Classification by Convolutional Neural Network"
    Article in Journal of Biomedical Science and Engineering · January 2016
    DOI: 10.4236/jbise.2016.95021
    """
    def __init__(self):
        self.element_vector_length = 4   # one-hot A U G C
        self.word_length = 3             # n elements -> a word
        self.region_size = 4             # n word -> a column

        # map word into a scalar
        temp1 = np.array(range(self.element_vector_length)).reshape((self.element_vector_length,1))
        temp2 = np.array([np.power(self.element_vector_length, i) for i in range(self.word_length)]).reshape((1, self.word_length))

        self.word_encode_table = np.dot(temp1, temp2)

        self.vector_encode_table = np.eye(64)

    def __call__(self, sequence):
        ori_word_list = []
        # print(sequence.shape)
        num_word = sequence.shape[-1] // self.word_length
        # print("num_words:{}".format(num_word))
        for i in range(num_word):
            word_start = i * self.word_length
            word_end = word_start + self.word_length
            word = sequence[:, word_start:word_end]
            scalar = (self.word_encode_table * word).sum()   # (word_length * word_vector) X (word_vector * sequence_length)
            vector = self.vector_encode_table[scalar]
            ori_word_list.append(vector)

        new_word_list = []
        for j in range(self.region_size):
            new_word_list.append(np.array(ori_word_list[j:len(ori_word_list) - self.region_size + 1 + j]).T)
        sequence = np.concatenate(new_word_list, axis=0)
        # print(sequence.shape)
        sequence = np.expand_dims(sequence,0)  # the channel of 'image'

        return sequence


class RandomCrop(object):
    """
        input: numpy
    """
    def __init__(self, crop_length):
        self.crop_length = crop_length #2000  # 777 ——> encode 256*256

    def __call__(self, sequence):
        length = sequence.shape[-1]
        if self.crop_length > 0:
            if self.crop_length < length:
                start = random.randint(0, length - self.crop_length)
            elif self.crop_length == length:
                start = 0
            else:
                raise Exception("Crop Length is greater than Sequence Length!")

            sequence = sequence[:, start:start + self.crop_length]

        return sequence


class ClampLength(object):
    """
        input: numpy  V * L
    """
    def __init__(self, min=0, max=-1):
        self.min = abs(min)   # abs for ensemble prediction
        self.max = abs(max)
        if self.max < self.min and self.min > 0 and self.max > 0:
            raise Exception("Set Max less than Min! for Length")

    def __call__(self, sequence):
        length = sequence.shape[-1]
        if self.min > length:
            diff = self.min - length
            sequence = np.pad(sequence, ((0,0), (0, diff)), "constant")
        elif self.max < length:
            start = random.randint(0, length - self.max)
            sequence = sequence[:, start:start + self.max]
        else:
            pass
        return sequence
