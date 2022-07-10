from typing import Sequence, Tuple, List, Union

import torch
import numpy as np

RawMSA = Sequence[Tuple[str, str]]


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

class BatchConverter(object):
    """
    Callable to convert an unprocessed (labels + strings) batch to a processed (labels + tensor) batch.
    """
    def __init__(self, alphabet, data_type="seq",):
        """
        :param alphabet:
        :param data_type: seq, msa
        """
        self.alphabet = alphabet
        self.data_type = data_type.split("+")

    def __call__(self, raw_data, raw_anns=None):
        """
        :param raw_data: each element in raw data should contain (description, seq)
        :param raw_anns:
        :return:
        """
        # creat a new batch of data tensors
        data = {}
        for key in raw_data.keys():
            if key == "seq" or key == "msa":
                if key == "seq":
                    labels, strs, tokens = self.__call_seq__(raw_data["seq"])
                else:
                    labels, strs, tokens = self.__call_msa__(raw_data["msa"])

                data["description"] = labels
                data["string"] = strs
                data["token"] = tokens
                if key == "seq":
                    data["depth"] = [1] * len(strs)
                    data["length"] = [len(s) for s in strs]
                elif key == "msa":
                    data["depth"] = []
                    data["length"] = []
                    for m in strs:
                        data["depth"].append(len(m))
                        data["length"].append(len(m[0]))
            else:
                if isinstance(raw_data[key][0], str):
                    data[key] = raw_data[key]
                elif isinstance(raw_data[key][0], np.ndarray):
                    try:   # same length
                        data[key] = torch.Tensor(raw_data[key])
                    except:
                        # here we padding them with 0 for consistance with cnn's padding, which is different with ann's padding
                        data[key] = torch.Tensor(self.__padding_numpy_matrix__(raw_data[key], data["length"], pad_idx=0))
                elif isinstance(raw_data[key][0], float) or isinstance(raw_data[key][0], int):
                    data[key] = torch.Tensor(raw_data[key])


        # creat a new batch of ann tensors
        if raw_anns is not None:
            anns = {}
            for key in raw_anns.keys():
                if isinstance(raw_anns[key][0], str):
                    anns[key] = raw_anns[key]
                elif isinstance(raw_anns[key][0], np.ndarray):
                    try:   # same length
                        anns[key] = torch.Tensor(raw_anns[key])
                    except:
                        anns[key] = torch.Tensor(self.__padding_numpy_matrix__(raw_anns[key], data["length"]))
                elif isinstance(raw_anns[key][0], float) or isinstance(raw_anns[key][0], int):
                    anns[key] = torch.Tensor(raw_anns[key])
        else:
            anns = None

        return data, anns

    def __call_seq__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        max_len = max(len(seq_str) for _, seq_str in raw_batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str) in enumerate(raw_batch):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(
                [self.alphabet.get_idx(s) for s in seq_str], dtype=torch.int64
            )
            tokens[
            i,
            int(self.alphabet.prepend_bos): len(seq_str)
                                            + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[
                    i, len(seq_str) + int(self.alphabet.prepend_bos)
                ] = self.alphabet.eos_idx

        return labels, strs, tokens

    def __call_msa__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens = self.__call_seq__(msa)
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens

        return labels, strs, tokens

    def __padding_numpy_matrix__(self, inputs, lengths, pad_idx=-1):
        """
        :param inputs: list (B) of numpy (L*L)  must be digital
        :param lengths: list of different length
        :return: tensor with shape (B * T * T)
        default right padding
        * should watch out for there is the meaning of -1 in some cases
        """
        max_length = max(lengths)
        dim_input = len(inputs[0].shape)
        max_shape = [max_length] * dim_input

        padded_inputs = []
        for input in inputs:
            pad_shape = []
            for i in range(dim_input):
                pad_shape.append((0, max_shape[i]-input.shape[i]))
            padded_inputs.append(np.pad(input, pad_shape, constant_values=pad_idx))
        padded_inputs = np.stack(padded_inputs, axis=0)

        return padded_inputs









