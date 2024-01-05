
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from transformers import Wav2Vec2Processor, WavLMModel, HubertModel, Wav2Vec2Model
# from datasets import load_dataset
# import librosa
# import warnings
# import torch
from transformers import Wav2Vec2Processor
# import torch
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process
# from .position_encoding import build_position_encoding

# from pytorch_pretrained_bert.modeling import BertModel


class WAVLM(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        # if name == 'bert-base-uncased':
        self.num_channels = 768
        # else:
        #     self.num_channels = 1024
        # self.enc_num = enc_num

        # self.bert = BertModel.from_pretrained(name)
        self.wavlm = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")

        if not train_bert:
            for parameter in self.wavlm.parameters():
                parameter.requires_grad_(False)

    def forward(self, audio_data):

        # remove this block of code for providing the embedding of any layer
        # currently lets hardcode it for last layer only
        # if self.enc_num > 0:
        #     all_encoder_layers, _ = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
        #     # use the output of the X-th transformer encoder layers
        #     xs = all_encoder_layers[self.enc_num - 1]
        # else:
        #     xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        wlm_op = self.wavlm(audio_data)
        
        # currently not tracking masks
        # mask = tensor_list.mask.to(torch.bool)
        # mask = ~mask
        # out = NestedTensor(xs, mask)

        return wlm_op

def build_wavlm(args):
    # position_embedding = build_position_encoding(args)
    train_bert = args.lr_bert > 0
    bert = WAVLM(args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num)
    # model = Joiner(bert, position_embedding)
    # model.num_channels = bert.num_channels
    return bert
