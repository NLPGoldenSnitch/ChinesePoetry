# python
# -*- coding: utf-8 -*-
# file: data.py
# author: Jie Li
# ------------------------------------------------------------------------
import collections
import os
import sys
import numpy as np

from data import DataHandle
from model import RNNModel
from interface import Interface


class ControlParm(object):
    batch_size = 32
    n_epoch = 100
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.9
    grad_clip = 5

    cell_size = 128
    num_layers = 3
    seq_length = 20
    log_dir = './logs'
    metadata = 'metadata.tsv'
    gen_num = 32 # how many chars to generate

def run(isGen,topic=u"静夜思:"):
    file_name = "poem.txt"
    args = ControlParm()
    data = DataHandle(file_name, args)
    interface = Interface()
    model = RNNModel(args, data, isGen=isGen)
    if isGen == 0:
        print("[INFO] start trainning")
        interface.train(data,model,args)
    else:
        interface.generate(data,model,args,topic)


if __name__ =="__main__":
    msg = """
    Usage:
    Training: 
        python3 gen_lyrics.py 0
    Sampling:
        python3 gen_lyrics.py 1
    """
    if len(sys.argv) == 2:
        isGen = int(sys.argv[-1])
        print('--Sampling--' if isGen else '--Training--')
        run(isGen)
    else:
        print(msg)
        sys.exit(1)
