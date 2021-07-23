import tensorflow as tf
import keras
from keras import Sequential
from tensorflow.keras.utils import Sequence
from keras.layers import LSTM, Dense, Masking
from keras.utils import np_utils
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
import datetime
from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras.preprocessing.sequence import  pad_sequences
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import fnmatch
import os
import random
import re
import threading
import librosa,librosa.display
import tensorflow as tf
from six.moves import xrange
import time
import json
import tqdm
import soundfile
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import sleep
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
import functools
import sentencepiece
import torch 
import torch as t
import argparse
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

MAX_LEN = 512
load_model = True

np.random.seed(1337)

parser = argparse.ArgumentParser(description='Headline Generation MT5')
parser.add_argument('--filename', type=str, default='', help='Enter text file path to be classified into tech/non-tech')
parser.add_argument('--sen', type=str, default='', help='')
parser.add_argument('--num_sentences', type=int, default=1, help='Enter number of headlines you wish to generate')

args = parser.parse_args()
# print(args.filename)
# print(args.sen)

if args.filename != '':
  txt_file = args.filename
  with open(txt_file, 'r') as files:
    article = files.read()
elif args.sen != '':
  article = args.sen
else:
  print('Please enter a string in --file or --sen')
  exit()

num_sentences = args.num_sentences

#Loading pretrained XLMR
device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')

checkpoint_path = 'Copy of checkpoint_latest.pt'

if(load_model):
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])

model.cuda()

text =  "English Headline: " + article

encoding = tokenizer.encode_plus(text, return_tensors = "pt")
input_ids = encoding["input_ids"].to(device)
attention_masks = encoding["attention_mask"].to(device)

beam_outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True,
    max_length=64,
    top_k=150,
    top_p=0.95,
    early_stopping=True,
    num_return_sequences=num_sentences
)

final_outputs =[]
for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    # if sent.lower() != sentence.lower() and sent not in final_outputs:
    final_outputs.append(sent)

print(f"Generated Headlines : ")

for i, final_output in enumerate(final_outputs):
    print("{}: {}".format(i + 1, final_output))




