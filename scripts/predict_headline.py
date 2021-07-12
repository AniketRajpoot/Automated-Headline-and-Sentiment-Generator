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

MAX_LEN = 512

np.random.seed(1337)

parser = argparse.ArgumentParser(description='LUS keypoint network pytorch-lightning parallel')
parser.add_argument('--filename', type=str, default='', help='Enter text file path to be classified into tech/non-tech')
parser.add_argument('--sen', type=str, default='', help='')

args = parser.parse_args()
# print(args.filename)
# print(args.sen)

if args.filename != '':
  txt_file = args.filename
  with open(txt_file, 'r') as files:
    text = files.read()
elif args.sen != '':
  text = args.sen
else:
  print('Please enter a string in --file or --sen')
  exit()

#Loading pretrained XLMR
device = t.device("cuda" if t.cuda.is_available() else "cpu")
from transformers import XLMRobertaForSequenceClassification, AdamW, XLMRobertaConfig, XLMRobertaTokenizer

model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
)

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

model.cuda()
model.load_state_dict(torch.load('article(chunked)+tweets_model_8epochs.bin')) 
#Our pretrained model downloaded at top level

#Processing text
tokens = tokenizer.tokenize(text)
encoded_sent = tokenizer.encode(
  tokens,
  # max_length = 512,
  # return_tensors = 'pt'   
  )


#Padding the input sequence to be of length 512 
'''
this is implemented to handle the cases with tokens in the sentence less that 512, for example, a sentence might be short, 
having only 200 tokens (for instance)(note that the words in the sentence are >200)

the transformer in usage takes token length of exactly 512, hence, the remaining 512-200=312 tokens would be filled with '0' value,
as seen below in the attribute 'value=0'
'''

input_ids = pad_sequences([encoded_sent], maxlen = MAX_LEN, dtype = "long", value=0, 
  truncating="post", padding="post")
                        
attention_masks = []

for sent in input_ids:
    # set mask to 0 if token_id is 0 (becoz its padding) and vice versa
    attn_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(attn_mask)

input_ids = t.Tensor(input_ids).long().to(device)  
attention_mask = t.Tensor(attention_masks).long().to(device)  # The input masks for padding

#labels = batch[2].to(device)        # The labels 

with t.no_grad():
  outputs = model(input_ids, attention_mask=attention_mask)

logits = outputs[0]
logits = logits.detach().cpu().numpy()
#label_ids = labels.to('cpu').numpy()
pred_flat = np.argmax(logits, axis = 1).flatten()
#labels_flat = label_ids.flatten()

#y_true.append(labels_flat[0])
# print(pred_flat,labels_flat)
if pred_flat == 0:
  print('Final prediction: Non-Mobile Tech')
else:
  print('Final prediction: Mobile Tech')





