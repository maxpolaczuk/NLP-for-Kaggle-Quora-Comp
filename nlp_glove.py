'''
KAGGLE COMPETITION
QUORA QUESTION DUPLICATION
'''

import os
import numpy as np
import pandas as pd
import nltk

# change dir:
os.chdir('/home/max/Downloads/glove.twitter.27B')

# Read in txt file of pretrained word vectors using GloVe on
# 27Billion tweets
# More casual vocabulary size of 1.2M words

fname = 'glove.twitter.27B.100d.txt' # 100dimensional takes about 30s

with open(fname) as f:
    content = f.readlines()
# remove whitespace characters:
content = [x.strip() for x in content] 

# make a list of words only:
words = [x.split()[0] for x in content]

# create a lookup table from the list of words:
words = {k: str(v).lower() for v, k in enumerate(words)}

# make function to locate each word idx in a sentence from glove vector:

from keras.preprocessing.text import text_to_word_sequence

def find_glove(sentence):
    ''' take in the sentece of words, then try and locate their index'''
    # convert sentect to list of words:
    sentence = text_to_word_sequence(sentence)
    
    # set up empty array for GloVe indexs
    idxs = []
    for w in sentence:
        # go over each word and add its index:
        idxs.append(words[w])
        
    
    return idxs,sentence
        
# make function to get glove vectors at
def glove_seq(sequen):
    seq = [] # empty list for us to create the sequence
    for i in sequen:
        # i is the ID number of the word
        # iterate over the list &
        # create vector from glove id number
        seq.append([float(x) for x in content[int(i)].split()[1:]])
        
    return seq
# mess with some sentences / Questions from quora:

q1 = "How can I increase the speed of my internet connection while using a VPN?"
q2 = "How can Internet speed be increased by hacking through DNS?"
qs = [q1,q2]

# Keras imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

'''
Preprocess the text and get a padded sequence of glove inputs 
for each question...
'''
# get each tokenized sentence and their glove idxs:
questions = []
glov_qs = []
seq = []
for i in qs:
    tmp1,tmp2 = find_glove(i)
    questions.append(tmp2)
    # check that there is the word in the corpus 
    if True == True: # checking statement goes here
        # make the sequences:
        glov_qs.append(tmp1)
        # convert our integer seq, to seq of glove vectors:
        seq.append(glove_seq(tmp1))
        
    else:
        # if it's not in corpus, then we need to 
        # model the closest word by characters / context 
        # do this at a later date
        continue
    
print(questions)

# Pad out the sequences...
data = pad_sequences(seq, maxlen=20)
print(np.shape(data))

