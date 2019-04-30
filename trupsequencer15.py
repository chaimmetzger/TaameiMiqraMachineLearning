#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
import random
from collections import Counter
import numpy as np
from codecs import open
import time
from dynet import *
from sys import argv
import argparse
import re
import os
import sys
import codecs

random.seed(1847659371)
print('Seed: 1847659371')

#predicting trop in general

LAYERS = 2
INPUT_DIM = 50
HIDDEN_DIM = 50

LEXICAL = True
POS_FOR_PARTIAL_WORD = 0
POS_FOR_FINAL_WORD = 1
POS_AGGREGATE_FOR_WORD = 2

EMET = False

# default is this:
pos_behavior = POS_AGGREGATE_FOR_WORD
argument = '' # type: str
include = [ True, True, True, True, True, True, True, True] # type: List[bool]
features = [ 'Trup', 'POS', 'Lex', 'HalfVerse', 'PType', 'Function', 'CType', 'POS_Sub'] # type: List[str]
strFeatures = '' # type: str

for argument in sys.argv[1:]:
    if argument == '-aggregate':
        pos_behavior = POS_AGGREGATE_FOR_WORD
        strFeatures = 'aggregate'
    elif argument == '-partial':
        pos_behavior = POS_FOR_PARTIAL_WORD
        strFeatures = 'wordbreak'
    elif argument == '-emet':
        EMET = True
    elif argument.startswith('-0') or argument.startswith('-1'):
        toggles = argument[1:] # strip the dash
        include = [ch == '1' for ch in toggles]

if EMET:
    strFeatures += 'EMET'

for b, f in zip(include, features):
    strFeatures += f
    if b:
        strFeatures += '1'
    else:
        strFeatures += '0'


if not os.path.isdir("Resultslog"):
    os.makedirs("Resultslog")
if not os.path.isdir("Resultslog/Resultslog4" + strFeatures):
    os.makedirs("Resultslog/Resultslog4" + strFeatures)
if not os.path.isdir("Modelslogs"):
    os.makedirs("Modelslogs")
if not os.path.isdir("Modelslogs/Modelslog4" + strFeatures):
    os.makedirs("Modelslogs/Modelslog4" + strFeatures)

def logmessage(s):
    with open("Resultslog/Resultslog4" + strFeatures + '/Resultslogmessage' + strFeatures + ".txt", "a") as w:
        w.write(str(s))
    print(s)

# def model.save (s):
#   with open("Modelslog/")

def read_data(input_file=""):
    if not input_file:
        if pos_behavior == POS_AGGREGATE_FOR_WORD:
            input_file = 'POSAndTaamPairsForAllOfTanakh17_word_aggregate'
        else:
            input_file = 'POSAndTaamPairsForAllOfTanakh17_wordbreak'

        #if EMET:
        #    input_file += '_EMET'

        input_file += '.txt'
    # open the file
    with codecs.open(input_file, mode='r', encoding='utf-8') as f:
        for line in f:
            # get the label
            label, tags = line.split(':')

            # filter label? - remove emet
            if EMET:
                if label.startswith('Genesis') or label.startswith('Exodus') or label.startswith('Leviticus') or label.startswith('Numbers') or label.startswith('Deuteronomy') or label.startswith('Joshua') or label.startswith('Judges') or label.startswith('1_Samuel') or label.startswith('2_Samuel') or label.startswith('1_Kings') or label.startswith('2_Kings') or label.startswith('Isaiah') or label.startswith('Jeremiah') or label.startswith('Ezekiel')  or label.startswith('Hosea') or label.startswith('Amos') or label.startswith('Joel') or label.startswith('Jonah') or label.startswith('Micah') or label.startswith('Nahum') or label.startswith('Habakkuk') or label.startswith('Zephaniah') or label.startswith('Haggai') or label.startswith('Zechariah') or label.startswith('Malachi') or label.startswith('Song_of_songs') or label.startswith('Ecclesiastes') or label.startswith('Lamentations') or label.startswith('Esther') or label.startswith('Ruth') or label.startswith('Daniel') or label.startswith('Ezra') or label.startswith('Nehemiah') or label.startswith('1_Chronicles') or label.startswith('2_Chronicles') or label.startswith('Job21') : continue
            else:
                if label.startswith('Psalms') or label.startswith('Job ') or label.startswith('Proverbs'): continue

            tags = tags.strip()
            cur_sentence_tags = eval(tags)
            # yield
            yield (label, cur_sentence_tags)

class Vocabulary(object):
    def __init__(self):
        self.all_items = []

    def add_text(self, paragraph):
        self.all_items.extend(paragraph)

    def finalize(self):
        self.vocab = list(set(self.all_items))
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.i2c = self.vocab
        self.all_items = None

    def get_c2i(self):
        return self.c2i

    def size(self):
        return len(self.c2i)

    def __getitem__(self, c):
        return self.c2i.get(c, 0)

    def getItem(self, i):
        return self.i2c[i]

class WordEncoder(object):
    def __init__(self, name, dim, model, vocab):
        self.vocab = vocab
        self.enc = model.add_lookup_parameters((vocab.size(), dim))

    def __call__(self, char, NON_DEFAULT=False):
        if NON_DEFAULT: return self.enc[char]
        return self.enc[self.vocab[char]]

class MLP:
    def __init__(self, model, name, in_dim, hidden_dim, out_dim):
        self.mw = model.add_parameters((hidden_dim, in_dim))
        self.mb = model.add_parameters((hidden_dim))
        self.mw2 = model.add_parameters((out_dim, hidden_dim))
        self.mb2 = model.add_parameters((out_dim))

    def __call__(self, x, DO_SOFTMAX=True):
        W = parameter(self.mw)
        b = parameter(self.mb)
        W2 = parameter(self.mw2)
        b2 = parameter(self.mb2)
        mlp_output = W2 * (tanh(W * x + b)) + b2
        if not DO_SOFTMAX: return mlp_output
        return softmax(mlp_output)

class BILSTMTransducer:
    def __init__(self, LSTM_LAYERS, IN_DIM, OUT_DIM, model):
        self.lstmF = LSTMBuilder(LSTM_LAYERS, IN_DIM, (int)(OUT_DIM / 2), model)
        self.lstmB = LSTMBuilder(LSTM_LAYERS, IN_DIM, (int)(OUT_DIM / 2), model)

    def __call__(self, seq):
        """
        seq is a list of vectors (either character embeddings or bilstm outputs)
        """
        fw = self.lstmF.initial_state()
        bw = self.lstmB.initial_state()
        outf = fw.transduce(seq)
        outb = list(reversed(bw.transduce(reversed(seq))))
        return [concatenate([f, b]) for f, b in zip(outf, outb)]

def CalculateLossForParagraph(para, fValidation=False):
    renew_cg()
    # SWTICH TROP & POS here
    # generate the BiLSTM inputs
    bilstm_outputs = bilstm([pos_enc(pos) for _, pos, pos_sub, lex, hv, part, func, typ in para])
    bilstm_outputslex = bilstm([lex_enc(lex) for _, pos, pos_sub, lex, hv, part, func, typ in para])
    bilstm_outputshv = bilstm([hv_enc(hv) for _, pos, pos_sub, lex, hv, part, func, typ in para])
    bilstm_outputspart = bilstm([part_enc(part) for _, pos, pos_sub, lex, hv, part, func, typ in para])
    bilstm_outputsfunc = bilstm([func_enc(func) for _, pos, pos_sub, lex, hv, part, func, typ in para])
    bilstm_outputstyp = bilstm([typ_enc(typ) for _, pos, pos_sub, lex, hv, part, func, typ in para])
    bilstm_outputstr = bilstm([tr_enc(pos_sub) for _, pos, pos_sub, lex, hv, part, func, typ in para])


    # SWTICH TROP & POS here
    lstm_prev_state = lstm.initial_state().add_input(trop_enc('BOS'))

    acc = 0.0
    losses = []

    output_trop = []
    # go through each item

    outputs = [bilstm_outputs, bilstm_outputslex, bilstm_outputshv, bilstm_outputspart, bilstm_outputsfunc, bilstm_outputstyp, bilstm_outputstr]
    for iTrop, (trop, pos, tr, lex, hv, part, func, typ) in enumerate(para):
        # create the input - output from the LSTM and the BiLSTM output
        lst = [o[iTrop] for o, yn in zip(outputs, include[1:]) if yn] + [lstm_prev_state.output()]
        mlp_input = concatenate(lst)
        mlp_output = mlp(mlp_input)

        # get the predicted - check accuracy
        predicted_iTrop = np.argmax(mlp_output.npvalue())
        if predicted_iTrop == trop_vocab[trop]:
            acc += 1

        # loss
        losses.append(-log(pick(mlp_output, trop_vocab[trop])))

        # the next trop that goes into the LSTM - can either be the predicted one, or the correct one
        # if we are training, we want to put in the correct one
        # if we are evaluating, put in the predicted one
        if fValidation:
            lstm_prev_state = lstm_prev_state.add_input(trop_enc(trop_vocab.getItem(predicted_iTrop)))
            output_trop.append((lex, pos, tr, hv, part, func, typ, trop, trop_vocab.getItem(predicted_iTrop)))
        else:
            lstm_prev_state = lstm_prev_state.add_input(trop_enc(trop))

    if fValidation:
        return esum(losses), acc / len(para) * 100, output_trop

    # return the loss and accuracy
    return esum(losses), acc / len(para) * 100

def run_network_on_validation(data_to_run, suffix):
    val_loss = 0.0
    val_acc = 0.0

    output_lines = []
    # go through the val data
    for label, paragraph in data_to_run:
        # get the loss, acc, output
        loss, acc, output_trop = CalculateLossForParagraph(paragraph, fValidation=True)

        val_loss += loss.value() / len(paragraph)
        val_acc += acc

        line = label + '('
        for x in output_trop:
            for i, item in enumerate(x):
                if i == 0:
                    line += item.decode('utf-8')
                    line += ' - '
                else:
                    line += str(item) + ' - '
        line += ')'
        output_lines.append(line)
        # otuput (ADJV, TIRCHA, REVIA) - (SUBS, MIRCHA, MIRCHA) - ....

    logmessage('loss: ' + str(val_loss / len(data_to_run)) + ', acc: ' + str(val_acc / len(data_to_run)) + "\n")

    with codecs.open("Resultslog/Resultslog4" + strFeatures + '/Resultslog' + strFeatures + str(suffix) + '.txt', mode='w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')

    return val_loss, val_acc

all_data = list(read_data())
random.shuffle(all_data)

#('TIPCHA', 'subs', 'none', 'R>CJT/', 0, 'PP', 'Time', 'xQtX')
# (trup, part_of_speech, pos_subtype, simple_lex, hv, p_typ, p_function, c_typ)

# pos vocab
pos_vocab = Vocabulary()
for _, sentence in all_data:
    pos_vocab.add_text([t[1] for t in sentence])
pos_vocab.finalize()

# lex vocab
lex_vocab = Vocabulary()
for _, sentence in all_data:
    lex_vocab.add_text([t[2] for t in sentence])
lex_vocab.finalize()

# trailer vocab
tr_vocab = Vocabulary()
for _, sentence in all_data:
    tr_vocab.add_text([t[3] for t in sentence])
tr_vocab.finalize()

# halfverse vocab
hv_vocab = Vocabulary()
for _, sentence in all_data:
    hv_vocab.add_text([t[4] for t in sentence])
hv_vocab.finalize()

# ???? vocab
part_vocab = Vocabulary()
for _, sentence in all_data:
    part_vocab.add_text([t[5] for t in sentence])
part_vocab.finalize()

# func vocab
func_vocab = Vocabulary()
for _, sentence in all_data:
    func_vocab.add_text([t[6] for t in sentence])
func_vocab.finalize()

# typ vocab
typ_vocab = Vocabulary()
for _, sentence in all_data:
    typ_vocab.add_text([t[7] for t in sentence])
typ_vocab.finalize()

# trop vocab
trop_vocab = Vocabulary()
for _, sentence in all_data:
    trop_vocab.add_text([t[0] for t in sentence])
# SWTICH TROP & POS here - add BOS to pos as opposed to trop
trop_vocab.add_text(['BOS'])
trop_vocab.finalize()

# split into train & validation
val_size = int(len(all_data) * .1)
val_data = all_data[:val_size]
train_data = all_data[val_size:]

model = Model()  # or ParameterCollection()
# encoders:
pos_enc = WordEncoder('posenc', INPUT_DIM, model, pos_vocab)
lex_enc = WordEncoder('lexenc', INPUT_DIM, model, lex_vocab)
hv_enc = WordEncoder('hvenc', INPUT_DIM, model, hv_vocab)
part_enc = WordEncoder('partenc', INPUT_DIM, model, part_vocab)
func_enc = WordEncoder('funcenc', INPUT_DIM, model, func_vocab)
typ_enc = WordEncoder('typenc', INPUT_DIM, model, typ_vocab)
trop_enc = WordEncoder('tropenc', INPUT_DIM, model, trop_vocab)
tr_enc = WordEncoder('trenc', INPUT_DIM, model, tr_vocab)


# lstms - What should i add for lexical forms?
bilstm = BILSTMTransducer(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
lstm = LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

# our mlp
# SWTICH TROP & POS here
num_dimensions = include.count(True)
mlp = MLP(model, 'mlp', HIDDEN_DIM * num_dimensions, HIDDEN_DIM, trop_vocab.size())

trainer = AdamTrainer(model)


filename_to_load = 0

if filename_to_load:
    model.populate(filename_to_load)

average_loss = 0 # so that it exists
average_acc = 0
# train!

for epoch in range(25):
    # shuffle the train data
    random.shuffle(train_data)

    overall_loss = 0.0
    overall_acc = 0.0
    items_seen = 0
    # go through each para
    for label, paragraph in train_data:
        # calculate loss
        loss, acc = CalculateLossForParagraph(paragraph)

        overall_loss += loss.value() / len(paragraph)
        overall_acc += acc

        # backwards caluclation
        loss.backward()
        trainer.update()

        items_seen += 1
        # every 200 items, print an update
        # breakpoint = 200
        # if items_seen % breakpoint == 0:
        #     average_loss = overall_loss / breakpoint
        #     average_acc = overall_acc / breakpoint
        #
        #     # print
        #     logmessage('Items seen: ' + str(items_seen) + ', loss: ' + str(average_loss) + ', acc: ' + str(average_acc))
        #
        #     overall_acc = 0.0
        #     overall_loss = 0.0

    logmessage('Finished epoch: ' + str(epoch) + ", ")
    model.save("Modelslogs/Modelslog4" + strFeatures + "/rnn_trup_e" + str(epoch) + '_loss' + str(average_loss) + '_acc' + str(average_acc) + '.model')

    # run validation
    run_network_on_validation(val_data, epoch)

