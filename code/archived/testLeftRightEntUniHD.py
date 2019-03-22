#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys


#import torch.nn as nn
#import torch
#from torch.autograd import Variable
import math
from math import log, exp, sqrt

#from pyro.infer import SVI
#from pyro.optim import Adam
#
#
#import pyro
#import pyro.distributions as dist
#
#import pyro
#from pyro.distributions import Normal, Bernoulli
#from pyro.infer import SVI
#from pyro.optim import Adam
#
#
#import pyro
#import pyro.poutine as poutine
#from pyro.infer.elbo import ELBO
#from pyro.poutine.util import prune_subsample_sites
#from pyro.util import check_model_guide_match

import os

language = sys.argv[1]
languageCode = sys.argv[2]

myID = random.randint(0,10000000)



deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp
from random import shuffle


from corpusIterator import CorpusIterator

corpus = CorpusIterator(language)
devSet = CorpusIterator(language,"dev")


leftCounts = {}
rightCounts = {}
conditionalCountsLR = {}
conditionalCountsRL = {}


def register(x, y, counts):
  if x not in counts:
    counts[x] = {'_TOTAL_' : 0}
  counts[x][y] = counts[x].get(y,0.0) + 1.0
  counts[x]['_TOTAL_'] = counts[x]['_TOTAL_'] + 1.0


counter = 0

for sentence in corpus.iterator():
  counter += 1
  if counter % 100 == 0:
    print counter
  for line in sentence:
  
    dependency = line['dep']
    if dependency == 'root':
       continue
    assert line['head'] > 0
    head = sentence[line['head']-1]
    depPOS = line['posUni']
    headPOS = head['posUni']
    assert line['index'] > 0
    DH = (line['head'] >  line['index'])
    left = (depPOS if DH else headPOS)
    right = (depPOS if not DH else headPOS)
    register(left, right, conditionalCountsLR)
    register(right, left, conditionalCountsRL)

print conditionalCountsLR
print conditionalCountsRL



delta = 0.5

sign = 0.0
sign_DH = 0.0
sign_HD = 0.0
counter = 0
dependencyCounter = 0
for sentence in corpus.iterator():
  counter += 1
  if counter % 100 == 0:
    print counter
  surpLR = 0.0
  surpRL = 0.0

  surpLR_DH = 0.0
  surpRL_DH = 0.0
  surpLR_HD = 0.0
  surpRL_HD = 0.0



  for line in sentence:
    dependency = line['dep']

    if dependency == 'root':
       continue
    assert line['head'] > 0
    dependencyCounter += 1
    head = sentence[line['head']-1]
    depPOS = line['posUni']
    headPOS = head['posUni']
    assert line['index'] > 0
    DH = (line['head'] >  line['index'])
    left = (depPOS if DH else headPOS)
    right = (depPOS if not DH else headPOS)
    partLR = log(max(conditionalCountsLR[left][right]-delta, 0.0) + delta * conditionalCountsRL[right]['_TOTAL_'] * (len(conditionalCountsLR[left])-1)) - log(conditionalCountsLR[left]['_TOTAL_'])
    partRL = log(max(conditionalCountsRL[right][left]-delta, 0.0) + delta * conditionalCountsLR[left]['_TOTAL_'] * (len(conditionalCountsRL[right])-1)) - log(conditionalCountsRL[right]['_TOTAL_'])
    if DH:
       surpLR_DH -= partLR
       surpRL_DH -= partRL
    else:
       surpLR_HD -= partLR
       surpRL_HD -= partRL


    surpLR -= partLR
    surpRL -= partRL

  sign += (1 if surpLR < surpRL else (0.5 if surpLR == surpRL else 0))
  sign_DH += (1 if surpLR_DH < surpRL_DH else (0.5 if surpLR_DH == surpRL_DH else 0))
  sign_HD += (1 if surpLR_HD < surpRL_HD else (0.5 if surpLR_HD == surpRL_HD else 0))

  # sign ends up > 0.5, showing surpRL < surpLR, meaning predicting left from right results in lower surprisal
  if counter % 100 == 0:
    print (sign, counter, sign/counter, sign_DH/counter, sign_HD/counter)



# compute unigram entropy
entL = 0
entR = 0
for left, entry in conditionalCountsLR.iteritems():
   entL -= entry['_TOTAL_'] * (log(entry['_TOTAL_']) - log(dependencyCounter))
for right, entry in conditionalCountsRL.iteritems():
   entR -= entry['_TOTAL_'] * (log(entry['_TOTAL_']) - log(dependencyCounter))
print [entL/dependencyCounter, entR/dependencyCounter]






