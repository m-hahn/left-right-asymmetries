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
dependencyCount = 0

dependencies = set()

bad = set(["punct", "flat", "aux", "case", "fixed", "appos", "cop", "parataxis"])


posDepTypeJointL = {}
posDepTypeJointR = {}
totalDepTypeCount = {}

for sentence in corpus.iterator():
  counter += 1
  if counter % 100 == 0:
    print counter
  for line in sentence:
  
    dependency = line['dep']
    if dependency in bad:
      continue
    dependencies.add(dependency)

    if dependency == 'root':
       continue
    assert line['head'] > 0
    head = sentence[line['head']-1]
    depPOS = line['word']
    headPOS = head['word']
    assert line['index'] > 0
    DH = (line['head'] >  line['index'])
    left = (depPOS if DH else headPOS)
    right = (depPOS if not DH else headPOS)
    register(left, right, conditionalCountsLR)
    register(right, left, conditionalCountsRL)
    register(left, dependency, posDepTypeJointL)
    register(right, dependency, posDepTypeJointR)
    totalDepTypeCount[dependency] = totalDepTypeCount.get(dependency,0.0) + 1.0

    dependencyCount += 1
#print conditionalCountsLR
#print conditionalCountsRL


mi = 0
for pos, entry in posDepTypeJointL.iteritems():
   for depType, joint in entry.iteritems():
       if depType == '_TOTAL_':
           continue
       jointProb = joint / dependencyCount
       posProb = entry['_TOTAL_'] / dependencyCount
       depTypeProb = totalDepTypeCount[depType] / dependencyCount
       mi += jointProb * (log(jointProb) - log(posProb) - log(depTypeProb))
print("Left", mi)

mi = 0
for pos, entry in posDepTypeJointR.iteritems():
   for depType, joint in entry.iteritems():
       if depType == '_TOTAL_':
           continue
       jointProb = joint / dependencyCount
       posProb = entry['_TOTAL_'] / dependencyCount
       depTypeProb = totalDepTypeCount[depType] / dependencyCount
       mi += jointProb * (log(jointProb) - log(posProb) - log(depTypeProb))
print("Right", mi)


# Left-POS has more MI with the dependency type





