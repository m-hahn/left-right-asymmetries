#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

#4 : uses lemmas


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

myID = random.randint(0,10000000)





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

#bad = set() #set(["punct", "flat", "aux", "case", "fixed", "appos", "cop", "parataxis"])



#good = set(["nsubj", "obj"]) # interestingly, these are the ones that are morphologically marked in Japanese


goodContent = set(["appos", "nsubj", "obj", "nmod", "acl", "obl", "xcomp", "advmod", "advcl", "amod", "nummod", "ccomp", "csubj", "iobj"]) # interestingly, these are the ones that are morphologically marked in Japanese

goodFunctional = set(["clf", "det", "aux", "case", "cop", "mark"]) # interestingly, these are the ones that are morphologically marked in Japanese


good = goodContent.union(goodFunctional)
#good = goodContent
#good = goodFunctional

other = set(["vocative", "discourse", "fixed", "dep", "cc", "flat", "dislocated", "conj", "compound", "parataxis", "expl" ])
artifcats = set(["goeswith", "reparandum", "list", "orphan", "punct"])

#good = good.union(other)







#good = good.union(set(["nsubj", "obj", "nmod", "obl",  "advmod", "amod", "nummod", "iobj"]) )
#good = good.union(set(["acl", "xcomp","advcl", "ccomp", "csubj"]))


rejected = {}

posDepTypeJointL = {}
posDepTypeJointR = {}
totalDepTypeCount = {}

for sentence in corpus.iterator():
  counter += 1
  if counter % 100 == 0:
    print counter
  for line in sentence:
  
    dependency = line['dep']
    if ":" in dependency:
       dependencyCoarse = dependency[:dependency.index(":")]
    else:
       dependencyCoarse = dependency
    if dependencyCoarse not in good:
      rejected[dependency] = rejected.get(dependency, 0) + 1
      continue

#    dependency = dependencyCoarse

    dependencies.add(dependency)

    if dependency == 'root':
       continue
    assert line['head'] > 0
    head = sentence[line['head']-1]
    depPOS = line['lemma']
    headPOS = head['lemma']
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

print(sorted(list(rejected.iteritems()), key=lambda x:x[1]))

unigramEntropyLeft = 0
unigramEntropyRight = 0

mi = 0
for pos, entry in posDepTypeJointL.iteritems():
   unigramEntropyLeft += entry['_TOTAL_'] / dependencyCount * log(entry['_TOTAL_'] / dependencyCount)

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
   unigramEntropyRight += entry['_TOTAL_'] / dependencyCount * log(entry['_TOTAL_'] / dependencyCount)

   for depType, joint in entry.iteritems():
       if depType == '_TOTAL_':
           continue
       jointProb = joint / dependencyCount
       posProb = entry['_TOTAL_'] / dependencyCount
       depTypeProb = totalDepTypeCount[depType] / dependencyCount
       mi += jointProb * (log(jointProb) - log(posProb) - log(depTypeProb))
print("Right", mi)

print("Unigram entropies", unigramEntropyLeft, unigramEntropyRight)

# Left-POS has more MI with the dependency type





