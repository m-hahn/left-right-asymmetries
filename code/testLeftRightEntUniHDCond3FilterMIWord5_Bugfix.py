#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"



# TODO also try other optimizers

import random
import sys

objectiveName = "LM"

language = sys.argv[1]
model = sys.argv[2] #sys.argv[2]
#sys.argv.append(random.choice(["0.0", "0.05", "0.1"]))
#input_noise = float(sys.argv[12])
#assert len(sys.argv) == 13

batchSize = 1
lr_lm = 0.001 # used to be 0.05
sys.argv.append("LR_LM:"+str(lr_lm))
myID = random.randint(0,10000000)
#


#


posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]

deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp, sqrt
from random import random, shuffle, randint, choice
import os

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator_FuncHead import CorpusIteratorFuncHead

originalDistanceWeights = {}


def makeCoarse(x):
   if ":" in x:
      return x[:x.index(":")]
   return x

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIteratorFuncHead(language,partition).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          line["coarse_dep"] = makeCoarse(line["dep"])
          depsVocab.add(line["coarse_dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          if line["coarse_dep"] == "root":
             continue
          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["coarse_dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = dep
          keyWithDir = (dep, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   #print orderTable
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key, "HD"), 0) + 1.0
      dh = orderTable.get((key, "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum #+ sum(line.get("children_decisions_logprobs",[]))

   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = [distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]] for x in remainingChildren]
           #print logits
           if reverseSoftmax:
              logits = [-x for x in logits]
           #print (reverseSoftmax, logits)
           softmax = logits #.view(1,-1).view(-1)
           selected = numpy.argmax(softmax)
           #selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
           #log_probability = torch.log(softmax[selected])
           assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
           #sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
       return childrenLinearized           


def orderSentence(sentence, dhLogits, printThings):
   global model

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   if model in ["REAL_REAL", "REVERSE"]:
      eliminated = []
   for line in sentence:
      line["coarse_dep"] = makeCoarse(line["dep"])
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         if model in ["REAL_REAL", "REVERSE"]:
            eliminated.append(line)
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      probability = 1/(1 + exp(-dhLogit))
      dhSampled = (0.5 < probability)
      line["ordering_decision_log_probability"] = 0 #torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability)+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])

      sentence[headIndex]["children_total"] = (sentence[headIndex].get("children_total", []) + [line])


   if model not in ["REAL_REAL", "REVERSE"]:
      for line in sentence:
         if "children_DH" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
            line["children_DH"] = childrenLinearized
         if "children_HD" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
            line["children_HD"] = childrenLinearized
   else:
       while len(eliminated) > 0:
          line = eliminated[0]
          del eliminated[0]
          if "removed" in line:
             continue
          line["removed"] = True
          if "children_DH" in line:
            assert 0 not in line["children_DH"]
            eliminated = eliminated + [sentence[x-1] for x in line["children_DH"]]
          if "children_HD" in line:
            assert 0 not in line["children_HD"]
            eliminated = eliminated + [sentence[x-1] for x in line["children_HD"]]


   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0])))
   if model in ["REAL_REAL", "REVERSE"]:
      linearized = filter(lambda x:"removed" not in x, sentence)

   if model == "REVERSE":
       linearized = linearized[::-1]

   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))

   for i in range(len(linearized)):
      linearized[i]["reordered_index"] = i+1
   reordering = dict((x["index"], x["reordered_index"]) for x in linearized)

   currentlyCrossing = []
   for line in linearized: # they already come in linear order
      currentlyCrossing = [a for a in currentlyCrossing if max(a[0], a[1]) > line["reordered_index"]]
      if line["coarse_dep"] != "root":
        assert reordering[line["head"]] != line["reordered_index"]
        if reordering[line["head"]] > line["reordered_index"]:
           currentlyCrossing.append((reordering[line["head"]], line["reordered_index"], line["coarse_dep"], "L", reordering[line["head"]], line["reordered_index"]))
#        else:
 #          currentlyCrossing = [(x,y,z,a,b) for x,y,z,a,b in currentlyCrossing if y != line["reordered_index"]]
  #         print([y != line["reordered_index"] for x,y,z,a,b in currentlyCrossing])

      for dependent in line.get("children_total", []):
        if "reordered_index" not in dependent:
           continue
        assert dependent["coarse_dep"] != "punct"

        if dependent["reordered_index"] > line["reordered_index"]:
           assert reordering[dependent["head"]] == line["reordered_index"]
           currentlyCrossing.append((line["reordered_index"], dependent["reordered_index"], dependent["coarse_dep"], "R", dependent["reordered_index"], dependent["head"]))
#      currentlyCrossing = [x for x in currentlyCrossing if x[2] != "mrk" and x[2] != "adj"]
      line["crossing"] = sorted(currentlyCrossing, key=lambda x:abs(x[4]-x[5]))
 #     print(line["reordered_index"], line["crossing"])
#   quit()
   return linearized, logits


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()

posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

posFine = list(posFine)
itos_pos_ptb = posFine
stoi_pos_ptb = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

itos_state = [x+y for x in itos_pure_deps for y in ["R", "L"]] + ["<pad>", "<start>", "<eos>"]
stoi_state = dict(zip(itos_state, range(len(itos_state))))


print itos_deps

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)
for i, key in enumerate(itos_deps):

   # take from treebank, or randomize
   dhLogits[key] = 2*(random()-0.5)
   dhWeights[i] = dhLogits[key]

   originalDistanceWeights[key] = random()  
   distanceWeights[i] = originalDistanceWeights[key]

import os

if model != "RANDOM" and model not in ["REAL_REAL", "REVERSE"] and model != "RLR":
   temperature = 1.0
   inpModels_path = "/u/scr/mhahn/deps/"+"/"+BASE_DIR+"/"
   models = os.listdir(inpModels_path)
   models = filter(lambda x:"_"+model+".tsv" in x, models)
   if len(models) == 0:
     assert False, "No model exists"
   if len(models) > 1:
     assert False, [models, "Multiple models exist"]
   
   with open(inpModels_path+models[0], "r") as inFile:
      data = map(lambda x:x.split("\t"), inFile.read().strip().split("\n"))
      header = data[0]
      data = data[1:]
    
   #print header
   #quit()
   # there might be a divergence because 'inferWeights...' models did not necessarily run on the full set of corpora per language (if there is no AllCorpora in the filename)
   #assert len(data) == len(itos_deps), [len(data), len(itos_deps)]
   if "Dependency" not in header:
      header[header.index("CoarseDependency")] = "Dependency"
   if "DH_Weight" not in header:
      header[header.index("DH_Mean_NoPunct")] = "DH_Weight"
   if "DistanceWeight" not in header:
      header[header.index("Distance_Mean_NoPunct")] = "DistanceWeight"

   for line in data:
   #   print line
#      head = line[header.index("Head")]
 #     dependent = line[header.index("Dependent")]
      dependency = line[header.index("Dependency")]
      key = dependency
      dhWeights[stoi_deps[key]] = temperature*float(line[header.index("DH_Weight")])
      distanceWeights[stoi_deps[key]] = temperature*float(line[header.index("DistanceWeight")])
elif model == "RANDOM":
   #assert BASE_DIR == "RANDOM"
   save_path = "/juicier/scr120/scr/mhahn/deps/"
   #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
#   with open(save_path+"/manual_output_funchead_RANDOM/"+language+"_"+"RANDOM"+"_model_"+str(myID)+".tsv", "w") as outFile:
#      print >> outFile, "\t".join(map(str,["FileName","DH_Weight", "CoarseDependency","DistanceWeight" ]))
#      for i in range(len(itos_deps)):
#         key = itos_deps[i]
#         dhWeight = dhWeights[i]
#         distanceWeight = distanceWeights[i]
#         dependency = key
#         print >> outFile, "\t".join(map(str,[myID, dhWeight, dependency, distanceWeight]))
elif model == "RLR":
   assert BASE_DIR == "RLR"
#   for i, key in enumerate(itos_deps):
#   
#      # take from treebank, or randomize
#      dhLogits[key] = 2*(random()-0.5)
#      dhWeights[i] = dhLogits[key]
#   
#      originalDistanceWeights[key] = 8*(random()-0.5) # so the range is [-4, 4]
#       = originalDistanceWeights[key]
   
   

   temperature = 1.0
   inpModels_path = "/u/scr/mhahn/deps/manual_output_funchead_ground_coarse_final/"
   models = os.listdir(inpModels_path)
   models = filter(lambda x:x.startswith(language+"_"), models)
   if len(models) == 0:
     assert False, "No model exists "+language
   if len(models) > 1:
     assert False, [models, "Multiple models exist"]
   
   with open(inpModels_path+models[0], "r") as inFile:
      data = map(lambda x:x.split("\t"), inFile.read().strip().split("\n"))
      header = data[0]
      data = data[1:]
    
   if "Dependency" not in header:
      header[header.index("CoarseDependency")] = "Dependency"
   if "DH_Weight" not in header:
      header[header.index("DH_Mean_NoPunct")] = "DH_Weight"
   if "DistanceWeight" not in header:
      header[header.index("Distance_Mean_NoPunct")] = "DistanceWeight"

   dontPredefine = ["lifted_case", "lifted_cop", "aux", "nmod", "acl", "lifted_mark", "obl", "xcomp", "nsubj", "advmod"]
   print("UNKNOWN RELATIONS", [x for x in dontPredefine if x not in stoi_deps])
   for line in data:
      dependency = line[header.index("Dependency")]
      key = dependency
      if dependency not in dontPredefine:
         dhWeights[stoi_deps[key]] = float(line[header.index("DH_Weight")])
      distanceWeights[stoi_deps[key]] = float(line[header.index("DistanceWeight")])
   for key in dontPredefine:
     if key in stoi_deps:
        other = key
        while other in dontPredefine:
           other = choice(itos_deps)
        distanceWeights[stoi_deps[key]] = distanceWeights[stoi_deps[other]]


   save_path = "/juicier/scr120/scr/mhahn/deps/"
   #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
#   with open(save_path+"/manual_output_funchead_RLR/"+language+"_"+"RLR"+"_model_"+str(myID)+".tsv", "w") as outFile:
#      print >> outFile, "\t".join(map(str,["FileName","DH_Weight", "CoarseDependency","DistanceWeight" ]))
#      for i in range(len(itos_deps)):
#         key = itos_deps[i]
#         dhWeight = dhWeights[i]
#         distanceWeight = distanceWeights[i]
#         dependency = key
#         print >> outFile, "\t".join(map(str,[myID, dhWeight, dependency, distanceWeight]))
#
#
#

words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))
#print stoi
#print itos[5]
#print stoi[itos[5]]

assert stoi[itos[5]] == 5

#print dhLogits

#for sentence in getNextSentence():
#   print orderSentence(sentence, dhLogits)

vocab_size_words = min(len(itos), 50000)


# 0 EOS, 1 UNK, 2 BOS
word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size_words+3, embedding_dim = 50).cuda()
pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()

state_embeddings = torch.nn.Embedding(num_embeddings = len(itos_state), embedding_dim=50).cuda()


#baseline = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim=1).cuda()
#baseline_upos = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim=1).cuda()
#baseline_ppos = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=1).cuda()



dropout = nn.Dropout(0.3).cuda()

rnn = nn.LSTM(70, 128, 1).cuda()
for name, param in rnn.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

rnn_state = nn.LSTM(50, 128, 1).cuda()
for name, param in rnn_state.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)


vocab_size_states = len(itos_state)

decoder = nn.Linear(128,vocab_size_states).cuda()

components = [word_embeddings, pos_u_embeddings, pos_p_embeddings, rnn, rnn_state, decoder, state_embeddings ]

#def parameters_policy():
# yield dhWeights
# yield distanceWeights

def parameters_lm():
 for c in components:
   for param in c.parameters():
      yield param



def parameters():
 for c in components:
   for param in c.parameters():
      yield param
# yield dhWeights
# yield distanceWeights

parameters_cached = [x for x in parameters()]

#parameters_policy_cached = [x for x in parameters_policy()]
parameters_lm_cached = [x for x in parameters_lm()]

#for pa in parameters():
#  print pa

initrange = 0.1
word_embeddings.weight.data.uniform_(-initrange, initrange)
pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
pos_p_embeddings.weight.data.uniform_(-initrange, initrange)
state_embeddings.weight.data.uniform_(-initrange, initrange)

decoder.bias.data.fill_(0)
decoder.weight.data.uniform_(-initrange, initrange)
##baseline.bias.data.fill_(0)
#baseline.weight.data.fill_(0) #uniform_(-initrange, initrange)
#baseline_upos.weight.data.fill_(0) #uniform_(-initrange, initrange)
#baseline_ppos.weight.data.fill_(0) #uniform_(-initrange, initrange)




crossEntropy = 10.0

def encodeWord(w, doTraining):
#   if doTraining and random() < input_noise and len(stoi) > 10:
 #     return 3+randint(0, len(itos)-1)
   return stoi[w]+3 if stoi[w] < vocab_size_words else 1

def regularisePOS(w, doTraining):
   return w
#   if doTraining and random() < 0.01 and len(stoi_pos_ptb) > 10:
#      return 3+randint(0, len(stoi_pos_ptb)-1)
#   return w

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



import torch.cuda
import torch.nn.functional


baselineAverageLoss = 0

counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 
devLossesWords = []
devLossesPOS = []

loss_op = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index = stoi_state["<pad>"])



def doForwardPass(current, train=True):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       global baselineAverageLoss
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y==0 and printHere), zip(range(len(current)),current)))


      
       batchOrdered = batchOrderedLogits[0]
       logits = batchOrderedLogits[1]



   
       lengths = map(len, current)
       # current is already sorted by length
       maxLength = max(lengths)
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"], train) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 if i == 0 else (regularisePOS(stoi_pos_ptb[x[i-1]["posFine"]]+3, train) if i <= len(x) else 0), batchOrdered))

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()))
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       for c in components:
          c.zero_grad()
       totalQuality = 0.0
       if True:
           crossing = [[None for _ in x] for x in batchOrdered]
           for i, x in enumerate(batchOrdered):
              for j, line in enumerate(x):
#                print(line["crossing"])
                crossing[i][j] = ["<start>"]+[y[2]+y[3] for y in line["crossing"]]+["<eos>"]
              crossing[i] = [[]] + crossing[i]
              while len(crossing[i]) < maxLength+2:
                 crossing[i].append([])
              assert len(crossing[i]) == maxLength+2

           maxCrossHeight = max([max([len(y) for y in x]) for x in crossing])
 #          print("maxCrossHeight", maxCrossHeight)
           for j in crossing:
               for i in range(len(j)):
  #               print(len(j[i]))
                 j[i] = j[i] + ["<pad>" for _ in range(maxCrossHeight-len(j[i]))]
                 j[i] = [stoi_state[x] for x in j[i]]

           assert batchSize == 1
           crossingIndices = Variable(torch.LongTensor(crossing)).cuda().squeeze(0).transpose(0,1)
   #        print(crossingIndices)


           wordIndices = Variable(torch.LongTensor(input_words)).cuda()

           #print("TWO INPUTS",crossingIndices.size(), wordIndices.size(), maxLength+2)


           pos_p_indices = Variable(torch.LongTensor(input_pos_p)).cuda()
           words_layer = word_embeddings(wordIndices)
           pos_u_indices = Variable(torch.LongTensor(input_pos_u)).cuda()
           pos_u_layer = pos_u_embeddings(pos_u_indices)
           pos_p_layer = pos_p_embeddings(pos_p_indices)
           input_concat = torch.cat([words_layer, pos_u_layer, pos_p_layer], dim=2)
           inputEmbeddings = dropout(input_concat) if train else input_concat

           stateInput = state_embeddings(crossingIndices)
#           print("--------")
#           print(torch.LongTensor(input_pos_p).size())
#           print(input_pos_p)
#           print(crossingIndices[1:].size())
#           print(crossingIndices[1:])
           output, _ = rnn(inputEmbeddings, None)
           assert batchSize == 1
           output = output.squeeze(1).unsqueeze(0)
 #          print("OUTPUT", output.size())
#           print("stateInput", stateInput.size())
           output_state, _ = rnn_state(stateInput[:-1], (output, output))

           droppedOutput = dropout(output_state) if train else output_state

           logits = decoder(droppedOutput)
           logits = logits.view(-1, vocab_size_states)
           word_softmax = logsoftmax(logits)
           word_softmax = word_softmax.view(-1, vocab_size_states)

  #         print(word_softmax.size(), crossingIndices[1:].size())
           loss = loss_op(word_softmax.view(-1, vocab_size_states), crossingIndices[1:].contiguous().view(-1)).view(-1, batchSize).sum()

       wordNum = maxLength
       if printHere and wordNum > 0:
         print loss/wordNum
#           losses = loss(predictions, input_words[i+1]) 
#           print losses
#    for i, sentence in enumerate(batchOrderLogits):
#       embeddingsLayer
         print lossWords/wordNum
         print ["CROSS ENTROPY", crossEntropy, (crossEntropy)]
         print baselineAverageLoss
       crossEntropy = 0.99 * crossEntropy + 0.01 * float((loss).data.cpu().numpy())
       totalQuality = loss.data.cpu().numpy() # consists of lossesWord + lossesPOS
       numberOfWords = wordNum
#       probabilities = torch.sigmoid(dhWeights)
#       print ["MEAN PROBABILITIES", torch.mean(probabilities)]
       #print ["PG", policyGradientLoss]

       
 #      neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))

       policy_related_loss = 0# 0 * (0 * neg_entropy + policyGradientLoss) # lives on CPU
       return loss, baselineLoss, policy_related_loss, totalQuality, numberOfWords, loss.data.cpu().numpy(),0 


def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       global lastDevLoss
       global failedDevRuns
       if printHere:
         print "BACKWARD 1"
#       print("POLICY LOSS")
#       print(policy_related_loss)
       #policy_related_loss.backward()
       if printHere:
         print "BACKWARD 2"

#       loss += 0 * neg_entropy
#       loss += 0 * policyGradientLoss

#       loss += lr_baseline * baselineLoss.sum() # lives on GPU
       loss.backward()
       if printHere:
         print sys.argv
         print "BACKWARD 3 "+__file__+" "+language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+(" ".join(map(str,["ENTROPY", 0, "LR_POLICY", 0, "MOMENTUM", 0])))
         print devLosses
         print devLossesWords
         print devLossesPOS

#       print("Before")
       #if printHere:
      #    print([counter, "Distance grad", torch.abs(distanceWeights.grad).sum()])
    #   torch.nn.utils.clip_grad_norm(parameters_policy_cached, 5.0, norm_type='inf')
#       print(torch.abs(distanceWeights.grad).sum())
  #     torch.nn.utils.clip_grad_norm_(parameters_lm_cached, 5.0, norm_type='inf')
#       if  printHere:
#          print("grad norm", max(p.grad.data.abs().max() for p in parameters_lm_cached))
 #      print(distanceWeights.grad.sum())
  #     print(";;;")
       counterHere = 0
       for param in parameters_cached:
         counterHere += 1
         #print "UPDATING"
         if counter < 50 and (param is distanceWeights or param is dhWeights): # allow baseline to warum up
             continue
         if param.grad is None:
           print counterHere
           print "WARNING: None gradient"
           continue
         param.data.sub_(lr_lm * param.grad.data)


def computeDevLoss():
   global printHere
   global counter
   devLoss = 0.0
   devLossWords = 0.0
   devLossPOS = 0.0
   devWords = 0
#   corpusDev = getNextSentence("dev")
   corpusDev = CorpusIteratorFuncHead(language,"dev").iterator(rejectShortSentences = True)

   while True:
     try:
        batch = map(lambda x:next(corpusDev), range(batchSize))
     except StopIteration:
        break
     batch = sorted(batch, key=len)
     partitions = range(1)
     shuffle(partitions)
     for partition in partitions:
        counter += 1
        printHere = (counter % 50 == 0)
        current = batch[partition*batchSize:(partition+1)*batchSize]
 
        _, _, _, newLoss, newWords, lossWords, lossPOS = doForwardPass(current, train=False)
        devLoss += newLoss
        devWords += newWords
        devLossWords += lossWords
        devLossPOS += lossPOS
   return devLoss/devWords, devLossWords/devWords, devLossPOS/devWords

while True:
#  corpus = getNextSentence("train")
  corpus = CorpusIteratorFuncHead(language)
  corpus.permute()
  corpus = corpus.iterator(rejectShortSentences = True)


  while True:
    try:
       batch = map(lambda x:next(corpus), 10*range(batchSize))
    except StopIteration:
       break
    batch = sorted(batch, key=len)
    partitions = range(10)
    shuffle(partitions)
    for partition in partitions:
       counter += 1
       printHere = (counter % 100 == 0)
       current = batch[partition*batchSize:(partition+1)*batchSize]

       loss, baselineLoss, policy_related_loss, _, wordNumInPass, lossWords, lossPOS = doForwardPass(current)
       if wordNumInPass > 0:
         doBackwardPass(loss, baselineLoss, policy_related_loss)
       else:
         print "No words, skipped backward"
  if True: #counter % 10000 == 0:
          newDevLoss, newDevLossWords, newDevLossPOS = computeDevLoss()
          devLosses.append(newDevLoss)
          devLossesWords.append(newDevLossWords)
          devLossesPOS.append(newDevLossPOS)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
          print "Saving"
          save_path = "../results5/"
          #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
          with open(save_path+"/"+language+"_"+__file__+"_languageModel_performance_"+model+"_"+str(myID)+".tsv", "w") as outFile:
             print >> outFile, language
             print >> outFile, "\t".join(map(str, devLosses))
#             print >> outFile, "\t".join(map(str, devLossesWords))
#             print >> outFile, "\t".join(map(str, devLossesPOS))
             print >> outFile, "\t".join(map(str, sys.argv))




          if lastDevLoss is None or newDevLoss < lastDevLoss:
             lastDevLoss = newDevLoss
             failedDevRuns = 0
          else:
             failedDevRuns += 1
             print "Skip saving, hoping for better model"
             print devLosses
             quit()
#             lr_lm *= 0.5
 #            continue
#dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#for i, key in enumerate(itos_deps):
#
#   # take from treebank, or randomize
#   dhLogits[key] = 2*(random()-0.5)
#   dhWeights.data[i] = dhLogits[key]
#
#   originalDistanceWeights[key] = random()  
#   distanceWeights.data[i] = originalDistanceWeights[key]
#
#
#
