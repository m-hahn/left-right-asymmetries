import os
import random
import accessISWOCData
import accessTOROTData
import sys

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]


def encode(word):
   return {"posUni" : word}

def readUDCorpus(language, partition):
      data = [[]]
      if language == "even":
         state = 0
         for _ in range(100000 if partition == "dev" else 100000):
           if state == 0:
             if random.random() < 0.5:
                 data[-1].append(encode("0"))
             else:
                state = 1
                data[-1].append(encode("1"))
           else:
               state = 0
               data[-1].append(encode("1"))
      elif language == "rrxor": # Stiller and Crutchfield 2010

#          # random starting position
#          a = (random.random() > 0.5)
#          b = (random.random() > 0.5)
#          if random.random() < 0.66:
#              if random.random() > 0.5:
#                  data[-1].append(encode("1" if a else "0"))
#              data[-1].append(encode("1" if b else "0"))
#          data[-1].append(encode("1" if a != b else "0"))

          for _ in range(10000 if partition == "dev" else 100000):
              a = (random.random() > 0.5)
              b = (random.random() > 0.5)

              while random.random() > 0.9: # randomness makes it stationary
                  data[-1].append(encode("2"))
              data[-1].append(encode("1" if a else "0"))
              while random.random() > 0.9:
                  data[-1].append(encode("2"))
              data[-1].append(encode("1" if b else "0"))
              while random.random() > 0.9:
                  data[-1].append(encode("2"))
              data[-1].append(encode("1" if a != b else "0"))
      elif language == "modulo8":
         for _ in range(10000 if partition == "dev" else 5000):
           # add 0s
           for _ in range(random.randint(1,8)):
              data[-1].append(encode("0"))
           # add 1s
           for _ in range(8):
              data[-1].append(encode("1"))
      elif language == "modulo8_2":
         for _ in range(10000 if partition == "dev" else 5000):
           # add 0s
           data[-1].append(encode("0"))
           while random.random() > 0.5:
              data[-1].append(encode("0"))
           # add 1s
           for _ in range(8):
              data[-1].append(encode("1"))

      elif language == "forget2":
         lastLast = "0"
         last = "0"
         for _ in range(100000 if partition == "dev" else 100000):
            if random.random() < 0.1: # replicate lastLast
                data[-1].append(encode(lastLast))
                new = lastLast
            else:
                new = random.choice(["0","1"])
                data[-1].append(encode(new))
            lastLast, last = last, new
      elif language == "forget2_0_5":
         lastLast = "0"
         last = "0"
         for _ in range(100000 if partition == "dev" else 100000):
            if random.random() < 0.5: # replicate lastLast
                data[-1].append(encode(lastLast))
                new = lastLast
            else:
                new = random.choice(["0","1"])
                data[-1].append(encode(new))
            lastLast, last = last, new
      elif language == "forget2_0_5b":
         lastLast = "0"
         last = "0"
         for _ in range(100000 if partition == "dev" else 100000):
            if random.random() < 0.5: # replicate lastLast
                data[-1].append(encode(lastLast))
                new = lastLast
            else:
                new = random.choice(["0","1"])
                data[-1].append(encode(new))
            lastLast = new
      elif language == "repeat":
        for _ in range(3000 if partition == "dev" else 3000):
            word = [random.choice(["0", "1", "2"]) for _ in range(15)]
            for i in range(len(word)):
               data[-1].append(encode(word[i]))
            word = word[::-1]
            for i in range(len(word)):
               data[-1].append(encode(word[i]))
      elif language == "repeat2":
        for _ in range(3000 if partition == "dev" else 3000):
            word = [random.choice(["0", "1"]) for _ in range(15)]
            for i in range(len(word)):
               data[-1].append(encode(word[i]))
            word = word[::-1]
            for i in range(len(word)):
               data[-1].append(encode(word[i]))
      elif language == "rip" or language == "rip_b":
         state = 0
         for _ in range(100000 if partition == "dev" else 100000):
             if state == 0:
                if random.random() < 0.5:
                    data[-1].append(encode("0"))
                    state = 1
                else:
                    data[-1].append(encode("1"))
                    state = 2
             elif state == 1:
                 if random.random() < 0.5:
                     data[-1].append(encode("0"))
                 else:
                     data[-1].append(encode("1"))
                 state = 2
             elif state == 2:
                 data[-1].append(encode("1"))
                 state = 0
             else:
                  assert False
         if language == "rip_b":
            data[-1] = data[-1][::-1]
      else:
          assert False
      return data

class CorpusIteratorToy():
   def __init__(self, language, partition="train", storeMorph=False, splitLemmas=False):
      data = readUDCorpus(language, partition)
      random.shuffle(data)
      self.data = data
      self.partition = partition
      self.language = language
      assert len(data) > 0, (language, partition)
   def permute(self):
      random.shuffle(self.data)
   def length(self):
      return len(self.data)
   def processSentence(self, sentence):
        return sentence 
   def getSentence(self, index):
      result = self.processSentence(self.data[index])
      return result
   def iterator(self, rejectShortSentences = False):
     for sentence in self.data:
        if len(sentence) < 3 and rejectShortSentences:
           continue
        yield self.processSentence(sentence)


