import sys


def readTuebaJSTreebank(partition):
      if partition == "valid":
         partition = "dev"
      prefix = "/u/scr/corpora/ldc/2015/LDC2015T11/conll2006_ten_lang/data/japanese/verbmobil/"
      if partition == "train":
          path = prefix+"train/japanese_verbmobil_train.conll"
      elif partition in ["dev", "valid"]:
          path = prefix+"test/japanese_verbmobil_test_gs.conll"
      else:
          assert False
      with open(path, "r") as inFile:
          data = inFile.read().strip().split("\n\n")
          if len(data) == 1:
            data = data[0].split("\r\n\r\n")
          assert len(data) > 1, (path, [len(x) for x in data], "\n\n" in data[0], "\r\n\r\n" in data[0] )
      assert len(data) > 0, (language, partition, files)
      print >> sys.stderr, "Read "+str(len(data))+ " sentences from 1 "+partition+" datasets."
      return data


