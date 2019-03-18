import os

files = os.listdir("../results5/")

print "\t".join(["Language", "Entropy", "Count", "Type", "Model", "LR"])
for name in sorted(files):
   if "Bugfix" not in name:
      continue
   with open("../results5/"+name, "r") as inFile:
      language = next(inFile).strip()
      entropies = [float(x) for x in next(inFile).strip().split("\t")]
      if len(entropies) < 2:
         continue
      commands = next(inFile).strip().split("\t")
#      print(commands)
      script = commands[0]
      model = commands[2]
      if len(commands) > 3:
        lr = commands[3]
        lr = lr[lr.index(":")+1:]
      else:
          lr = "NA"
      print "\t".join(map(str, [language, entropies[-2], len(entropies), script.replace("testLeftRightEntUniHDCond3Filter", ""), model, lr]))


