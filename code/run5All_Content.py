from ud_languages import languages

import subprocess

languages = sorted(languages, reverse=True)

for language in languages:
  for model in ["REAL_REAL", "REVERSE"]: #, "GROUND"] + (["RANDOM_BY_TYPE"] * 5):
    command = ["./python27", "testLeftRightEntUniHDCond3FilterMIWord5_Content_Bugfix.py", language, model]
    subprocess.call(command)


