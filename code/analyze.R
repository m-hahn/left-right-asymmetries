data = read.csv("results.tsv", sep="\t")

library(dplyr)
library(tidyr)
library(lme4)
data = data %>% filter(grepl("Bugfix", Type)) %>% filter(LR < 0.1) # 0.1 was the first learning rate I tried, but it leads to bad losses on some of the larger corpora


##################################
# Columns of data:
#  Language
#  Entropy: cross entropy of predicting the stack of dependencies
#  Count ??
#  Type: what dependencies are considered, a suffix of the name of the script used to generate
#  Model: REAL_REAL or REVERSE
#  LR: learning rate used (initially, there was some hyperparameter tuning. Can be ignored.)

# Levels of data$Type:
# [1] "MIWord5_Bugfix.py"                       default (UD with function word head modification)
# [2] "MIWord5_Content_Bugfix.py"               only content dependencies
# [3] "MIWord5_Content_PlainUD_Bugfix.py"       only content + plain UD (no function word heads)
# [4] "MIWord5_NoUDArtifacts_Bugfix.py"         only 'core' dependencies, no 'UD artifacts'
# [5] "MIWord5_NoUDArtifacts_PlainUD_Bugfix.py" plain UD + no 'artifacts'
# [6] "MIWord5_PlainUD_Bugfix.py"               plain UD

# Is entropy different between REAL and REVERSE in the "MIWord5_NoUDArtifacts_Bugfix.py" setting?
summary(lmer(Entropy ~ Model + (1|Language), data=data %>% filter(Type == "MIWord5_NoUDArtifacts_Bugfix.py")))

best = data %>%  group_by(Language, Model, Type) %>% summarise(Entropy = min(Entropy))
best = best %>% spread(Model, Entropy)

best %>% group_by(Type) %>% summarise(dir = mean(REAL_REAL < REVERSE, na.rm=TRUE)) 

# For each language, record whether REAL or REVERSE results in lower cross entropy
u = best %>% group_by(Language) %>% summarise(dir = mean(REAL_REAL < REVERSE, na.rm=TRUE))



