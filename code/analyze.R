data = read.csv("results.tsv", sep="\t")

library(dplyr)
library(tidyr)
library(lme4)
data = data %>% filter(grepl("Bugfix", Type)) %>% filter(LR < 0.1) # 0.1 was the first learning rate I tried, but it leads to bad losses on some of the larger corpora

summary(lmer(Entropy ~ Model + (1|Language), data=data %>% filter(Type == "MIWord5_NoUDArtifacts_Bugfix.py")))

best = data %>%  group_by(Language, Model, Type) %>% summarise(Entropy = min(Entropy))
best = best %>% spread(Model, Entropy)

best %>% group_by(Type) %>% summarise(dir = mean(REAL_REAL < REVERSE, na.rm=TRUE)) 

u = best %>% group_by(Language) %>% summarise(dir = mean(REAL_REAL < REVERSE, na.rm=TRUE))



