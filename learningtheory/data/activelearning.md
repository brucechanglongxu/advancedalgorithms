# [Active Learning](https://burrsettles.com/pub/settles.activelearning.pdf)

途中遇上的，总会道别，有缘再见，就此别过
文武双全
避风头
自己无法胜任这千斤重担

What should we do when we are faced with a limited about of labeled data for supervised learning tasks? We will try to involve some expert labelers when possible, but within a certain budget in order to be smart regarding which samples to label. Given an unlabeled dataset $$\mathcal{U}$$ and a fixed amount of labeling cost $$B$$, active learning techniques aim to select a subset of $$B$$ examples from $$\mathcal{U}$$ to be labeled to result in maximized improved model performance; this is particularly important in the medical field, where labeled data is costly to come by. 

We want to select the **right samples** with active learning, and the best strategies explicitly model epistemic uncertainty, enforce data diversity, or anticipate downstream model improvement. When working under tight annotation budgets (e.g. medical AI, satellite vision, robotics) active learning principles become critical. 

## Contrastive Active Learning

_Contrastive Active Learning_ (Margatina) 

[1] Margatina et al. _“Active Learning by Acquiring Contrastive Examples.”_ EMNLP 2021.
[2] Ash, Jordan et al. _"Deep Batch Active Learning by Diverse Gradient Embeddings."_ ICLR 2020
[3] Gal, Yarin, and Zoubin Ghahramani. _"Dropout as a Bayesian approximation."_ ICML 2016
