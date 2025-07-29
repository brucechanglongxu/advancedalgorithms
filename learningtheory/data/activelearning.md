# [Active Learning](https://burrsettles.com/pub/settles.activelearning.pdf)

途中遇上的，总会道别，有缘再见，就此别过
文武双全
避风头
自己无法胜任这千斤重担
有什么资格说你不愿意
连你自己身上的责任都看不到
永世不能再见到天日
白纸黑字写下来
发誓你再也不会出手
你觉得我是乱世之因？
开张酬宾
你这有点敷衍了
杀机暗藏

What should we do when we are faced with a limited about of labeled data for supervised learning tasks? We will try to involve some expert labelers when possible, but within a certain budget in order to be smart regarding which samples to label. Given an unlabeled dataset $$\mathcal{U}$$ and a fixed amount of labeling cost $$B$$, active learning techniques aim to select a subset of $$B$$ examples from $$\mathcal{U}$$ to be labeled to result in maximized improved model performance; this is particularly important in the medical field, where labeled data is costly to come by. 

We want to select the **right samples** with active learning, and the best strategies explicitly model epistemic uncertainty, enforce data diversity, or anticipate downstream model improvement. When working under tight annotation budgets (e.g. medical AI, satellite vision, robotics) active learning principles become critical. 

## Uncertainty Sampling

Suppose that we have a model trained on a small seed set or pseudo-labels, and we're working in a low-label-budget setting (e.g. medical/surgical video). The basic assumption of uncertainty sampling is that if the model is unsure, then the sample is likely close to a decision boundary - so labeling it will help the model the most. 

> **Key Idea:** Label the examples that the model is most unsure about. 

## Diversity Sampling

## Contrastive Active Learning

_Contrastive Active Learning_ (Margatina) relies on contrastive learning principles to find representative samples - it focuses on finding data points that are _semantically close in representation space, **yet belong to different classes**_ as the most informative samples for training. 

In deep learning, samples from different classes can sometimes have _overlapping_ or _ambiguous_ representations in the latent feature space (for example embeddings from the penultimate layer of a neural network), these are the examples that the model is most confused about, even if it is confident **(in other words, the representation/feature space is not perfect, and points from different classes can be proximal in this space)**. These points are _exactly_ where the model is confused/incorrect about, and hence should be relabeled / classified by human labelers. 

> **Key Idea:** If an unlabeled sample looks very similar (in feature space) to labeled examples, but the model predicts something (class) quite different for it - then that sample is likely to be a contrastive, **high-value** acquisition. The core premise for this is that _feature space is not the same as prediction space_ (because the model is not perfect, especially early on in training). 

1. For each unlabeled sample, compute its feature embedding (e.g. through a ResNet or ViT)
2. Find its nearest labeled neighbors in the feature space
3. Compare the model's predictions on the unlabeled sample vs. neighbors through KL divergence _(this tells us how different our model's predictions are for similar-looking samples)_. High KL means High contrastiveness, and higher labeling priority. 

[1] Margatina et al. _“Active Learning by Acquiring Contrastive Examples.”_ EMNLP 2021.

[2] Ash, Jordan et al. _"Deep Batch Active Learning by Diverse Gradient Embeddings."_ ICLR 2020

[3] Gal, Yarin, and Zoubin Ghahramani. _"Dropout as a Bayesian approximation."_ ICML 2016
