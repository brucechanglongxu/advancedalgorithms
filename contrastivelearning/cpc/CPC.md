# Contrastive Predictive Coding (CPC) (Precursor to SimCLR, MoCo)

Contrastive Predictive Coding (CPC) is a self-supervised learning method introduced by Aaron van den Oord, Yazhe Li, and Oriol Vinyals in 2018 (DeepMind). It learns powerful latent representations by predicting the future in latent space using a contrastive loss, without relying on explicit labels. The core intuition is as follows:

_Instead of reconstructing the original data (as in autoencoders), CPC learns to predict future representations and discriminate true future samples from negatives, thereby capturing temporal or spatial dependencies_
