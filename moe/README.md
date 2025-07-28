# MoE: Scalable Sparse Models


<img width="1964" height="990" alt="image" src="https://github.com/user-attachments/assets/1505cd95-18a2-4de8-be9d-e6174e36cdc8" />


The Mixture-of-Experts (MoE) approach attracts a lot of attention recently as researchers (mainly from Google) try to push the limit of model size. The core of the idea is (ensemble learning), where _combining multiple weak learners gives us a strong learner_. Within one deep neural network, ensembling can be implemented with a gating mechanism connecting multiple experts (Shazeer et al. 2017). The gating mechanism controls which subset of the network (e.g. which experts) should be activated to produce outputs. 
