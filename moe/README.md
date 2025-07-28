# MoE: Scalable Sparse Models


<img width="1964" height="990" alt="image" src="https://github.com/user-attachments/assets/1505cd95-18a2-4de8-be9d-e6174e36cdc8" />


The Mixture-of-Experts (MoE) approach attracts a lot of attention recently as researchers (mainly from Google) try to push the limit of model size. The core of the idea is (ensemble learning), where _combining multiple weak learners gives us a strong learner_. Within one deep neural network, ensembling can be implemented with a gating mechanism connecting multiple experts (Shazeer et al. 2017). The gating mechanism controls which subset of the network (e.g. which experts) should be activated to produce outputs. One MoE layer contains:

1. $$n$$ feedforward networks as experts $$\{E_i\}_{i=1}^n$$
2. A trainable gating network $$G$$ to learn a probability distribution over $$n$$ experts so as to route the traffic to a few selected experts

Depending on the gating outputs, not every expert has to be evaluated. When the number of experts is too large, we can consider using a two-level hierarchical MoE. 

<img width="1040" height="960" alt="image" src="https://github.com/user-attachments/assets/b3a3b66c-4c19-4e56-b019-4ecaabd4fa7f" />

