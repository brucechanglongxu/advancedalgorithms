# Inductive Bias

Inductive bias refers to the assumptions a learning algorithm makes to generalize from limited data. No model can learn from scratch without some prior assumptions. These assumptions act as shortcuts that help the model:
- Generalize beyond the training data
- Learn more efficiently
- Avoid overfitting
In simple terms:
```
An inductive bias tells the model what patterns to expect before it even sees any data.
```
When we design a new architecture, we’re implicitly or explicitly injecting a belief about the structure of the input space. Good inductive biases:
- Help generalize from fewer examples
- Reduce parameter count
- Improve interpretability or robustness
Inductive biases can come from _architecture_[^1], _data_, _training objective_ and even _optimization_. 

[^1]: Transformer architectures have minimal inductive bias compared to other architectures e.g. CNNs; they were designed to be as general and assumption-free as possible (especially compared to CNNs/RNNs). Inductive biases are "assumptions built into the model that shape how it learns". In neural networks, this often shows up in architectural structure, constraints on parameter sharing, and input/output flow (causality, locality, order etc.). Unlike RNNs, Transformers **do not assume tokens come in a strict order** they use self-attention where every token can attend to every other (regardless of position), and thus no built-in notion of _temporal dependence_ is needed (only through positional embeddings). Whilst this makes them very flexible for text, code, vision and more - they require **more data** to learn structure (huge pretraining corpora). 