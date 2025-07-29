# Inductive Bias

Inductive bias refers to the assumptions a learning algorithm makes to generalize from limited data. No model can learn from scratch without some prior assumptions. These assumptions act as shortcuts that help the model:
- Generalize beyond the training data
- Learn more efficiently
- Avoid overfitting
In simple terms:
```
An inductive bias tells the model what patterns to expect before it even sees any data
```
When we design a new architecture, we’re implicitly or explicitly injecting a belief about the structure of the input space. Good inductive biases:
- Help generalize from fewer examples
- Reduce parameter count
- Improve interpretability or robustness
Inductive biases can come from _architecture_, _data_, _training objective_ and even _optimization_. 