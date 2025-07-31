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

Inductive biases can come from _architecture_[^1], _data_, _training objective_ and even _optimization_. As mentioned below, because transformers have minimal inductive bias in their architecture (and hence require huge text corpora to learn structure in the data), researchers often _reintroduce_ [^2] inductive bias into Transformers for specific gains. Precisely because the Transformer contains minimal architectural inductive bias (there is little ordering mandate on input token (aside from positional embeddings), no locality (global self-attention), and no structure), it can generalize to many data modalities, but therefore also requires large-scale pre-training of data corpora. 

## Verifier's Law 

> The ease of training AI to solve a task is proportional to how verifiable the task is. 

To train AI effectively, tasks must be objectively correct (clear ground truth), quick to verify, scalable in verification, low in noise (good signal quality), and continuously rewarded (not just binary correctness). Most major AI benchmarks succeed because they fit these criteria. For example in the case of Google's AlphaEvolve, this was a system that excels at _guess-and-check_ optimization, and solves constrained mathematical problems (e.g. packing hexagons), which shows that solvable + verifiable means soon-to-be-solved by AI. AI will rapidly conquer verifiable tasks, but lag behind on fuzzy, hard-to-verify ones, which leads to a "jagged edge" of intelligence, which is a sharp AI competence in some areas and weak in others. The future belogns to tasks we can define and measure clearly. 

Some examples of verification asymmetry:
- **Sudoku/Crosswords:** Hard to solve, easy to verify
- **Engineering Instagram:** Hard to build, easy to test
- **Inverse Asymmetry:** Fact-checking essays are harder to verify than to generate

We can reduce the latter problem by for example providing answer keys (for math problems), using unit tests or test cases for programming, and supplying reference datasets or lookup lists. In essence, through solutions via. data engineering or reframing the problem, we can shift a problem to be more susceptible to verifier's law. 

**GitHub Copilot** excels at tasks where _verification is cheap and well-degined_ such as completing boilerplate coding, suggesting syntax-correct functions, and rewriting code to match comments or test cases. Copilot works well because there is objective truth (either compiles or not), fast verification (instantly run a linter), scalable (validate many completions automatically) ,low noise (passing test are a strong signal of correctness) and continuous reward (some completions are more optimal, elegant, or idiomatic than others, giving us a gradient to learn from). 

Verifier’s Law holds because modern AI systems—especially those based on deep learning and reinforcement learning—require fast, accurate, and scalable feedback to learn effectively. Tasks that are easy to verify provide rich and frequent optimization signals, enabling models to take many low-noise gradient steps and iterate rapidly. When verification is objective, quick, and scalable, it aligns perfectly with the architecture of current AI training pipelines, allowing for massive parallelization and efficient learning. In contrast, tasks with subjective judgments, sparse rewards, or expensive evaluation bottlenecks offer poor training signals and remain difficult for AI. Ultimately, verifiability governs the tractability of optimization, making it the key driver of where and how fast AI can make progress.Ask ChatGPT

[^1]: Transformer architectures have minimal inductive bias compared to other architectures e.g. CNNs; they were designed to be as general and assumption-free as possible (especially compared to CNNs/RNNs). Inductive biases are "assumptions built into the model that shape how it learns". In neural networks, this often shows up in architectural structure, constraints on parameter sharing, and input/output flow (causality, locality, order etc.). Unlike RNNs, Transformers **do not assume tokens come in a strict order** they use self-attention where every token can attend to every other (regardless of position), and thus no built-in notion of _temporal dependence_ is needed (only through positional embeddings). Whilst this makes them very flexible for text, code, vision and more - they require **more data** to learn structure (huge pretraining corpora). 

[^2]: Relative positional encoding introduces temporal structure, Vision Transformer (with CNN stem) introduces locality/low-level features for more efficient early representations, sparse attention patterns reintroduce locality/block bias and reduce compute, graph transformers reintroduce node neighborhoods (and encode graph structure). 