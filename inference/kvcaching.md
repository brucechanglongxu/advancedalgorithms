# KV Caching

One of the most important bottlenecks in modern LLM inference is **[Key-Value (KV) caching](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management)**. In Transformer models, every token's output attends to all previous tokens through the **self-attention mechanism**. This means that for each new token during inference, the model would need to recompute attention over _all previous tokens_, which is extremely inefficient. 

KV caching avoids this recomputation. Instead of recomputing key and value matrices from scratch for each past token, we **compute them once and store/cache them**. Then, when generating teh next token, the model can just:

- Compute the _query_ for the new token.
- Reuse the cached keys/values from the past tokens.
- Compute attention using the new query and the cached key-value store.

This reduces computation from quadratic to linear in sequence length. 