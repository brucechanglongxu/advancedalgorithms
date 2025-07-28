## KV cache
## Speculate Decoding

Speculative decoding is an inference-time acceleration technique that allows faster text generation from large language models (LLMs) by using a smaller, faster draft model to propose several tokens at once — which are then verified and accepted or rejected by a larger, accurate target model.

**Goal:** Speed up LLM inference (e.g., GPT-3, LLaMA, Mixtral) without hurting output quality.

## Memory-efficient Attention
