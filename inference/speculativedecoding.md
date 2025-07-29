# Speculative Decoding

In LLM inference, traditional decoding (e.g. greedy or beam search) is _sequential_ i.e. each token depends on the previous one which **limits parallelism**. 

```
Instead of decoding one token at a time, we decode k tokens in parallel, speculatively
```