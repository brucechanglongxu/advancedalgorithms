# KV Caching

One of the most important bottlenecks in modern LLM inference is **[Key-Value (KV) caching](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management)**. In Transformer models, every token's output attends to all previous tokens through the **self-attention mechanism**. This means that for each new token during inference, the model would need to recompute attention over _all previous tokens_, which is extremely inefficient. 

![Alt text](image.png)

So when a model is generating text, it looks at all the previous tokens to predict the next one, and normally it would _repeat the same calculations_ for every new token, which can slow things down. 

KV caching avoids this recomputation. Instead of recomputing key and value matrices from scratch for each past token, we **compute them once and store/cache them**. Then, when generating the next token, the model can just:

- Compute the _query_ for the new token.
- Reuse the cached keys/values from the past tokens.
- Compute attention using the new query and the cached key-value store.

This reduces computation from quadratic to linear in sequence length. 

> KV caching solves compute overlap by _remembering these calculations_ from previous steps, which can be achieved by storing the intermediate states of attention layers during inference

```
# Pseudocode for KV Caching in PyTorch
class KVCache:
    def __init__(self):
        self.cache = {"key": None, "value": None}

    def update(self, key, value):
        if self.cache["key"] is None:
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            self.cache["key"] = torch.cat([self.cache["key"], key], dim=1)
            self.cache["value"] = torch.cat([self.cache["value"], value], dim=1)

    def get_cache(self):
        return self.cache

```

**Latency-memory trade-off:** Storing data in cache uses up memory space. Systems with limited memory resources may struggle to accommodate this additional memory overhead, potentially resulting in out-of-memory errors. This is especially the case when long inputs need to be processed, as the memory required for the cache grows linearly with the input length. Another aspect to keep in mind is that the additional memory consumed by the cache is not available for storing the batches of data. As a result, one might need to reduce the batch size to keep it within the memory limits, thus decreasing the throughput of the system. We solve this through:

- Sequence truncation by limiting maximum input sequence length, thus capping the cache size at the expense of losing long-term context. 
- Reducing the number of layers or attention heads, thereby decreasing model size and cache memory requirements. 
- Quantization, to use lower-precision data types for caching to reduce memory usage. 

## Cache Invalidation

In large-scale production systems with many users the KV cache needs to be properly managed to ensure consistent/reliable response time while preventing excessive memory consumption. The two most criticla aspects of this are cache invalidation (when to clear it) and cache reuse (how to use the same cache multiple times). 

- **Session-based clearing:** We simply clear the cache at the end of a user session or conversation with the model. This simple strategy is a perfect fit for applications where conversations are short and independent of each other. Think about a customer support chatbot application in which each user session typically represents an individual conversation where the user seeks assistance with specific issues. In this context, the contents of this cache are unlikely to be needed again. Clearing the K-V cache once the user ends the chat or the session times out due to inactivity is a good choice, freeing up memory for the application to handle new users.
- **Time-to-live invalidation:** In time-to-live (TTL) invalidation, cache contents are automatically cleared after a certain period. This strategy is a good choice when the relevance of cached data diminishes predictably over time. Consider a news aggregator app that provides real-time updates. Cached keys and values might only be relevant for as long as the news is hot. Implementing a TTL policy where cached entries expire after, say, one day ensures that responses to similar queries about fresh developments are generated fast while old news doesn’t fill up memory.
- **Contextual-relevance:** Here, we clear the cache contents as soon as they become irrelevant to the current context or user interaction. This strategy is ideal when the application handles diverse tasks or topics within the same session, and the previous context doesn’t contribute value to the new one. Think about a coding assistant that works as an IDE plug-in. While the user is working on a particular set of files, the cache should be retained. As soon as they switch to a different codebase, however, the previous keys and values become irrelevant and can be deleted to free memory. Contextual relevance-based approaches might be challenging to implement, though, as they require pinpointing the event or point in time at which the context switch occurs. [This is very useful for copilot]

## Cache Reuse

Another important aspect of cache management is its reuse. On some occasions, a once-generated cache can be used again to speed up generation and save memory by avoiding storing the same data multiple times in different users’ cache instances. Cache reuse opportunities typically show up when there is shared context and/or a warm start is desirable. In scenarios where multiple requests share a common context, one can reuse the cache for that shared portion. In e-commerce platforms, certain products may have standard descriptions or specifications that are frequently asked about by multiple customers. These may include product details (“55-inch 4K Ultra HD Smart LED TV”), warranty information (“Comes with a 2-year manufacturer’s warranty covering parts and labor.”), or customer instructions (“For best results, mount the TV using a compatible wall bracket, sold separately.”). By caching the key-value pairs for these shared product descriptions, a customer support chatbot will generate responses to common questions faster. Similarly, one can precompute and cache the initial K-V pairs for frequently used prompts or queries. Consider a voice-activated virtual assistant application. Users frequently start interactions with phrases like “What’s the weather today?” or “Set a timer for 10 minutes.” The assistant can respond more quickly by precomputing and caching the key-value pairs for these frequently used queries.