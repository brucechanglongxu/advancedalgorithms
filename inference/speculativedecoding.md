Imagine we are using huge model like GPT-4 generating text is painfully slow because it works _one token at at a time_, for example:
```
[The] -> [The cat] -> [The cat sat] -> ...
```
Each step loads 100+ GB of model weights, runs massive matrix multiplications, and waits for the next token before moving on. 
> "We know what that the model is going to say next, can't we guess a few tokens ahead and only check if we're right?"

等事情临头了再说吧
有所不知

# [Speculative Decoding](https://arxiv.org/pdf/2211.17192)

In LLM inference, traditional decoding (e.g. greedy or beam search) is _sequential_ i.e. each token depends on the previous one which **limits parallelism**. _Speculative decoding_ changes an inherently serial problem into a parallel-friendly one. 

> Instead of decoding one token at a time, we decode k tokens in parallel, speculatively

We use a _small, fast_ model (like T5-small) to guess the next 3-8 tokens, and then run a big, slow model (like T5-XXL) once to check those guesses in parallel. We accept guesses that look good enough, and then fix mistakes with one true token from the big model. We think of the big model as a slow, meticulous writer, whilst the small model that runs ahead is a _reckless assistant_. 

Let's say that the user prompt is `"The mitochondria is the"`, we will then use a small draft model to generate `k` tokens in parallel:

```
["powerhouse", "of", "the", "cell"]
```

We then feed these tokens back into the _full model_, one at a time, and at each step, compute the full model's log-probability over tokens. We then accept the draft token if it matches the full model's top prediction, and stop at the first rejection and continue from there using the full model. we continue decoding (either with another roudn of speculative decoding or regular decoding). 

## Medusa

[MEDUSA](https://arxiv.org/pdf/2401.10774) replaces the need for an external _draft model_ in speculative decoding by attaching multiple decoding heads (Medusa Heads) directly on top of the frozen or jointly fine-tuned LLM backbone. 

> **MEDUSA** eliminates the need for a separate draft model and introduce multiple decoding heads into the _same model_ to predict multiple future tokens in parallel. This enables single-model, multi-token speculative decoding. 

One might ask, if we are speculatively decoding let's say $$7$$ different tokens from the same context $$C$$, wouldn't all $$7$$ of these tokens be the same? Indeed, this would be the case if all of the MEDUSA heads were trained in a similar fashion, however during training these heads are trained to be _position-aware_ i.e. they were trained to predict a _specific_ offset token even though all of the heads have access to the same hidden state. Because language is so structured, these heads can learn distributional patterns about what usually comes $$1, 2, 3$$ tokens later. 

> The same $$h_t$$ contains rich enough information to generate a _distribution over future_, the different heads act like different projections of that information. Think of it like looking at a weather radar and predicing the next $$7$$ hours of rain, the radar image is the same, but our models for $$1$$-hour ahead vs $$7$$ hours ahead use different heuristics and biases. 

