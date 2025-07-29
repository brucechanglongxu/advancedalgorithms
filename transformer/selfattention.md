# Deep Dive into Transformers
The transformer architecture is a neural network design that processes sequences in parallel using _self-attention_ instead of recurrent/convolutional layers. It was introduced in the seminal _"Attention Is All You Need"_ (Vaswani et al. 2017) paper originally for machine translation. The key components of a Transformer layer are:
1. **Multi-Head Self-Attention:** to relate tokens to each other.
2. **Position-Wise Feed-Forward Network:** to tranform each token's representation.
3. **Residual Connections and Layer Normalization:** to enable stable deep learning.
4. **Positional Encoding:** to inject sequence order information.
5. **Softmax Output Layer (Final):** for producing probabilities. 

Let's dive into the central question of _what actually gets learned in a transformer during pretraining?_ 

## (Multi-Head) Self-Attention

$$A(Q, K, V) = s(\frac{QK^T}{\sqrt{d_k}}) \cdot V$$

In tasks like language modeling or image understanding, context matters. The meaning of a word depends on other words around it; in images, the interpretation of a pixel depends on its surroundings. We will first focus on autoregressive[^1], decoder[^2] models which are trained on a task of _predicting the next token in a sequence_. During inference, the model is provided with some text, and its task is to predict how this text should continue. 

Mathematically, the goal of self-attention is to transform each input (embedded token from pretraining) into a _context vector_ which combines the information from all the inputs (embedded tokens) of your chat history so far. The word "dog" will have an initial embedding representation after pre-training (this will differ from model to model e.g. GPT-4 will have one embedding representation whereas LLaMA will have another, and starting with a clean chat we begin with this embedding), and as we go along and use the word "dog" in various contexts in our chat, we will update the ongoing representations of "dog" based on the increasing context that is provided to us.  

## Position-Wise Feed-Forward Networks (FFN)

## Residual Connections (Skip Connections)

## Layer Normalization

## Positional Encoding

## Softmax Output Layer

## Encoder-Decoder Architectures

Encoders are neural network components that transform input data into a compact representation or "encoding." They capture essential features of the input, reducing dimensionality while preserving important information. In natural language processing, encoders often process sequences of words or tokens. 

![Alt text](image.png)

One of the key differences between encoder-decoder and decoder-only architectures is that the former allows for _bidirectional attention_ [^3] (in the encoder), but decoder-only models are restricted to causal (left-to-right) attention.

[^0]:
[^1]:
[^2]: 
[^3]: Each token can attend to all other tokens in the same sequence, including tokens **before and after itself**. It is a full reader that can look ahead and behind the current token. This highlights why encoder-decoder models would be highly inefficient for chatbots, because every time we have a new line in our conversation, _we would need to encode the entire transcript again_ (to look "bidirectionally") before subsequently decoding to do next token generation. 
