# Self-Attention

In tasks like language modeling or image understanding, context matters. The meaning of a word depends on other words around it; in images, the interpretation of a pixel depends on its surroundings. We will first focus on autoregressive[^1], decoder[^2] models which are trained on a task of _predicting the next token in a sequence_. During inference, the model is provided with some text, and its task is to predict how this text should continue. 

## Encoder-Decoder Architectures

Encoders are neural network components that transform input data into a compact representation or "encoding." They capture essential features of the input, reducing dimensionality while preserving important information. In natural language processing, encoders often process sequences of words or tokens.

![Alt text](image.png)

[^1]:
[^2]: 
