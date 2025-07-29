# Self-Attention

In tasks like language modeling or image understanding, context matters. The meaning of a word depends on other words around it; in images, the interpretation of a pixel depends on its surroundings. We will first focus on autoregressive[^1], decoder[^2] models which are trained on a task of _predicting the next token in a sequence_. During inference, the model is provided with some text, and its task is to predict how this text should continue. 

Mathematically, the goal of self-attention is to transform each input (embedded token from pretraining) into a _context vector_ which combines the information from all the inputs (embedded tokens) of your chat history so far. The word "dog" will have an initial embedding representation after pre-training (this will differ from model to model e.g. GPT-4 will have one embedding representation whereas LLaMA will have another, and starting with a clean chat we begin with this embedding), and as we go along and use the word "dog" in various contexts in our chat, we will update the ongoing representations of "dog" based on the increasing context that is provided to us.  

## Encoder-Decoder Architectures

Encoders are neural network components that transform input data into a compact representation or "encoding." They capture essential features of the input, reducing dimensionality while preserving important information. In natural language processing, encoders often process sequences of words or tokens. 

![Alt text](image.png)

[^1]:
[^2]: 
