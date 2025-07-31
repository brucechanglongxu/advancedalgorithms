# In Context Learning

Even though LLMs are only trained to predict the next token, that task ends up _implicitly training them on many NLP tasks_, so we can actually consider next-word prediction as multi-task learning at scale. 

| Prompt                                                       | Implied Task         |
| ------------------------------------------------------------ | --------------------- |
| “I enjoy swimming, biking, and {running, toaster}”           | Grammar / Syntax      |
| “The ingredients are flour, eggs, and {sugar, bicycle}”       | Semantic Consistency  |
| “The currency of Japan is the {yen, dollar}”                 | Factual Knowledge     |
| “The food was delicious, and the service was {excellent, rude}” | Sentiment Analysis |
| “The French word for ‘hello’ is {bonjour, adiós}”            | Translation           |
| “12 × (4 + 3) = {84, 96}”                                    | Basic Arithmetic      |
| “She picked up the violin and began to {play, drive}”        | Common Sense Inference |
| “Photosynthesis occurs in the {chloroplast, pancreas}”       | Science Knowledge     |

Because the internet contains _examples of nearly every human cognitive task expressed in language_, training on next-word prediction _forces_ the model to learn these tasks to minimize loss. So next-word prediction on a massive corpus becomes equivalent to solving thousands of diverse NLP tasks, with no labels required. 

> Learning input-output relationships can be cast as next-word prediction. This is known as in-context learning.

In-Context Learning (ICL) is the ability of large language models (LLMs) to learn new tasks at inference time by conditioning on examples provided in the input prompt, without updating any model weights. It allows LLMs to generalize to unseen tasks and domains purely through pattern recognition and attention over tokens. ICL is foundational to how models like GPT-3/4, ChatGPT, Claude, and LLaMA operate. It underpins behaviors such as few-shot learning, zero-shot generalization, instruction-following, and even tool-use — all via prompt design.

> Real in-context learning happens, but only in large-enough language models

> Some tokens require more "thinking" to let the model reason. **Not all tokens are equal**. 

"A researcher working at Google Deepmind is working on artificial ___", predicting "intelligence" is easy.

"The power of the Qi nation lies in ___", this is very uncertain, and difficult for the model to guess.

"What is [INSERT COMPLEX EQUATION HERE]" -> this requires extremely complex reasoning. 

To handle the latter two "hard" tokens, we use **Chain-of-Thought prompting** to encourage the model to reason step-by-step before giving the final answer. It mimics how humans do multi-step tasks. This leads to more advanced prompting methods like _least-to-most prompting_ and _ReAct prompting_ (combining reasoning and actions). Reasoning means that we are spending more computation/time on hard tokens, and prompting helps allocate that time. 