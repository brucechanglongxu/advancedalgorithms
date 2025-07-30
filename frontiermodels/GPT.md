# GPT

## Number of Transformer Layers

| Model                  | Release Year | Params       | Layers (Depth) | Hidden Size | Notes                             |
|------------------------|--------------|--------------|----------------|-------------|------------------------------------|
| GPT-1                  | 2018         | 117M         | 12             | 768         | BookCorpus, causal LM              |
| BERT-Large             | 2018         | 340M         | 24             | 1024        | Bidirectional MLM                  |
| GPT-2 (XL)             | 2019         | 1.5B         | 48             | 1600        | WebText, causal                    |
| T5-11B                 | 2020         | 11B          | 24 encoder + 24 decoder | 1024  | Text-to-text, encoder-decoder     |
| GPT-3                  | 2020         | 175B         | 96             | 12288       | 96 heads                           |
| Jurassic-1 Jumbo       | 2021         | 178B         | 96             | 12288       | AI21 Labs                          |
| Gopher                 | 2021         | 280B         | 80             | 16384       | DeepMind                           |
| Chinchilla             | 2022         | 70B          | 80             | 8192        | Training-optimal, DeepMind        |
| PaLM-540B              | 2022         | 540B         | 118            | 18432       | Google, sparse MoE                |
| OPT-175B               | 2022         | 175B         | 96             | 12288       | Meta, open weights                 |
| BLOOM-176B             | 2022         | 176B         | 70             | 14336       | Multilingual                       |
| LLaMA-65B              | 2023         | 65B          | 80             | 8192        | Meta, tokenizer-efficient          |
| Falcon-180B            | 2023         | 180B         | 80             | 12288       | TII, trained on RefinedWeb        |
| GPT-4 (est.)           | 2023         | ~1T (MoE)    | ~120–150       | ~16384      | Mixture-of-Experts, private        |
| Claude 2.1             | 2023         | est. ~70–100B| ~80–100        | Unknown     | Context: 200K tokens               |
| Gemini 1 (Pro)         | 2023         | est. 60–100B | est. 80–100    | Unknown     | Google DeepMind                    |
| Mistral-7B             | 2023         | 7.3B         | 32             | 4096        | Grouped-query attention            |
| Mixtral (MoE)          | 2023         | 12.9B active | 56             | 4096        | 2-of-8 experts active              |
| Phi-2                  | 2023         | 2.7B         | 32             | 2560        | Small but highly efficient         |
| GPT-4 Turbo (est.)     | 2024         | ?            | ?              | ?           | 128K context, fast + cheap         |
| Gemini 1.5 (Flash)     | 2024         | ?            | ?              | ?           | 1M-token context                   |
| Claude 3.5 Sonnet      | 2024         | ?            | ?              | ?           | Released June 2024                |
| GPT-5 (rumored)        | 2025         | >1T?         | >150?          | >16K?       | Expected to be MoE + vision       |

## GPT-1 

Published in 2018, GPT-1 was a decoder-only transformer with 117M parameters and 12 Transformer Layers. The dimension of the token embedding space was $$768$$, the number of attention heads is $$12$$