# SFT vs RL

## Supervised Fine-tuning

Start from a pretrained base model, collect or generate a dataset of `<prompt, response>` pairs, and tokenize both the prompt and response (merge into one sequence). 

```json
{
  "instruction": "If Alice has 3 apples and gives away 1, how many are left?",
  "output": "<step> Alice starts with 3. <step> She gives away 1. <answer> 2"
}
```

## Direct Preference Optimization

