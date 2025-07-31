# [CoCA: Contrastive Captioners are Image-Text Foundation Models](https://research.google/blog/image-text-pre-training-with-contrastive-captioners/)

<img width="1999" height="594" alt="image" src="https://github.com/user-attachments/assets/9177f9df-ca47-4a42-a79f-fc3638fcc4cf" />

CoCA is a unified vision-language foundaiton model that combines contrastive learning (e.g. CLIP/ALIGN) for zero-shot retrieval and classification, with Captioning loss (like SimVLM) for strong multimodal generation and VQA. It introduces a two-part decoder on top a Vision Transformer (ViT) encoder):

- A unimodal text decoder for contrastive learning (no cross-attention)
- A multimodal decoder for image-conditioned captioning (with cross attention)

This architecture allows _joint training_ using both contrastive and generative losses in _one forward/backward pass_, resulting in a single backbone that is effective for classification, retrieval, VQA [^1], captioning and video understanding. By skipping the cross-attention in the first decoder and enabling it in the second, CoCa learns unimodal representations for alignment, and joint multimodal representatiosn for captioning. **Co-training both decoupled representations gives disentangled yet synergistic feature learning**. 

[^1]: We are given an image, and a natural language question about that image, and it must generate the correct answer, typically in free-form text or multiple choice. VQA is a cornerstone task in vision-language research because it tests a model's ability to jointly understand visual scenes and natural language, requiring not just perception but also reasoning, grounding, and sometimes external knowledge. It serves as a real-world proxy for building AI systems that can interact naturally with humans, such as assistive agents for the visually impaired, intelligent robotics, or interactive tutoring systems. The multimodal nature of VQA makes it a rigorous benchmark for evaluating general-purpose foundation models like CoCa, which must flexibly process and align both text and images.
