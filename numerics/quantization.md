# Quantization

Quantization refers to converting higher-precision data types into lower-precision formats. _Lower-precision types mean fewer bytes per element_. Reducing bytes per element will increase **arithmetic intensity[^1]**. This can turn memory-bound ops into math-bound, if our math bandwidth can keep up[^2].

[^1]: Recall that arithmetic intensity is defined as FLOPs/bytes. Intuitively, it answers the question _for every byte that is loaded from memory, how much math is done before I need another one?_. It is a property at the algorithm/kernel level and not a function of the hardware (e.g. a single SM block). It doesn't matter how many SMs are used during the computation, the ratio is computed globally for the entire kernel launch. 
[^2]: ReLU is traditionally a memory bound operation in FP16, but it could become _less so_ in INT8 not because it does more math, but because it is moving fewer bytes. 
