# JEPA

[V-JEPA 2](https://scontent-sjc6-1.xx.fbcdn.net/v/t39.2365-6/505938564_1062675888787033_5500377980002407548_n.pdf?_nc_cat=101&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=x_Y5GzGrhAkQ7kNvwESpSyc&_nc_oc=Adl3hNhD69_sG4PEvcclpuiQwi2up4gKSOIkuwaO_f9SIsqS6o56ad9pO8-eq9HCB5ZQrq8KorwJALGH4n3JFaf-&_nc_zt=14&_nc_ht=scontent-sjc6-1.xx&_nc_gid=Z9zdoJN26-AZX--5u7PTPw&oh=00_AfRHmWrS6wVvPrVjpVU19TjhgQy7eQwsP_1OwMfaWC_hYg&oe=688EB1F0)

At its core, JEPA is a self-supervised learning (SSL) paradigm where a model predicts abstract representations of missing or future parts of input, not the input itself. This differs from traditional SSL methods like autoencoders, masked autoencoders, or diffusion models, which reconstruct in raw pixel or input space.

> **Key idea:** JEPA predicts _representations_ of future or masked content, not the content itself. "Predict in latent space, not pixel space."

JEPA is usually composed of three modules, a context encoder $$E_c$$ which takes in _known inputs_ (e.g. visible image patches, past observations) and produces context embeddings $$s_c$$. A target encoder $$E_t$$, which takes in masked/missing/future inputs, and produces target embeddings $$s_t$$ (the "ground truth" in latent space), and updated through exponential moving average (EMA) of $$E_c$$. A predictor $$P$$, which takes $$s_c$$ and auxiliary tokens (e.g. mask tokens or latent actions) and predicts $$\hat{s}_t$$, a latent prediction of the future/missing piece. 

JEPA is useful even with failure data (e.g. robots falling over) because it focuses on _what happened_ and not _what should happen_. 

> **ACT-JEPA:** Even if a demo shows a car is driving off a cliff, the model learns _how_ the car got there, not that it's good to do so. 