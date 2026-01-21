| Name        | Year | Core Idea / Short Description                                                                 | Trigger Type                     | Label Flipped? | Training-time Attack | Stealth Level | Original Paper                    |
|-------------|------|------------------------------------------------------------------------------------------------|----------------------------------|----------------|----------------------|---------------|-----------------------------------|
| BadNets     | 2017 | Injects a fixed visible pattern into images; model learns to associate trigger with target label | Static patch (visible)            | Yes (to target) | Yes                  | Low           | https://arxiv.org/abs/1708.06733  |
| Blended     | 2017 | Trigger blended into entire image using low opacity to reduce visual detectability             | Global blended pattern            | Yes (to target) | Yes                  | Medium        | https://arxiv.org/abs/1712.05526  |
| SIG         | 2019 | Uses sinusoidal signal added to image to hide trigger in frequency domain                       | Signal-based (sinusoidal)         | Yes (to target) | Yes                  | Medium–High   | https://arxiv.org/abs/1902.11237  |
| ReFool      | 2020 | Physically inspired reflection patterns mimic real-world reflections                           | Natural reflection               | Yes (to target) | Yes                  | High          | https://arxiv.org/abs/2002.11230  |
| ISSBA       | 2021 | Invisible trigger embedded in latent space via steganography                                    | Invisible (latent / stego)        | Yes (to target) | Yes                  | Very High     | https://arxiv.org/abs/2103.04047  |
| WaNet       | 2021 | Non-additive spatial warping creates trigger without introducing pixel artifacts               | Spatial warping (non-additive)    | Yes (to target) | Yes                  | Very High     | https://arxiv.org/abs/2102.10369  |
| TrojanNN    | 2018 | Inserts malicious neurons activated by rare internal patterns                                  | Internal neuron activation        | Yes (to target) | Yes                  | High          | https://arxiv.org/abs/1804.00792  |
| Clean-Label | 2019 | Backdoor without changing labels; poisons data so trigger causes misclassification             | Visible / invisible               | No             | Yes                  | High          | https://arxiv.org/abs/1901.02217  |
| Invisible Trigger | 2020 | Optimizes imperceptible perturbations constrained by human vision                               | Imperceptible noise               | Yes (to target) | Yes                  | Very High     | https://arxiv.org/abs/1911.10347  |
| Dynamic Backdoor | 2021 | Trigger generated dynamically by a neural network instead of fixed pattern                     | Input-conditioned                 | Yes (to target) | Yes                  | Very High     | https://arxiv.org/abs/2108.00673  || MakeupAttack | 2021 | Semantic clean-label backdoor using facial makeup styles as trigger; no artificial patterns | Semantic (makeup / appearance) | No | Yes | Very High | https://arxiv.org/abs/2105.10973 |
| MakeupAttack | 2024 | Semantic clean-label backdoor using facial makeup styles as trigger; no artificial patterns | Semantic (makeup / appearance) | No | Yes | Very High | https://arxiv.org/pdf/2408.12312  |


Backdoor attacks - Survey \
https://arxiv.org/pdf/2509.07504

How transferable are features in deep neural networks? \
https://arxiv.org/pdf/1411.1792

Catastrophic Forgetting of the backdoor

