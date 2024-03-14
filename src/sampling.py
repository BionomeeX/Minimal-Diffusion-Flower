"""_summary_
Sampling algorithm for basic diffusion
"""

import numpy as np
import matplotlib.pyplot as plt
from denoiser import Denoiser


T = 1000
beta_t = np.linspace(10e-4, 0.02, T)


alpha_t = 1 - beta_t
sigma2_t = beta_t

C1 = 1.0 / np.sqrt(alpha_t)
C2 = (1.0 - alpha_t) / (np.sqrt(1.0 - np.cumprod(alpha_t)))
sigma_t = np.sqrt(sigma2_t)

denoiser = Denoiser.load("models/save")

# initial latent
latent = np.random.normal(0, 1, (256, 256, 3))

for t in range(T)[::-1]:
    z = np.random.normal(0, 1, (256, 256, 3)) * (t != 0)
    latent = C1[t] * (latent - C2[t] * denoiser(latent, t)) + sigma_t[t] * z

image = np.asarray(latent * 127.5 + 127.5, np.uint8)

plt.imshow(image)
plt.show()
