# Diffusion

A model can learn to denoise increasingly altered data, until it is pure noise.

Then, noise samples can be used to generate new data.

---

## Notations

| Symbol                                                                    | Meaning                                                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| $\mathcal{N}(\mu, \Sigma)$                                                | The normal distribution, with mean $\mu$ and variance $\Sigma$                    |
| $q$                                                                       | The distribution of the real data                                                 |
| $p\_{\theta$                                                              | The approximation learnt by the model $\theta$                                    |
| $T$                                                                       | The number of diffusion steps                                                     |
| $N$                                                                       | The cardinality of $\mathcal{D}$                                                  |
| $\mathcal{D} = \lbrace \mathbf{x}\^{(1)},\dots,\mathbf{x}\^{(N)} \rbrace$ | The original dataset, composed of $N$ inputs                                      |
| $\mathbf{x}\_{0}$                                                         | One sample from the dataset                                                       |
| $\mathbf{x}\_{t}$                                                         | The sample $x\_{0}$ after $t$ diffusion steps                                     |
| $\beta\_{t}$                                                              | The power / variance of the iterative noise, at diffustion step $t$               |
| $\alpha\_{t} = 1 - \beta\_{t}$                                            | The power of the iterative signal, at diffusion step $t$                          |
| $\bar{\alpha}\_{t} = \alpha\_{1} \cdots \alpha\_{t}$                      | The power of the original signal in $\mathbf{x}\_{t}$                             |
| $\bar{\beta}\_{t} = 1 - \bar{\alpha}\_{t}$                                | The power / variance of the cumulative noise                                      |
| $\sigma\_{t} = \sqrt{\bar{\beta}\_{t}} = \sqrt{1 - \bar{\alpha}\_{t}}$    | The standard deviation of the cumulative noise                                    |

---

## Forward Diffusion

Successive samples of gaussian noise are added to the original data $\mathbf{x}\_{0}$, with predefined amplitudes:

$$\begin{align}
\mathbf{x}\_{0} \sim q \\\\ 
\mathbf{x}\_{t} &= {\sqrt {1 - \beta\_{t}}} \mathbf{x}\_{t-1} + {\sqrt {\beta\_{t}}} \mathbf{z}\_{t} \\\\
\end{align}$$

The overall diffusion is a gaussian process:

$$\begin{align}
q(\mathbf{x}\_{0:T}) &= q(\mathbf{x}\_{0}) q(\mathbf{x}\_{1} \vert \mathbf{x}\_{0}) \cdots q(\mathbf{x}\_{T} \vert \mathbf{x}\_{T-1}) \\\\
                     &= q(\mathbf{x}\_{0}) \mathcal{N}(\mathbf{x}\_{1} \vert {\sqrt {\alpha\_{1}}} \mathbf{x}\_{0}, \beta\_{1} \mathbf{I}) \cdots \mathcal{N}(\mathbf{x}\_{T} \vert {\sqrt {\alpha\_{T}}} \mathbf{x}_{T-1}, \beta\_{T} \mathbf{I})
\end{align}$$

The cumulative noise up to the step $t$ is also gaussian:

$$\begin{align}
\mathbf{x}\_{t} \vert \mathbf{x}\_{0} \sim \mathcal{N} \left( \sqrt{{\bar{\alpha}}\_{t}} \mathbf{x}\_{0}, \sigma\_{t}\^{2} \mathbf{I} \right)
\end{align}$$

So the sum of all the noise can be directly added to $\mathbf{x}\_{0}$ to form the input of the model $\mathbf{x}\_{t}$.

---

## Backward Diffusion

Predicting the cumulative noise $\mathbf{\epsilon}\_{\theta}(\mathbf{x}\_{t}, t)$ allows to estimate $\mathbf{x}\_{0}$:

$$\begin{align}
\mathbf{x}\_{0} \approx \frac{\mathbf{x}\_{t} - \sigma\_{t} \mathbf{\epsilon}\_{\theta}(\mathbf{x}\_{t}, t)}{\sqrt{{\bar{\alpha}}\_{t}}}
\end{align}$$

Then, the model can reverse the latest diffusion step and estimate the parameters of $\mathbf{x}\_{t-1}$:

$$\begin{align}
\mu\_{\theta}(\mathbf{x}\_{t},t) &= {\tilde{\mathbf{\mu}}}\_{t} \left( \mathbf{x}\_{t}, \frac{\mathbf{x}\_{t} - \sigma\_{t} \mathbf{\epsilon}\_{\theta}(\mathbf{x}\_{t}, t)}{\sqrt{{\bar{\alpha}}\_{t}}} \right)
                                 &= {\frac{\mathbf{x}\_{t} - \mathbf{\epsilon}\_{\theta}(\mathbf{x}\_{t}, t) \beta\_{t} / \sigma\_{t}}{\sqrt{\alpha\_{t}}}}
\end{align}$$

And iterate until it composes an estimation of $\mathbf{x}\_{0}$.

---

## Base Diffusion Model

### Dataset

A model could be trained on all diffusion steps for each dataset sample.

In practice, a random diffusion step $t\^{(i)}$ is sampled for each data point, along with a random noise $\mathbf{z}\^{(i)}$:

### Computation

### Loss

$$\begin{align}
L\_\text{DDM} = E_{\mathbf{x}\^{(i)}\_{0} \sim \mathcal{D}; \mathbf{z}\^{(i)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \left \Vert \epsilon\_{\theta}(\mathbf{x}\^{(i)}\_{t\^{(i)}}, t\^{(i)}) - \mathbf{z}\^{(i)} \right \Vert\^{2} \right]
\end{align}$$

---
