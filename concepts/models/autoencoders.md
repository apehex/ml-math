# Autoencoders

Before learning the mapping from input to target, a model can learn a compressed encoding of the input.

This training can be self-supervised, and the resulting encodings can be the basis of further models.

These encodings have both lower dimensionality and meaning embedded geometrically.

## Notations

| Symbol                                                                    | Meaning                                                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| $r$                                                                       | The rank of all inputs (number of axes)                                           |
| $n = \vert \mathcal{D} \vert$                                             | The cardinality of $\mathcal{D}$                                                  |
| $\mathcal{D} = \lbrace \mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n)} \rbrace$ | The original dataset, composed of $n$ inputs                                      |
| $\mathbf{x}^{(k)} = [x^{(k)}\_1, x^{(k)}\_2, \dots, x^{(k)}\_{r}]$        | An original sample                                                                |
| $\mathbf{x}$                                                              | An arbitrary input of the model                                                   |
| $\mathbf{\ddot{x}}$                                                       | A corrupted input                                                                 |
| $\mathbf{\hat{x}}$                                                        | A reconstruction of some input                                                    |
| $\mathbf{z}$                                                              | A latent encoding of some input                                                   |
| $g_{\phi}(.)$                                                             | The encoding function parametrized by $\phi$                                      |
| $f_{\theta}(.)$                                                           | The decoding function parametrized by $\theta$                                    |
| $q_{\phi}(\mathbf{z} \vert \mathbf{x})$                                   | Estimated posterior probability function, also known as probabilistic encoder     |
| $q_{\theta}(\mathbf{x} \vert \mathbf{z})$                                 | Likelihood of generating true data sample given the latent code, also known as probabilistic decoder. |

## Base Autoencoders (AE)

The model is trained to reconstruct the inputs:

$$\begin{align}
\mathcal{D}\_{T} = \lbrace (\mathbf{x}^{(1)}, \mathbf{x}^{(1)}), \dots, (\mathbf{x}^{(n)}, \mathbf{x}^{(n)}) \rbrace
\end{align}$$

It learns to reduce the dimensionality:

$$\begin{align}
\mathbf{z} &= g_{\phi}(\mathbf{x}) \\\\
\mathbf{\hat{x}} &= f_{\theta}(\mathbf{z})
\end{align}$$

While minimizing the error:

$$\begin{align}
L\_\text{AE}(\theta, \phi) = \frac{1}{n} \sum\_{i=1}^{n} (\mathbf{x}^{(i)} - f\_{\theta}(g\_{\phi}(\mathbf{x}^{(i)})))^2
\end{align}$$

## Denoising Autoencoders (DAE)

A DAE is actually a regular AE operating on scrambled inputs:

$$\begin{align}
\mathcal{D}\_{T} = \lbrace (\mathbf{\ddot{x}}^{(1)}, \mathbf{x}^{(1)}), \dots, (\mathbf{\ddot{x}}^{(n)}, \mathbf{x}^{(n)}) \rbrace
\end{align}$$

$$\begin{align}
\mathbf{z} &= g_{\phi}(\mathbf{\ddot{x}}) \\\\
\mathbf{\hat{x}} &= f_{\theta}(\mathbf{z})
\end{align}$$

$$\begin{align}
L\_\text{DAE}(\theta, \phi) = \frac{1}{n} \sum\_{i=1}^{n} (\mathbf{x}^{(i)} - f\_{\theta}(g\_{\phi}(\mathbf{\ddot{x}}^{(i)})))^2
\end{align}$$

## Variational Autoencoders (VAE)

### Re-parametrization

