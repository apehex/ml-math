# Approximation Theorem

Notations:

- $\mathcal{X} \subseteq \mathbb{R}^{n}$: A compact subset of the $\mathbb{R}^{n}$.
- $f \in C_0(\mathcal{X}, \mathbb{R}^{m})$: A continuous target function to approximate.
- $\mathcal{H}$: The hypothesis class of functions represented by the neural network.
- $\sigma: \mathbb{R} \to \mathbb{R}$: A non-linear activation function (e.g., sigmoid, ReLU).
- $\varepsilon > 0$: A small positive value representing the allowed approximation error.
- $\mathcal{N}_\sigma$: The set of functions representable by a single hidden-layer neural network with activation $\sigma$.
- $\|\cdot\|_\infty$: The uniform norm, defined as $\|f - g\|_\infty = \sup_{x \in \mathcal{X}} |f(x) - g(x)|$.

Short:

$$\begin{align}
\|f - g\|_\infty < \varepsilon \\
\|f - g\|_\infty = \max_{1 \leq i \leq m} \|f_i - g_i\|_\infty < \varepsilon
\end{align}$$

Rigorous:

$$\begin{align}
\phi_{n}\to f
\end{align}$$

Activations:

$$\begin{align}
\sigma(x) = \frac{1}{1 + e^{-x}} \\
\sigma(x) = \max(0, x)
\end{align}$$

Classes of neural networks:
