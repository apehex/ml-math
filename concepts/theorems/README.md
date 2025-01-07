# Approximation Theorems

These theorems establish the existence of neural network approximation of functions.

---

### Notations

| Symbol                                                                    | Meaning                                                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| $f \in \mathcal{L}(\mathbb{R}^{n}, \mathbb{R}^{m})$                       | A target function                                                                 |
| $g \in \mathcal{L}(\mathbb{R}^{n}, \mathbb{R}^{m})$                       | An approximate function of the target                                             |
| $\mathcal{F} \subset \mathcal{L}(\mathbb{R}^{n}, \mathbb{R}^{m})$         | The space of target functions                                                     |
| $\mathcal{G} \subset \mathcal{L}(\mathbb{R}^{n}, \mathbb{R}^{m})$         | The space of approximate functions                                                |
| $\Vert \cdot \Vert$                                                       | A norm on $\mathcal{L}(\mathbb{R}^{n}, \mathbb{R}^{m})$                           |
| $\epsilon \in \mathcal{R}^{*+}$                                           | The approximation error between $f$ and $g$                                       |

---

### Statement

$$\begin{align}
\forall f \in \mathcal{F}, \\ \forall \epsilon \in \mathcal{R}^{*+} \\\\
\exists g \in \mathcal{G}, \\ \text{such that} \\ \Vert f - g \Vert < \epsilon
\end{align}$$

---

### Special Cases

This theorem has many variants, depending on the architecture of the approximation function and / or the target function.

In particular, the [arbitrary width](arbitrary-width.md) and [arbitrary depth](arbitrary-depth.md) variants are the theoretical foundation for the perceptron models.
