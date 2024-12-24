# Arbitrary Depth Approximation Theorem

With enough layers, a fully connected ReLU network can approximate any Bochner–Lebesgue p-integrable function.

## Notations

| Symbol                                                                    | Meaning                                                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| $\sigma: x \mapsto 0.5 (x + \vert x \vert)$                               | The ReLU function                                                                 |
| $\mathcal{F} = \mathcal{L}_p(\mathbb{R}^{n}, \mathbb{R}^{m})$             | The space of Bochner–Lebesgue p-integrable function                               |
| $\mathcal{G} = \mathcal{G}_{d}$                                           | The space of multi-layer ReLU nertworks of width $d$                              |
| $\Vert \cdot \Vert = \int_{\mathbb{R}^{n}} \Vert f(x) \Vert^{p} dx$       | The norm p                                                                        |

## Statement

$$
\begin{align}
\forall f \in \mathcal{L}_{p}(\mathbb{R}^{n}, \mathbb{R}^{m}), \\ \forall \epsilon \in \mathcal{R}^{*+} \\\\
\exists g \in \mathcal{G}_{max(n+1, m)}, \\ \text{such that} \\ \int_{\mathbb{R}^{n}} \Vert f(x) - g(x) \Vert^{p} dx < \epsilon
\end{align}
$$

a