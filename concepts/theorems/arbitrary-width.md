# Arbitrary Width Approximation Theorem

With enough neurons, a single layer perceptron can approximate any continuous function.

## Notations

| Symbol                                                                    | Meaning                                                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| $\sigma \in \mathcal{C}_0(\mathcal{R}, \mathcal{R})$                      | A non polynomial activation function                                              |
| $\mathcal{X} \subset \mathbb{R}^{n}$                                      | A compact subset of $\mathbb{R}^{n}$                                              |
| $\mathcal{F} = \mathcal{C}_0(\mathcal{X}, \mathbb{R}^{m})$                | The space of continuous funtions on $\mathcal{X}$                                 |
| $\mathcal{G} = \mathcal{G}_{\sigma}$                                      | The space of perceptrons with $\sigma$ activation                                 |
| $\Vert \cdot \Vert = \Vert \cdot \Vert_\infty$                            | The uniform norm                                                                  |

The set of single layer perceptrons is defined as:

```math
\mathcal{G}_{\sigma} = \lbrace x \mapsto C . ( \sigma \bullet ( A . x + b ) ), \quad k \in \mathbb{N}, \quad A \in \mathbb{R}^{k \times n}, \quad b \in \mathbb{R}^{k}, \quad C \in \mathbb{R}^{m \times k} \rbrace
```

Where $\bullet$ is the element-wise application of a function, eg $\sigma$.

## Statement

```math
\forall \mathcal{X} \subset \mathcal{R}^{n} \quad \text{compact}, \quad \forall f \in \mathcal{C}_0(\mathcal{X}, \mathbb{R}^{m}), \quad \forall \epsilon \in \mathcal{R}^{*+} \\
\exists g \in \mathcal{G}_{\sigma}, \quad \text{such that} \quad \Vert f - g \Vert_{\infty} < \epsilon
```
