# KL Divergence

This statistical distance measures how a model probability distribution $Q$ differs from a true distribution $P$.

---

### Notations

| Symbol                                                                    | Meaning                                                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| $X$                                                                       | A random variable, the modelization target                                        |
| $\mathcal{X}$                                                             | The support of $X$                                                                |
| $P$                                                                       | A true probability distribution for $\mathcal{X}$                                 |
| $Q$                                                                       | A model / approximate probability distribution for $\mathcal{X}$                  |

---

### KL Divergence

$$\begin{align}
D\_{\text{KL}} (P \parallel Q) = \mathrm{E}\_{\sim P} \left[ \log \left( \frac{\\ P(x) \\ }{Q(x)} \right) \right]
\end{align}$$

For discrete variables:

$$\begin{align}
D\_{\text{KL}} (P \parallel Q) = \sum\_{x\in \mathcal{X}} P(x) \\ \log \left( \frac{\\ P(x) \\ }{Q(x)} \right)
\end{align}$$

For continuous variables:

$$\begin{align}
D\_{\text{KL}} (P \parallel Q) = \int\_{\mathcal{X}} P(x)  \\ \log \left( \frac{\\ P(x) \\ }{Q(x)} \right) \\ \mathrm{d} x
\end{align}$$

---

### Relation To Cross Entropy

$$\begin{align}
D\_{\text{KL}} (P \parallel Q) = \mathrm{H} (P, Q) - \mathrm{H} (P, P)
\end{align}$$

See [this doc](./cross-entropy.md).
