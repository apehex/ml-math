# (Cross) Entropy

---

### Notations

| Symbol                                                                    | Meaning                                                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| $X$                                                                       | A random variable, the modelization target                                        |
| $\mathcal{X}$                                                             | The support of $X$                                                                |
| $E$                                                                       | An event drawn from $\mathcal{X}$                                                 |
| $P$                                                                       | A true probability distribution for $X$                                           |
| $Q$                                                                       | A model / approximate probability distribution for $X$                            |

---

### Information Content

$$\begin{align}
I(E) = \log\_{2} \left( \frac{1}{P(E)} \right)
\end{align}$$

---

### Cross Entropy

$$\begin{align}
H(P, Q) = -\mathrm{E}\_{\sim P} \left[ \log Q \right]
\end{align}$$

For discrete variables:

$$\begin{align}
H(P, Q) = -\sum\_{x \in \mathcal{X}} P(x) \\ \log Q(x)
\end{align}$$

And continuous variables:

$$\begin{align}
H(P, Q) = -\int\_{\mathcal{X}} P(x) \\ \log Q(x) \\ \mathrm{d} x
\end{align}$$

---

### Computation

Often used to compare the predicted probabilities of the output with the target output:

- $\mathcal{X}$ is the set of possible outputs, for example the IDs of text tokens
- $P$ assigns probability:
    - $1 - \epsilon$ to the target output
    - $\epsilon$ for all the other outputs
    - the smoothing $\epsilon$ is a small constant or zero
- $Q$ is the probability distribution estimated by the model
