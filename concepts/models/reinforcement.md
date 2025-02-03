# Reinforcement Learning



---

### Notations

| Symbol                                                                    | Meaning                                                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| $\mathcal{S}$                                                             | The state space, both environment and agent states                                |
| $\mathcal{A}$                                                             | The action space, for the agent                                                   |
| $P\_{a}(s,s') = \Pr(S\_{t+1}=s' \mid S\_{t}=s, A\_{t}=a)$                 | The transition probability from state $s$ to $s'$ under action $a$                |
| $R\_{a}(s,s')$                                                            | The immediate reward after transition from $s$ to $s'$ with $a$                   |
| $G = \sum\_{t=0}\^{\infty} \gamma\^{t} R\_{t+1}$                          | The total **discounted** return                                                   |
| $\gamma \in [0, 1[$                                                       | The discount rate, weighting the future rewards down                              |
| $\tau = (s\_{0}, a\_{0}, s\_{1}, a\_{1}, \dots)$                          | A trajectory                                                                      |

---

### Taxonomy

![][image-taxonomy]

---

### Base RL

The probability of a given trajectory depends on both the environment and the policy:

$$\begin{align}
\Pr(\tau \mid \pi) = \rho\_{0}(s\_{0}) \prod\_{t=0}\^{T-1} \Pr(s\_{t+1} \mid s\_{t},a\_{t}) \pi(a\_{t} \mid s\_{t})
\end{align}$$

The goal of RL agents is to find a policy $\pi$ that maximizes the expected return:

$$\begin{align}
V\_{\pi}(s)
&= \mathbb{E} \left[ G \mid S\_{0} = s \right] \\\\
&= \mathbb{E} \left[ \sum\_{t=0}\^{\infty} \gamma\^{t} R\_{t+1} \mid S\_{0} = s \right]
\end{align}$$

Expressed to evaluate a specific action, as the advantage function:

$$\begin{align}
Q\_{\pi}(s, a) &= \mathbb{E} \left[ G \mid S\_{0} = s, a \right] \\\\
A\_{\pi}(s, a) &= Q\_{\pi}(s, a) - V\_{\pi}(s)
\end{align}$$

### Proximal Policy Optimization (PPO)

At step $k$, the change from the old policy $\pi\_{\theta\_{k}}$ to a new one $\pi\_{\theta}$ is quantified with:

$$\begin{align}
r\_{t}(\theta) = {\frac{\pi\_{\theta}\left( a\_{t} \mid s\_{t} \right)}{\pi\_{\theta\_{k}}\left( a\_{t} \mid s\_{t} \right)}}
\end{align}$$

And a given action is evaluated according to the current (old) policy:

$$\begin{align}
A\_{t} = A\_{\pi\_{\theta\_{k}}}(s\_{t}, a\_{t})
\end{align}$$

Then the policy is optimized in the neighborhood of its current value with the objective:

$$\begin{align}
L(\theta) = E\_{t} \left[ \min(r\_{t}(\theta) A\_{t}, clip(r\_{t}(\theta), 1 - \epsilon, 1 + \epsilon) A\_{t})) \right]
\end{align}$$

### Group Relative Policy Optimization (GRPO)

### RL From Human Feedback (RLHF)

[image-taxonomy]: .images/algorithms.svg
