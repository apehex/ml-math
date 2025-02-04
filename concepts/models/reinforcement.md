# Reinforcement Learning



---

### Notations

| Symbol                                                                    | Meaning                                                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| $T \in [0, \infty]$                                                       | The time horizon, which can be infinite                                           |
| $\gamma \in [0, 1[$                                                       | The discount rate, weighting the future rewards down                              |
| $\epsilon \in [0, 1[$                                                     | The clipping rate, around $0.1$                                                   |
| $\mathcal{S}$                                                             | The state space, both environment and agent states                                |
| $\mathcal{A}$                                                             | The action space, for the agent                                                   |
| $P\_{a}(s,s') = \Pr(S\_{t+1}=s' \mid S\_{t}=s, A\_{t}=a)$                 | The transition probability from state $s$ to $s'$ under action $a$                |
| $R\_{a}(s,s')$                                                            | The immediate reward after transition from $s$ to $s'$ with $a$                   |
| $G = \sum\_{t=0}\^{\infty} \gamma\^{t} R\_{t+1}$                          | The total discounted return                                                       |
| $\pi\_{\theta}$                                                           | The agent's policy, optionally parametrized by $\theta$                           |
| $\tau = (s\_{0}, a\_{0}, s\_{1}, a\_{1}, \dots)$                          | A trajectory, as a flattened sequence of state-action pairs                       |

---

### Taxonomy

![][image-taxonomy]

---

### Base RL

The probability of a given trajectory depends on both the environment and the policy:

$$\begin{align}
\Pr(\tau \mid \pi) = \rho\_{0}(s\_{0}) \prod\_{t=0}\^{T} \Pr(s\_{t+1} \mid s\_{t},a\_{t}) \pi(a\_{t} \mid s\_{t})
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

### Actor-Critic RL

These methods estimate both the policy and the value functions:

- the actor's distribution $\pi\_{\theta}(a \mid s)$ rules the selection of actions
- the critic's evaluation $V\_{\psi}(s)$ guides the actor

#### Critic Optimization

The parameters of the critic are optimized according to the temporal difference (TD) loss:

$$\begin{align}
L(\psi) = \mathbb{E} \left[ (V\_{\psi}(s\_{t}) - (R\_{t} + \gamma V\_{\psi}(s\_{t+1}))\^{2} \right]
\end{align}$$

The precision can be pushed further:

$$\begin{align}
L(\psi) = \mathbb{E} \left[ (V\_{\psi}(s\_{t}) - (\sum\_{k=0}\^{n-1} R\_{t+k} + \gamma\^{n} V\_{\psi}(s\_{t+n})))\^{2} \right]
\end{align}$$

#### Actor Optimization

The parameters of the policy are updated according to the gradient update rule:

$$\begin{align}
\nabla\_{\theta}J(\theta)
&= \mathbb{E}\_{\pi\_{\theta}} \left[ \sum\_{t=0}\^{T} \nabla\_{\theta} \ln \pi\_{\theta }(a\_{t} \mid s\_{t}) \cdot A\_{t } \right]
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
L(\theta) = \mathbb{E}\_{t} \left[ \min(r\_{t}(\theta) A\_{t}, clip(r\_{t}(\theta), 1 - \epsilon, 1 + \epsilon) A\_{t})) \right]
\end{align}$$

### RL From Human Feedback (RLHF)

RLHF is a special case of RL algorithms where:

- the state space $\mathcal{S}$ is the set of possible user prompts $s$
- the action space $\mathcal{A}$ is the set of possible text completions $a$
- the policy $\pi\_{\theta}$ is the LLM itself
- the critic $V\_{\phi}$ is a reward model trained on human preferences

The loss function for the reward model is:

$$\begin{align}
\mathcal{L}(\theta) = -{\frac{1}{K \choose 2}} \mathbb{E} \left[ \ln(\sigma (V\_{\phi}(s,a\^{+}) - V\_{\phi}(s,a\^{-}))) \right]
\end{align}$$

Where the prompt $s$ and the preferred output $a\^{+}$ over $a\^{-}$ are sampled from $K$ labeled completions.

### Group Relative Policy Optimization (GRPO)

GRPO removes the critic model from PPO and approximates it with group averages:

$$\begin{align}
\hat{A}\_{i} = \frac{R\_{i} - \bar{R}}{\sigma\_{R}}
\end{align}$$

And adds a regularization KL divergence to the overall loss:

$$\begin{align}
L(\theta) = \mathbb{E}\_{x \sim \Pr(X)} \frace{1}{G} \sum\_{i=1}\^{G} \left[ \min(r\_{i}(\theta) \hat{A}\_{i}, clip(r\_{i}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}\_{i})) \right] - \beta \mathbb{D}\_{KL}(\pi\_{\theta} \Vert \pi\_{ref})
\end{align}$$

Where $r\_{i}(\theta)$ is the measure of policy change on the output $i$:

$$\begin{align}
r\_{i}(\theta) = {\frac{\pi\_{\theta}\left( a\_{i} \mid s \right)}{\pi\_{\theta\_{k}}\left( a\_{i} \mid s \right)}}
\end{align}$$

[image-taxonomy]: .images/algorithms.svg
