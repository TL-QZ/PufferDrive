# Discrete SAC actor: gradients, optimal policy, and the role of Q

The actor learns action probabilities from the critic's **relative action values** and the entropy coefficient **alpha**. For a fixed critic and positive alpha, the ideal action distribution at an observation is `softmax(Q / alpha)`. Alpha changes this optimum, not merely the time needed to reach it.

This note records my questions and initial ideas alongside the explanations developed in conversation. The first-person prompts below paraphrase what I asked or proposed; they are not verbatim quotations. Initial ideas are kept distinct from the conclusions reached afterward. This is a conceptual note, not a new experimental result or an established explanation for cautious driving.

## 1. Actor loss and notation

**My starting question:** I want to step back from the calibration experiment and understand SAC training from its losses. The actor loss depends on Q and entropy—what is its gradient formulation?

For a fixed observation $o$, define

$$
p_a=\pi_\theta(a\mid o),\qquad
q_a=\operatorname{stopgrad}\!\left[\min\left(Q_{\phi_1}(o,a),Q_{\phi_2}(o,a)\right)\right].
$$

Here $\theta$ denotes actor parameters. The critics and $\alpha$ are held fixed during the actor update. The per-observation loss is

$$
\boxed{
\ell_\pi(o)=\sum_a p_a\left[\alpha\log p_a-q_a\right]
=-\mathbb E_{a\sim\pi_\theta}[q_a]-\alpha H(\pi_\theta(\cdot\mid o)).
}
$$

The batch loss averages this expression over replay observations. In the discrete SAC formulation discussed here, the action expectation is computed by summing over all action classes. It is not estimated using only the action sampled for environment interaction.

The two influences are maximizing expected Q and maximizing entropy, with alpha multiplying the entropy term. This is **alpha times entropy**, not alpha plus entropy.

## 2. Gradient with respect to actor parameters

**My question about the derivation:** Why can we write $\nabla_\theta p_a=p_a\nabla_\theta\log p_a$? I want to understand this step before using it in the actor gradient.

The identity used in the derivation follows directly from the chain rule:

$$
\nabla_\theta\log p_a
=\frac{1}{p_a}\nabla_\theta p_a
\quad\Longrightarrow\quad
\boxed{\nabla_\theta p_a=p_a\nabla_\theta\log p_a.}
$$

It requires $p_a>0$, which softmax guarantees mathematically for finite logits. It rewrites a derivative of a probability as a probability-weighted derivative of its logarithm.

Differentiating the loss gives

$$
\nabla_\theta\ell_\pi
=\sum_a p_a\left[\alpha(\log p_a+1)-q_a\right]
\nabla_\theta\log p_a.
$$

Because probabilities sum to one,

$$
\sum_a p_a\nabla_\theta\log p_a
=\nabla_\theta\sum_a p_a=0.
$$

The extra $+1$ term therefore cancels:

$$
\boxed{
\nabla_\theta\ell_\pi
=\sum_a p_a(\alpha\log p_a-q_a)\nabla_\theta\log p_a
=\mathbb E_{a\sim\pi_\theta}
\left[(\alpha\log\pi_\theta(a\mid o)-q_a)
\nabla_\theta\log\pi_\theta(a\mid o)\right].
}
$$

There is no gradient through Q in this actor update. The Q values weight the changes in action probabilities. The sampled replay observations are also treated as fixed inputs during this update.

## 3. Gradient for each action logit

**My understanding to check:** Is the actor update largely determined by differences between the Q values of the action choices, together with alpha and entropy?

**Clarification:** Yes, with the current probabilities also entering the gradient. Alpha multiplies the entropy contribution; it is not an additive “alpha plus entropy” term. The expression below makes each action's contribution explicit.

Let $z_a$ denote the actor logits, with $p=\operatorname{softmax}(z)$. Define the policy-weighted averages

$$
\bar q=\sum_b p_bq_b,\qquad
\overline{\log p}=\sum_b p_b\log p_b.
$$

Using $\partial p_b/\partial z_a=p_b(\mathbf 1_{a=b}-p_a)$,

$$
\boxed{
\frac{\partial\ell_\pi}{\partial z_a}
=p_a\left[
\alpha(\log p_a-\overline{\log p})-(q_a-\bar q)
\right].
}
$$

If the logits themselves were independently optimized by gradient descent, their updates would be

$$
\Delta z_a=\eta p_a\left[
\underbrace{q_a-\bar q}_{\text{relative Q contribution}}
-\underbrace{\alpha(\log p_a-\overline{\log p})}_{\text{entropy contribution}}
\right].
$$

| Contribution | Direction of its direct logit update |
|---|---|
| Q above the policy-weighted average | Raises the action's logit |
| Q below the policy-weighted average | Lowers the action's logit |
| Log probability above its policy-weighted average | Entropy lowers the action's logit |
| Log probability below its policy-weighted average | Entropy raises the action's logit |

The total direction depends on both contributions. Both are multiplied by $p_a$, so very unlikely actions can receive small logit gradients, even though all actions appear in the loss.

The actual actor is a neural network, not a table of independent logits:

$$
\nabla_\theta L_\pi
=\frac{1}{B}\sum_{i=1}^{B}\sum_a
\frac{\partial\ell_\pi(o_i)}{\partial z_a(o_i)}
\nabla_\theta z_a(o_i).
$$

Shared parameters, gradients from other observations, and the optimizer affect the resulting changes. The direct-logit expression explains the local training signal; it is not a guarantee about each logit's change after a network update.

## 4. Why relative Q values matter

**Where I got stuck:** I did not initially understand the explanation about what the gradient means for driving behavior. I also asked why subtracting a constant from Q had entered the discussion—I had not proposed that idea.

**Why this example is here:** The agent introduced a common Q shift to illustrate why relative action values matter. It was an explanatory example, not my hypothesis about the experiment.

Adding the same observation-dependent constant $c(o)$ to every action value changes the loss by a constant:

$$
q'_a=q_a+c(o)
\quad\Longrightarrow\quad
\ell'_\pi(o)=\ell_\pi(o)-c(o).
$$

With the critic held fixed, the actor gradient is unchanged. Equivalently,

$$
q'_a-\bar q'=q_a-\bar q.
$$

For example, Q values $(10,9)$ and $(-90,-91)$ produce the same actor gradient for the same policy and alpha. Both give the first action an advantage of one over the second.

The constant shift is not an observed error pattern or an assumption about our experiment.

The practical implication is that a common absolute Q bias does not directly explain a wrong action preference. **Errors in action rankings and action-value gaps** are more directly relevant to the actor. This invariance describes a fixed actor update; it does not imply absolute calibration is irrelevant to critic training and bootstrapping.

## 5. Optimal policy under fixed Q and alpha

**My follow-up question:** The connection between Q differences and action preferences is useful. Can we elaborate on the “optimal policy” under the current Q estimates?

**Meaning of optimal here:** I am considering the probabilities that minimize the actor loss for fixed Q and alpha at one observation. This is different from claiming that the resulting driving policy is globally optimal.

For one observation, optimize the action probabilities directly:

$$
\min_{\{p_a\}}\sum_a p_a(\alpha\log p_a-q_a),
\qquad \sum_a p_a=1.
$$

For $\alpha>0$, introduce a multiplier $\lambda$:

$$
\mathcal L=\sum_a p_a(\alpha\log p_a-q_a)
+\lambda\left(\sum_a p_a-1\right).
$$

At the optimum,

$$
\frac{\partial\mathcal L}{\partial p_a}
=\alpha(\log p_a+1)-q_a+\lambda=0.
$$

Thus $p_a=\exp(q_a/\alpha)\exp(-1-\lambda/\alpha)$. Normalizing gives

$$
\boxed{
p_a^*=\frac{\exp(q_a/\alpha)}{\sum_b\exp(q_b/\alpha)}
=\operatorname{softmax}(q/\alpha)_a.
}
$$

For finite Q values and positive alpha, this is the unique minimum over action probabilities. It is the optimum for the **current critic**, not a claim of globally optimal driving.

For any two actions,

$$
\boxed{\frac{p_a^*}{p_b^*}=\exp\left(\frac{q_a-q_b}{\alpha}\right).}
$$

| Q gap $q_a-q_b$ | Optimal probability ratio $p_a^*/p_b^*$ |
|---|---:|
| $0$ | $1$ |
| $\alpha$ | $e\approx2.72$ |
| $2\alpha$ | $e^2\approx7.39$ |
| $5\alpha$ | $e^5\approx148$ |

These ratios hold with any number of actions. With exactly two actions,

$$
p_a^*=\frac{1}{1+\exp(-(q_a-q_b)/\alpha)}.
$$

For a Q gap of 0.01, alpha values 0.1, 0.01, and 0.001 produce ideal probabilities of approximately 52.5%, 73.1%, and 99.995%, respectively.

As alpha approaches zero from above, probability concentrates on the maximum-Q actions. As alpha grows relative to finite Q gaps, the policy approaches uniform. Alpha changes the desired distribution even after convergence; it is not merely a learning-rate parameter.

## 6. Actor learning as fitting the critic-induced distribution

**Connection to my question:** I want to connect the gradient of each action to the optimal policy and understand how that optimum relates to the actor loss. The KL reformulation below was introduced in the explanation to make that connection explicit.

Let $Z(o)=\sum_a\exp(q_a/\alpha)$ and $p^*=\operatorname{softmax}(q/\alpha)$. The loss is exactly

$$
\boxed{
\ell_\pi(o)
=\alpha D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid o)\,\|\,p^*(\cdot\mid o)\right)
-\alpha\log Z(o).
}
$$

The last term is constant with respect to actor parameters during the update. The actor therefore fits its probabilities toward the distribution induced by the current Q values, using the KL direction shown above.

The trained network need not match this distribution exactly: its parameters are shared across observations, optimization is incomplete, and Q and alpha can change during training.

This suggests separating two questions: does the critic-induced distribution favor a reasonable action, and does the actor match that distribution? If the actor and critic-induced distribution both favor braking, the actor is following the critic. If they disagree, actor fitting or lag is another possibility. Neither observation establishes the true return of an action.

## 7. Alpha affects both actor preferences and soft Q targets

**My initial interpretation:** Combining the actor gradient with the optimal-policy result, it seems that the learned driving policy is mostly determined by Q once we have a way to choose alpha. Alpha controls how strongly a Q gap is expressed, but accurate Q estimation seems central. I wondered whether alpha's remaining role was mainly in how long it takes to reach the optimum while Q is being learned.

**What needed refining:** Q is the actor's main action-value signal, but alpha changes the destination as well as potentially affecting optimization dynamics. Even with fixed Q, different alpha values give different optimal probabilities. In soft SAC, alpha also changes the return that Q is trained to estimate.

There are two distinct entropy contributions:

| Location | Meaning |
|---|---|
| Explicit actor entropy term | Encourages diversity of actions at the current observation |
| Future entropy included in soft Q | Values the entropy encountered after the initial action |

The current-state entropy term does not inherently prefer braking. Any driving-specific preference must be understood through action values and the current distribution.

For a fixed policy, the soft return can be written, using full state $s$ for the theoretical definition, as

$$
Q^{\pi,\alpha}(s,a)
=\mathbb E\left[
r_t+\sum_{k=1}^{\infty}\gamma^k
\left(r_{t+k}-\alpha\log\pi(a_{t+k}\mid o_{t+k})\right)
\mid s_t=s,a_t=a
\right],
$$

with continuation ending at true terminals. The initial action's entropy is excluded. Our critic receives observations, so limited information in those observations also matters to what it can predict.

For a fixed policy and environment,

$$
Q^{\pi,\alpha}=Q^\pi_{\mathrm{reward}}
+\alpha Q^\pi_{\mathrm{future\ entropy}}.
$$

Consequently,

$$
\Delta Q^{\pi,\alpha}
=\Delta Q^\pi_{\mathrm{reward}}
+\alpha\Delta Q^\pi_{\mathrm{future\ entropy}}.
$$

Alpha both changes the actor's response to a given Q gap and changes the soft value being learned. The decomposition is linear in alpha only while the policy is held fixed; retraining can change the policy and both components.

An accurate critic also evaluates continuation under the **current policy**, not the best possible future driving. A promising first action can have low value if the current policy handles the subsequent situation poorly.

## 8. Implications for our critic-first investigation

**My proposed analysis order:** I want to investigate Q learning first, because alpha is easier to tune through a hyperparameter sweep. Once I know whether Q is well learned, I can investigate alpha. I recognize that the current policy also collects experience, but I do not want that coupling to prevent us from diagnosing a biased or poorly learned critic.

**Working conclusion:** Diagnose the learned critic under the current policy and alpha before using alpha sweeps to investigate behavior. We do not need to resolve every interaction in RL before checking for critic errors. Data coverage can still be a cause of critic error, and changing alpha during retraining calls for renewed critic checks.

The main distinctions are:

1. **Wrong action values:** the critic misestimates relative soft returns and teaches the actor a wrong preference.
2. **Accurate values for an undesirable continuation:** the critic correctly evaluates a current policy or reward/entropy objective that favors cautious behavior.
3. **Actor mismatch:** the actor has not fitted the distribution implied by the current Q and alpha.

For fixed Q, changing positive alpha changes the strength of a preference but cannot reverse the Q ranking. An alpha sweep is therefore not a reliable substitute for correcting wrong relative values. Retraining at another alpha can change Q itself and should include renewed critic checks.

Critic evaluation should compare predicted and empirical action-value differences, in addition to absolute calibration. Replay coverage remains a possible cause of critic error, and low critic training loss alone does not establish accuracy because its targets also contain learned values.

**My next question about interpreting actions:** I want to inspect the critic's judgment of each action under different observations. But the 12 actions are longitudinal/lateral jerks that must be integrated into motion, rather than direct coordinate commands. How should I interpret the actor's chosen action and evaluate its consequences?

**Clarification for this analysis:** Labels such as “progress” and “brake” are shorthand, not fixed meanings of action classes. A positive longitudinal jerk can release braking while the vehicle continues slowing down. Q evaluates one selected jerk command followed by the current policy, not sustained application of that command. Action comparisons must therefore be grounded in the current motion state and resulting continuation.

The question guiding this stage is:

> At the current alpha and frozen policy, do the critics correctly estimate the relative soft returns of alternative first-step driving actions?
