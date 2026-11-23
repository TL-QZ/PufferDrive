# Brainstorm note for Soft Actor-Critic with Q Calibration Experiments
## How to design experiments to understand why SAC not up the standard of PPO learned policy?

### From actor gradient perspective (check doc actor_gradient_breakdown.md)
- SAC actor gradient depends on 
  - Q value estimation
  - Entropy regularization term
- SAC actor gradient of each action logit depends on
    - relative Q contribution ($q_a-\bar{q}$)
    - entropy contribution ($\alpha(\log p_a-\overline{\log p})$)
- Actor learning could be seen as fitting an optimal action distribution induced by critics estimates and alpha coefficient

- Two question
    - Is the actor correctly fitted to the critics evaluation?
    - Is the critics well learned in identifying the correct order of desirability of alternative actions?

> we start with the second question as it is more fundamental and it is less likely the actor is not well fitted (my hypothesis is actor learning is not the hardest task)

### Key objective: Is critic well learned?
Critic diagnostic is two-fold: 

(1) is the critic well learned -- predicting the correct order and estimate of Q values of each action (some action could be better supported in buffer than others); 
- Does the critic accurately estimate the relative values of all alternative first actions, followed by the current policy—not just actions commonly selected during training?

(2) is the critic estimates or the target estimate good representation of desirable behavior (better Q value corresponding to higher return or better driving quality)
- Do the critic’s estimates and training targets favor actions whose observed consequences match desirable driving behavior, as judged by reward return, safety, and goal achievement?

### Experiment: Probing the critic output and target via multi-verse rollouts
- Step 1: Selecting a set of initial configs (states) -- one frame in the middle of carla rollout (same setting as training buffer collection)
- Step 2: For each available action, choose that action and roll out the rest of episode with the current policy (multi-verse rollouts)
- Step 3: Record the critic estimates and training targets for each action, as well as the observed consequences (reward return, safety, goal achievement) of each action’s continuation


### Objective 1: Is the critic well learned?

#### Does the critic accurately estimate all alternative first actions?
- Here we assume an action has higher probability (from the current policy) represents that action is better supported in the training buffer. (not perfect assumption as this is only a snap shot of the last checkpoint)
- X-axis: probability of each action (from the current policy) -- linear & log scaled
- Y-axis: error: Esimated Q value - Target Q value (what was used for training)

Critic Esitmation Error Plot -- First result of 12 actions * 12 episodes*8 samples = 1152 points. PQL SAC 4B simulator step trained.
![alt text](image.png)
- Top ranked actions generally have really small error
- As actions become less likely, error magnifies -- but generally shown underestimation of target (negative error)
- Small number of episode shown actions have high overestimation error (positive error) -- potential concers for learning if they promote policy to choose overestimated action more often
- However error plot does not reflect the actual Q value and their relative order (compare with the action probabilities order)

Critic Raw Estimation Plot -- First result of 12 actions * 12 episodes*8 samples = 1152 points. PQL SAC 4B simulator step trained.
![alt text](runs/first_20261114/pql/plots/raw_q_plots/critic_raw_q.png)
- Actions with lower probability generally have lower Q value estimates 
  - showing consistency in the policy's learning given the critic's estiamtes
- Top ranked actions generally have small difference in their Q value estimates. 
- In general, actions have really small differences in their Q value estimates with the same initial configuration/state.
- Larger variation in Q value estimates between different initial configurations.
  - potentially showing may be an action alone is not influencing the overall outcome of the episode given the situation the agent has already been in.
Critic Target vs Estimation Error Plot -- First result of 12 actions * 12 episodes*8 samples = 1152 points. PQL SAC 4B simulator step trained.
![alt text](runs/first_20261114/pql/plots/target_error_plots/critic_errors_vs_target.png)
- Another interesting trend -- Under and over estimation for High vs Low Q scenarios
  - High Q scenarios (right of plot) more underestimation of lower prob. actions
    - All actions are good when the occation is easy
  - Low Q scenarios (left of plot) more over estimation of the low prob. actions
    - Need to be more careful not to choose some action given they could be really bad
3d Plot Critic Target vs Action Probability vs Estimation Error -- First result of 12 actions * 12 episodes*8 samples = 1152 points. PQL SAC 4B simulator step trained.
![alt text](runs/first_20261114/pql/plots/target_error_plots/critic_errors_3d.png)
- **Is this a concern?**
  - Depends on if the target Q values are accurately reflecting the desirability of the simulation outcome.
  - If low Q are actually bad driving -- overestimate is really bad and vice versa
- **Verdict**: Generally error is small except for the two type of extreme cases (high Q and low Q) -- need to check if the target Q values are accurately reflecting the desirability of the simulation outcome.

### Objective 2: Is the critic estimates or the target estimate good representation of desirable behavior?



### Future idea: Record historical policy choices during training

I want to keep this as a possible training diagnostic if the current analysis suggests it would be useful.

- For the current investigation, actor probabilities at each probing situation can show which actions the current policy is more likely to choose. I can examine whether critic errors are larger for low-probability actions.
- Current probabilities do not establish historical replay-buffer coverage: earlier policies may have chosen those actions more often.
- A future training feature could record the collection policy's action probabilities alongside the chosen action and policy version/training step. The replay already records the chosen action; the additional information would describe the policy that selected it.
- The question would be whether critic errors relate to limited exposure to particular actions in comparable situations. Global action counts alone would not establish coverage for a given situation; collection frequency and replay sampling frequency would also need to be distinguished.
- I will revisit this after the first analysis. This is a candidate diagnostic, not an implementation commitment yet.

### Pessimistic Critic Verification from VP
 
Have you compared how well the learned critic Q aligns with the actual soft return?  I briefly mentioned this last time but did not elaborate.  It would be good to do this just to understand the root causes of SAC underperformance.

This would specifically look at those situations where agents are in proximity of each other and/or near traffic lights *and* they exhibit very cautious behavior.


So you want to measure how far the critic’s predicted value $Q(s_t,a_t)$ is from the return the policy actually collects, $G_t$, as a function of distance to the nearest agent and distance to the next stop line. If $Q$ is consistently underestimating $G$ close to agents and lights, it will mean a pessimistic critic is driving the caution.

1. Freeze a snapshot. Take the current policy, both critics, and $\alpha$ from the same training checkpoint.

2. Run evaluation rollouts with that snapshot controlling all agents. Use long episodes (5000+ steps), so the discount over a horizon of $\gamma = 0.999$ doesn’t cut off returns early.

3. Log every step for every agent:

reward $r_t$, split by component (goal, collision, stop-line, comfort, etc.), computed exactly as the critic sees it (shaped, if you train with shaping);
$\log\pi(a_t\mid s_t)$;
terminal and time-out flags;
distance to the nearest agent, and the closing speed toward it;
distance to the next stop line, and that light’s state (red, yellow, green).

4. Compute the soft return backward through each episode:

$$G_t = r_t + \gamma\big(G_{t+1} - \alpha\log\pi(a_{t+1}\mid s_{t+1})\big)$$

Begin with $G_t = r_t$ at terminal step. Use only steps at least about 3000 steps before the end of the recording.

Best to also keep a log of $G$, separated into the reward and entropy components.

5. Compute the critic’s prediction with the online critics, no gradients, using the same inputs as in training:

$$\hat Q_t = Q_\phi(o^c_t, C_t)[a_t]$$

I assume you are using an ensemble of two critics. So, keep both the min and the mean of the two critics, and log their spread $|Q_1 - Q_2|$.

6. Bin and compare.

* Bin by distance to the nearest agent, for example 0–5, 5–10, 10–20, 20–40, and 40+ m. Splitting further by closing speed helps.
* Bin by distance to the stop line, separately for each light state.

In each bin, plot mean $G$, mean $\hat Q_{\min}$, mean $\hat Q_{\text{mean}}$, and the mean critic spread, with error bars.

7. Reduce noise where it matters. A single rollout gives a noisy return. For a sample of states close to agents or lights, clone the world state about 64 times. In each clone, force the same first action and roll out with a different random seed. The average return over the clones estimates $Q^\pi(s_t,a_t)$ for that exact state.

Interpretation:

a) If $\hat Q_{\min} \ll G$ near agents or lights, $\approx G$ in open road. The min over twin critics is pessimistic where the critics disagree most. Instead of min(Q_1,Q_2), use the mean of the two critics, or an ensemble estimate mean − β·std with small β.

b) If $\hat Q_{\text{mean}}$ is also $\ll G$ near agents, the bias for caution comes from the training data, not the min. It could be that incidents are oversampled or there are stale crashes in replay.
 Fix: implement prioritized replay and/or shorten buffer.

c) Gap between $\hat Q$ and $G$ grows with critic spread.
Likely that uncertainty is being converted into pessimism. 
May need to use stochasitc critic, more data in those states, or smaller \beta in the mean version of the critic, $\bar Q - \beta stdev(Q)$

d) $\hat Q \approx G$ everywhere
The critic evaluates the current SAC policy accurately. The caution comes from SAC’s entropy-augmented objective.  For $G$ broken into components, if the entropy part drops sharply near agents and lights, that drop is what drives the caution


## Third run — Does a higher critic target predict a better driving outcome?

I compared the 12 forced first actions within each starting state, with one rollout per action. For each checkpoint, the raw mean includes all 2,880 rollouts from 240 retained states pooled across bins. Pearson r and pairwise ranking accuracy are computed within each state and then averaged with equal weight across scored states. Parentheses give the number of states contributing to both scores, out of 240.

| Outcome | PQL raw mean | PQL r | PQL accuracy (n) | TES raw mean | TES r | TES accuracy (n) |
|---|---:|---:|---:|---:|---:|---:|
| Episode return | 0.3145 | 0.565 | 69.1% (240) | -1.6668 | 0.321 | 58.6% (240) |
| Sampled soft return | 0.2034 | 0.567 | 67.7% (240) | -0.2647 | 0.557 | 64.5% (240) |
| Puffer score | 0.3839 | 0.252 | 59.5% (168) | 0.0010 | 0.195 | 66.3% (4) |
| Number of goals reached | 0.8069 | 0.173 | 55.9% (57) | 0.0264 | -0.118 | 37.2% (8) |
| Total distance traveled (m) | 223.9655 | 0.196 | 56.2% (240) | 187.0909 | -0.090 | 46.7% (240) |
| Total infractions (lower is better) | 0.3205 | -0.183 | 57.8% (89) | 0.0559 | -0.335 | 71.4% (53) |
| Average distance per infraction (m) | 223.9655 | 0.196 | 56.2% (240) | 187.0909 | -0.090 | 46.7% (240) |
| Average speed (m/s) | 3.7828 | 0.179 | 56.4% (240) | 0.3188 | -0.250 | 43.2% (240) |
| Collision reward component | -0.0415 | 0.173 | 58.6% (35) | -0.0301 | 0.116 | 59.5% (20) |
| Off-road reward component | -0.1730 | 0.271 | 62.6% (75) | -0.0449 | 0.462 | 78.6% (35) |

Ranking accuracy credits fewer infractions, smaller collision and off-road penalties (signed rewards closer to zero), and higher values for the other outcomes. For speed this measures preference for faster driving. Pearson r uses the recorded metric directly, so a negative infraction correlation means higher targets accompany fewer infractions. Equal-outcome pairs are excluded from ranking and tied targets receive half credit.

Average distance per infraction equals total distance in these probes because each rollout records zero or one infraction and the denominator is bounded below by one. The TES Puffer score result draws on only four states with different outcomes across actions.

[PQL plots](runs/third_20261122/pql/plots/driving_quality_plots/README.md) · [TES plots](runs/third_20261122/tes/plots/driving_quality_plots/README.md)
