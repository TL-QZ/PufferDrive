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
![alt text](runs/first_20261114/pql/raw_q_plots/critic_raw_q.png)
- Actions with lower probability generally have lower Q value estimates 
  - showing consistency in the policy's learning given the critic's estiamtes
- Top ranked actions generally have small difference in their Q value estimates. 
- In general, actions have really small differences in their Q value estimates with the same initial configuration/state.
- Larger variation in Q value estimates between different initial configurations.
  - potentially showing may be an action alone is not influencing the overall outcome of the episode given the situation the agent has already been in.
Critic Target vs Estimation Error Plot -- First result of 12 actions * 12 episodes*8 samples = 1152 points. PQL SAC 4B simulator step trained.
![alt text](runs/first_20261114/pql/target_error_plots/critic_errors_vs_target.png)
- Another interesting trend -- Under and over estimation for High vs Low Q scenarios
  - High Q scenarios (right of plot) more underestimation of lower prob. actions
    - All actions are good when the occation is easy
  - Low Q scenarios (left of plot) more over estimation of the low prob. actions
    - Need to be more careful not to choose some action given they could be really bad
3d Plot Critic Target vs Action Probability vs Estimation Error -- First result of 12 actions * 12 episodes*8 samples = 1152 points. PQL SAC 4B simulator step trained.
![alt text](runs/first_20261114/pql/target_error_plots/critic_errors_3d.png)
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
