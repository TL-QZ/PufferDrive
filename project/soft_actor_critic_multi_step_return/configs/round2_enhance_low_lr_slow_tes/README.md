# round2_enhance_low_lr_slow_tes

Derived from `configs/round2_enhance/` and the saved trial configuration
`runs/round2_enhance/sac1_grid_20261103T200522Z728831771/configs/trial_01.yaml`. This isolated variant keeps the source recipe and changes the
initial actor and critic learning rates, alpha learning rate, and two adaptive
Target Entropy Scheduling (TES) thresholds.

The inherited training recipe remains at 3,200 agents per environment, 20
vector environments, `n_steps: 4`, 25% replay replacement, learned alpha,
learning-rate annealing, a 4B global-row budget, and evaluation every two
collection cycles. The source leaves `train.critic_entropy_in_target` unspecified, preserving the learner default. The copied evaluation catalog is
byte-identical to the source catalog; the config points to its isolated copy.

Changed settings:

- `train.learning_rate: 0.0001`
- `train.critic_learning_rate: 0.0001`
- `train.alpha_learning_rate: 0.00007`
- `train.target_entropy_std_threshold: 0.025`
- `train.target_entropy_required_count: 2000`

The remaining adaptive TES settings are inherited unchanged from the source.
The W&B group, fallback learner directory, benchmark path, and launcher output
directory use this variant's name to keep artifacts separate.

Preview trial 01:

```bash
source .venv/bin/activate
bash project/soft_actor_critic_multi_step_return/scripts/launch_round2_enhance_low_lr_slow_tes.sh --trial 01 --dry-run --no-wandb
```

Launch trial 01 on two selected GPUs by removing `--dry-run --no-wandb` and
adding `--gpus ID,ID` if the default `0,1` pair is not appropriate. Training
has not been launched.

Validation: launcher syntax and dry runs for selectors `01` and `1` passed.
The generated trial config matches the saved source exactly except for the
five listed parameters and isolated run/output/logging/catalog identities.
Benchmark bytes and launcher run controls are unchanged. The paired variants
also match after excluding critic-target entropy and their identity paths.
