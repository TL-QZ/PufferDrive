# Soft Actor-Critic Implementation on PufferDrive
This directory is for planning only. No actual implementation code should be placed here.

Here agent workout plans for SAC implementation on PufferDrive with human author.

The spirit of this implementation is to leverage and integrate the existing Pufferlib and PufferDrive repositories to build a robust Soft Actor-Critic (SAC) training pipeline for driving.

The Pufferlib and PufferDrive are clean and well-structured repositories written in highly concise and readable manner. They were written with only necessary code, no redundant checks, conditions or other superfluous elements.

So the SAC implementation should follow the same principles, maintaining clarity, conciseness, and readability while integrating seamlessly with the existing Pufferlib and PufferDrive codebase. This means we want to reuse existing components and structures whenever possible.

The Pufferlib PPO main loop and entrance is defined in the `pufferl.py` file, where the `main` function serves as the entry point for the PPO training loop.

Similarly, the SAC implementation should have a main loop and entry point defined in a file like `pufferl_sac.py`, where the `main` function serves as the entry point for the SAC training loop. `pufferl_sac.py` does not need to reimplement usable components from `pufferl.py` file, but should instead import and utilize them as needed.

New SAC-specific functions and features may require additional cuda or c implementaitons, but these should be integrated in a way that maintains the overall clarity and readability of the codebase, closely following styles and convention of the exising code.

If encounter a case where the existing function needs modification to fit SAC usage, do not modify, implement a new function or wrapper that extends the existing functionality instead.