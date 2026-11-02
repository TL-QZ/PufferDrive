"""Preallocated tensor storage for discrete SAC transitions."""

import math

import torch


class TensorReplay:
    """Store SAC transitions in fixed-size tensors.

    Fields follow the learner contract: observations, actions, rewards,
    next_observations, and discounts. ``add`` fills unused rows in order;
    ``replace`` updates existing slots without changing occupancy.
    """

    def __init__(self, capacity, observation_shape, device="cpu", pin_memory=False):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")

        self.capacity = capacity
        self.observation_shape = tuple(observation_shape)
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.observations = torch.empty(
            (capacity, *self.observation_shape), dtype=torch.float32,
            device=self.device, pin_memory=self.pin_memory
        )
        self.actions = torch.empty(
            capacity, dtype=torch.int64, device=self.device, pin_memory=self.pin_memory
        )
        self.rewards = torch.empty(
            capacity, dtype=torch.float32, device=self.device, pin_memory=self.pin_memory
        )
        self.next_observations = torch.empty(
            (capacity, *self.observation_shape), dtype=torch.float32,
            device=self.device, pin_memory=self.pin_memory
        )
        self.discounts = torch.empty(
            capacity, dtype=torch.float32, device=self.device, pin_memory=self.pin_memory
        )
        self._size = 0

    def __len__(self):
        return self._size

    def add(self, observations, actions, rewards, next_observations, discounts):
        """Append a batch of copied transitions to unused rows."""
        batch_size = observations.shape[0]
        end = self._size + batch_size
        if end > self.capacity:
            raise ValueError(
                f"Replay capacity exceeded: cannot append {batch_size} rows "
                f"to {self._size} of {self.capacity} occupied slots"
            )

        destination = slice(self._size, end)
        for storage, values in (
            (self.observations, observations),
            (self.actions, actions),
            (self.rewards, rewards),
            (self.next_observations, next_observations),
            (self.discounts, discounts),
        ):
            storage[destination].copy_(values)
        self._size = end

    def replace(self, indices, observations, actions, rewards, next_observations, discounts):
        """Copy transitions into existing slots without changing occupancy."""
        storage_indices = indices.to(device=self.device)
        for storage, values in (
            (self.observations, observations),
            (self.actions, actions),
            (self.rewards, rewards),
            (self.next_observations, next_observations),
            (self.discounts, discounts),
        ):
            storage[storage_indices] = values.to(device=self.device)

    def get_batch(self, indices):
        """Return independent indexed copies in transition-field order."""
        storage_indices = indices.to(device=self.device)
        if self.pin_memory:
            batches = []
            for storage in (
                self.observations,
                self.actions,
                self.rewards,
                self.next_observations,
                self.discounts,
            ):
                batch = torch.empty(
                    (storage_indices.numel(), *storage.shape[1:]),
                    dtype=storage.dtype,
                    device=self.device,
                    pin_memory=True,
                )
                torch.index_select(storage, 0, storage_indices, out=batch)
                batches.append(batch)
            return tuple(batches)

        return (
            self.observations[storage_indices],
            self.actions[storage_indices],
            self.rewards[storage_indices],
            self.next_observations[storage_indices],
            self.discounts[storage_indices],
        )


class NStepAccumulator:
    """Accumulate per-agent transitions into n-step replay rows.

    The accumulator keeps at most ``n_steps`` transitions per stable agent.
    A row is emitted once its full horizon is available, or all remaining
    shorter rows are emitted when an episode boundary or an explicit flush
    occurs. Returned rows are independent copies in TensorReplay field order.
    """

    def __init__(
        self,
        num_agents,
        observation_shape,
        n_steps,
        gamma,
        device="cpu",
        pin_memory=False,
    ):
        for name, value in (("num_agents", num_agents), ("n_steps", n_steps)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool):
            raise ValueError("gamma must be a finite number between zero and one")
        if not math.isfinite(gamma) or gamma < 0 or gamma > 1:
            raise ValueError("gamma must be a finite number between zero and one")

        self.num_agents = num_agents
        self.observation_shape = tuple(observation_shape)
        self.n_steps = n_steps
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory and self.device.type == "cpu")

        ring_shape = (num_agents, n_steps)
        self._observations = self._empty((*ring_shape, *self.observation_shape), torch.float32)
        self._actions = self._empty(ring_shape, torch.int64)
        self._rewards = self._empty(ring_shape, torch.float32)
        self._last_next_observations = self._empty(
            (num_agents, *self.observation_shape), torch.float32
        )
        self._counts = self._empty((num_agents,), torch.int64)
        self._write_positions = self._empty((num_agents,), torch.int64)
        self._counts.zero_()
        self._write_positions.zero_()

    def _empty(self, shape, dtype):
        return torch.empty(
            shape,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    @property
    def pending_count(self):
        """Number of transitions currently waiting for a horizon or flush."""
        return int(self._counts.sum().item())

    def clear(self):
        """Discard every pending transition and reset all ring positions."""
        self._counts.zero_()
        self._write_positions.zero_()

    def append(
        self,
        agent_ids,
        observations,
        actions,
        rewards,
        next_observations,
        terminals,
        truncations,
    ):
        """Append one valid transition per agent and emit ready replay rows.

        ``agent_ids`` must be unique within the batch. ``terminals`` suppress
        bootstrap discounts, while truncations retain their n-step discount.
        """
        agent_ids = torch.as_tensor(agent_ids, device=self.device)
        if agent_ids.ndim != 1 or agent_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("agent_ids must be a one-dimensional integer tensor")
        agent_ids = agent_ids.to(dtype=torch.int64)
        batch_size = agent_ids.numel()

        observations = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        next_observations = torch.as_tensor(
            next_observations, device=self.device, dtype=torch.float32
        )
        terminals = torch.as_tensor(terminals, device=self.device, dtype=torch.bool)
        truncations = torch.as_tensor(truncations, device=self.device, dtype=torch.bool)

        expected_observation_shape = (batch_size, *self.observation_shape)
        if tuple(observations.shape) != expected_observation_shape:
            raise ValueError(
                f"observations must have shape {expected_observation_shape}, "
                f"got {tuple(observations.shape)}"
            )
        if tuple(next_observations.shape) != expected_observation_shape:
            raise ValueError(
                f"next_observations must have shape {expected_observation_shape}, "
                f"got {tuple(next_observations.shape)}"
            )
        for name, values in (
            ("actions", actions),
            ("rewards", rewards),
            ("terminals", terminals),
            ("truncations", truncations),
        ):
            if tuple(values.shape) != (batch_size,):
                raise ValueError(f"{name} must have shape ({batch_size},)")

        if batch_size == 0:
            return self._empty_result()
        if bool(torch.any(agent_ids < 0)) or bool(torch.any(agent_ids >= self.num_agents)):
            raise ValueError("agent_ids must be within the configured agent range")
        if torch.unique(agent_ids).numel() != batch_size:
            raise ValueError("agent_ids must be unique within an append batch")

        write_positions = self._write_positions[agent_ids]
        self._observations[agent_ids, write_positions] = observations
        self._actions[agent_ids, write_positions] = actions
        self._rewards[agent_ids, write_positions] = rewards
        self._last_next_observations[agent_ids] = next_observations

        new_counts = self._counts[agent_ids] + 1
        self._write_positions[agent_ids] = (write_positions + 1) % self.n_steps
        self._counts[agent_ids] = new_counts

        boundaries = terminals | truncations
        emission_counts = torch.where(
            boundaries,
            new_counts,
            torch.where(
                new_counts == self.n_steps,
                new_counts,
                torch.zeros_like(new_counts),
            ),
        )
        emitting = emission_counts > 0
        emitted = self._emit_prefixes(
            agent_ids[emitting],
            emission_counts[emitting],
            self._write_positions[agent_ids[emitting]],
            next_observations[emitting],
            terminals[emitting],
            torch.where(
                boundaries[emitting],
                torch.ones_like(emission_counts[emitting]),
                emission_counts[emitting],
            ),
        )

        self._counts[agent_ids] = torch.where(
            boundaries,
            torch.zeros_like(new_counts),
            torch.clamp(new_counts, max=self.n_steps - 1),
        )

        return emitted

    def flush(self, agent_ids=None):
        """Emit all pending rows using each agent's last valid successor."""
        if agent_ids is None:
            selected_agent_ids = torch.arange(self.num_agents, device=self.device)
        else:
            selected_agent_ids = torch.as_tensor(agent_ids, device=self.device)
            if selected_agent_ids.ndim != 1 or selected_agent_ids.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise ValueError("agent_ids must be a one-dimensional integer tensor")
            selected_agent_ids = selected_agent_ids.to(dtype=torch.int64)
            if bool(torch.any(selected_agent_ids < 0)) or bool(
                torch.any(selected_agent_ids >= self.num_agents)
            ):
                raise ValueError("agent_ids must be within the configured agent range")
            if torch.unique(selected_agent_ids).numel() != selected_agent_ids.numel():
                raise ValueError("agent_ids must be unique within a flush request")

        pending = self._counts[selected_agent_ids] > 0
        selected_agent_ids = selected_agent_ids[pending]
        if selected_agent_ids.numel() == 0:
            return self._empty_result()

        counts = self._counts[selected_agent_ids]
        write_positions = self._write_positions[selected_agent_ids]
        emitted = self._emit_prefixes(
            selected_agent_ids,
            counts,
            write_positions,
            self._last_next_observations[selected_agent_ids],
            minimum_horizons=torch.ones_like(counts),
        )
        self._counts[selected_agent_ids] = 0
        return emitted

    def _emit_prefixes(
        self,
        agent_ids,
        counts,
        write_positions,
        successor_observations,
        terminals=None,
        minimum_horizons=None,
    ):
        """Emit each available suffix with discounted rewards in O(n_steps)."""
        if agent_ids.numel() == 0:
            return self._empty_result()

        if terminals is None:
            terminals = torch.zeros_like(counts, dtype=torch.bool)
        if minimum_horizons is None:
            minimum_horizons = torch.ones_like(counts)
        observation_rows = []
        action_rows = []
        reward_rows = []
        next_observation_rows = []
        discount_rows = []
        horizon_rows = []
        accumulated_rewards = torch.zeros(
            agent_ids.numel(), dtype=torch.float32, device=self.device
        )

        for horizon in range(1, self.n_steps + 1):
            available = counts >= horizon
            if not bool(torch.any(available)):
                continue

            available_agent_ids = agent_ids[available]
            available_start_positions = (
                write_positions[available] - horizon
            ) % self.n_steps
            accumulated_rewards[available] = (
                self._rewards[available_agent_ids, available_start_positions]
                + self.gamma * accumulated_rewards[available]
            )

            ready = available & (minimum_horizons <= horizon)
            if not bool(torch.any(ready)):
                continue

            ready_agent_ids = agent_ids[ready]
            start_positions = (write_positions[ready] - horizon) % self.n_steps
            observation_rows.append(
                self._observations[ready_agent_ids, start_positions]
            )
            action_rows.append(self._actions[ready_agent_ids, start_positions])
            reward_rows.append(accumulated_rewards[ready])
            next_observation_rows.append(successor_observations[ready])
            discounts = torch.full(
                (ready_agent_ids.numel(),),
                self.gamma**horizon,
                dtype=torch.float32,
                device=self.device,
            )
            discounts[terminals[ready]] = 0
            discount_rows.append(discounts)
            horizon_rows.append(
                torch.full(
                    (ready_agent_ids.numel(),),
                    horizon,
                    dtype=torch.int64,
                    device=self.device,
                )
            )

        return self._pack_result(
            observation_rows,
            action_rows,
            reward_rows,
            next_observation_rows,
            discount_rows,
            horizon_rows,
        )

    def _empty_result(self):
        return (
            (
                self._empty((0, *self.observation_shape), torch.float32),
                self._empty((0,), torch.int64),
                self._empty((0,), torch.float32),
                self._empty((0, *self.observation_shape), torch.float32),
                self._empty((0,), torch.float32),
            ),
            self._empty((0,), torch.int64),
        )

    def _pack_result(
        self,
        observation_rows,
        action_rows,
        reward_rows,
        next_observation_rows,
        discount_rows,
        horizon_rows,
    ):
        if not observation_rows:
            return self._empty_result()

        batches = (
            torch.cat(observation_rows),
            torch.cat(action_rows),
            torch.cat(reward_rows),
            torch.cat(next_observation_rows),
            torch.cat(discount_rows),
            torch.cat(horizon_rows),
        )
        if not self.pin_memory:
            return batches[:5], batches[5]

        pinned_batches = []
        for batch in batches:
            pinned = self._empty(tuple(batch.shape), batch.dtype)
            pinned.copy_(batch)
            pinned_batches.append(pinned)
        return tuple(pinned_batches[:5]), pinned_batches[5]
