"""Preallocated tensor storage for discrete SAC transitions."""

import torch


class TensorReplay:
    """Store SAC transitions in fixed-size tensors.

    Fields follow the learner contract: observations, actions, rewards,
    next_observations, and discounts. ``add`` fills unused rows in order;
    ``replace`` updates existing slots without changing occupancy.
    """

    def __init__(self, capacity, observation_shape, device="cpu"):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")

        self.capacity = capacity
        self.observation_shape = tuple(observation_shape)
        self.device = torch.device(device)
        self.observations = torch.empty(
            (capacity, *self.observation_shape), dtype=torch.float32, device=self.device
        )
        self.actions = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.rewards = torch.empty(capacity, dtype=torch.float32, device=self.device)
        self.next_observations = torch.empty(
            (capacity, *self.observation_shape), dtype=torch.float32, device=self.device
        )
        self.discounts = torch.empty(capacity, dtype=torch.float32, device=self.device)
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
        return (
            self.observations[storage_indices],
            self.actions[storage_indices],
            self.rewards[storage_indices],
            self.next_observations[storage_indices],
            self.discounts[storage_indices],
        )
