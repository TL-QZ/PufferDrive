"""FIFO replay storage and epoch planning for contiguous SAC sequences."""

import math

import torch

from pufferlib.sac_replay import TensorReplay


_INTEGER_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


class SequenceReplay:
    """Store raw SAC transitions and plan disjoint contiguous replay segments.

    Transition values use the same five-field order as :class:`TensorReplay`.
    Temporal links and scalar boundary metadata stay on CPU so replay planning
    does not require moving the observation store off its configured device.
    """

    def __init__(
        self,
        capacity,
        observation_shape,
        num_agents,
        device="cpu",
        pin_memory=False,
    ):
        for name, value in (("capacity", capacity), ("num_agents", num_agents)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        requested_device = torch.device(device)
        effective_pin_memory = bool(pin_memory and requested_device.type == "cpu")
        self._replay = TensorReplay(
            capacity,
            observation_shape,
            device=requested_device,
            pin_memory=effective_pin_memory,
        )
        self.capacity = self._replay.capacity
        self.device = self._replay.device
        self.pin_memory = self._replay.pin_memory
        self.observation_shape = self._replay.observation_shape
        self.num_agents = num_agents

        self._valid = torch.zeros(capacity, dtype=torch.bool)
        self._generation = torch.zeros(capacity, dtype=torch.int64)
        self._agent_ids = torch.full((capacity,), -1, dtype=torch.int64)
        self._terminals = torch.zeros(capacity, dtype=torch.bool)
        self._truncations = torch.zeros(capacity, dtype=torch.bool)
        self._previous_indices = torch.full((capacity,), -1, dtype=torch.int64)
        self._previous_generations = torch.full((capacity,), -1, dtype=torch.int64)
        self._next_indices = torch.full((capacity,), -1, dtype=torch.int64)
        self._next_generations = torch.full((capacity,), -1, dtype=torch.int64)
        self._insertion_steps = torch.zeros(capacity, dtype=torch.int64)
        self._tail_indices = torch.full((num_agents,), -1, dtype=torch.int64)
        self._tail_generations = torch.full((num_agents,), -1, dtype=torch.int64)

        self._next_slot = 0
        self._insertion_clock = 0
        self._overwrite_cuts_since_plan = 0
        self._gap_cuts_since_plan = 0
        self._stale_link_cuts_since_plan = 0
        self.last_plan_metrics = {}
        self.last_suffix_length_counts = torch.zeros(1, dtype=torch.int64)

    def __len__(self):
        return len(self._replay)

    def get_batch(self, indices):
        """Return copied transition tensors in TensorReplay field order."""
        return self._replay.get_batch(indices)

    def get_sequence_metadata(self, indices):
        """Return CPU metadata copies for selected replay rows."""
        indices = self._cpu_indices(indices, "indices")
        if indices.numel() and bool(torch.any((indices < 0) | (indices >= self.capacity))):
            raise ValueError("indices must be within replay capacity")
        return {
            "agent_ids": self._agent_ids[indices].clone(),
            "terminals": self._terminals[indices].clone(),
            "truncations": self._truncations[indices].clone(),
            "generations": self._generation[indices].clone(),
        }

    def clear_links(self):
        """Clear pending per-agent tails when collection resumes from a fresh replay."""
        self._tail_indices.fill_(-1)
        self._tail_generations.fill_(-1)

    def reset_collection_metrics(self):
        """Reset event counters after the caller has logged collection metrics."""
        self._overwrite_cuts_since_plan = 0
        self._gap_cuts_since_plan = 0
        self._stale_link_cuts_since_plan = 0
        for metric_name in (
            "cut_overwrite_count",
            "cut_gap_count",
            "cut_stale_generation_count",
        ):
            if metric_name in self.last_plan_metrics:
                self.last_plan_metrics[metric_name] = 0

    def break_chains(self, agent_ids):
        """Prevent later rows for these agents from joining their previous row."""
        selected_ids = self._agent_id_tensor(agent_ids)
        if selected_ids.numel() == 0:
            return
        selected_ids = torch.unique(selected_ids)
        tail_indices = self._tail_indices[selected_ids]
        tail_generations = self._tail_generations[selected_ids]
        has_tail = tail_indices >= 0
        safe_indices = tail_indices.clamp_min(0)
        has_tail &= self._valid[safe_indices]
        has_tail &= self._generation[safe_indices] == tail_generations
        has_tail &= ~self._terminals[safe_indices]
        has_tail &= ~self._truncations[safe_indices]
        self._gap_cuts_since_plan += int(has_tail.sum())
        self._tail_indices[selected_ids] = -1
        self._tail_generations[selected_ids] = -1

    def add_sequence(
        self,
        agent_ids,
        observations,
        actions,
        rewards,
        next_observations,
        discounts,
        terminals,
        truncations,
    ):
        """Insert one raw outcome per stable agent ID, linking valid neighbors.

        Terminal rows always store a zero bootstrap discount and take precedence
        over truncation. Both terminal and truncation rows end their chain.
        """
        agent_ids = self._agent_id_tensor(agent_ids)
        batch_size = agent_ids.numel()
        observation_shape = (batch_size, *self.observation_shape)
        observations = torch.as_tensor(
            observations, device=self.device, dtype=torch.float32
        )
        next_observations = torch.as_tensor(
            next_observations, device=self.device, dtype=torch.float32
        )
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        discounts = torch.as_tensor(discounts, device=self.device, dtype=torch.float32)
        terminals = torch.as_tensor(terminals, dtype=torch.bool, device="cpu")
        truncations = torch.as_tensor(truncations, dtype=torch.bool, device="cpu")

        if tuple(observations.shape) != observation_shape:
            raise ValueError(f"observations must have shape {observation_shape}")
        if tuple(next_observations.shape) != observation_shape:
            raise ValueError(f"next_observations must have shape {observation_shape}")
        for name, values in (
            ("actions", actions),
            ("rewards", rewards),
            ("discounts", discounts),
            ("terminals", terminals),
            ("truncations", truncations),
        ):
            if tuple(values.shape) != (batch_size,):
                raise ValueError(f"{name} must have shape ({batch_size},)")
        if batch_size == 0:
            return
        if torch.unique(agent_ids).numel() != batch_size:
            raise ValueError("agent_ids must be unique within an insertion batch")
        if not bool(torch.isfinite(discounts).all()) or bool(
            torch.any((discounts < 0) | (discounts > 1))
        ):
            raise ValueError("discounts must be finite values between zero and one")

        terminal_values = terminals
        truncation_values = truncations & ~terminal_values
        discount_values = discounts.clone()
        discount_values[terminal_values.to(device=self.device)] = 0

        incoming_indices = torch.arange(batch_size, dtype=torch.int64)
        retained_start = max(0, batch_size - self.capacity)
        retained_rows = incoming_indices[retained_start:]
        retained_count = retained_rows.numel()
        start_slot = self._next_slot
        retained_slots = (start_slot + incoming_indices[retained_start:]) % self.capacity

        if batch_size >= self.capacity and len(self) > 0:
            self._evict_slots(torch.nonzero(self._valid, as_tuple=False).flatten())
        else:
            self._evict_slots(retained_slots[self._valid[retained_slots]])

        retained_agent_ids = agent_ids[retained_rows]
        previous_indices = self._tail_indices[retained_agent_ids].clone()
        previous_generations = self._tail_generations[retained_agent_ids].clone()
        has_previous = previous_indices >= 0
        safe_previous = previous_indices.clamp_min(0)
        has_previous &= self._valid[safe_previous]
        has_previous &= self._generation[safe_previous] == previous_generations
        has_previous &= self._agent_ids[safe_previous] == retained_agent_ids
        has_previous &= ~self._terminals[safe_previous]
        has_previous &= ~self._truncations[safe_previous]
        if bool(has_previous.any()):
            previous_next_indices = self._next_indices[safe_previous[has_previous]]
            if bool(torch.any(previous_next_indices >= 0)):
                raise RuntimeError("sequence replay tail already has a successor")

        generations = self._generation[retained_slots] + 1
        self._write_storage(
            retained_slots,
            retained_rows,
            observations,
            actions,
            rewards,
            next_observations,
            discount_values,
        )
        self._generation[retained_slots] = generations
        self._valid[retained_slots] = True
        self._agent_ids[retained_slots] = retained_agent_ids
        self._terminals[retained_slots] = terminal_values[retained_rows]
        self._truncations[retained_slots] = truncation_values[retained_rows]
        self._previous_indices[retained_slots] = torch.where(
            has_previous, previous_indices, torch.full_like(previous_indices, -1)
        )
        self._previous_generations[retained_slots] = torch.where(
            has_previous, previous_generations, torch.full_like(previous_generations, -1)
        )
        self._next_indices[retained_slots] = -1
        self._next_generations[retained_slots] = -1
        self._insertion_steps[retained_slots] = (
            self._insertion_clock + retained_rows + 1
        )

        if bool(has_previous.any()):
            predecessor_indices = previous_indices[has_previous]
            self._next_indices[predecessor_indices] = retained_slots[has_previous]
            self._next_generations[predecessor_indices] = generations[has_previous]

        retained_terminals = terminal_values[retained_rows]
        retained_truncations = truncation_values[retained_rows]
        ending_rows = retained_terminals | retained_truncations
        continuing_ids = retained_agent_ids[~ending_rows]
        self._tail_indices[continuing_ids] = retained_slots[~ending_rows]
        self._tail_generations[continuing_ids] = generations[~ending_rows]
        self._tail_indices[retained_agent_ids[ending_rows]] = -1
        self._tail_generations[retained_agent_ids[ending_rows]] = -1

        self._insertion_clock += batch_size
        self._next_slot = (start_slot + batch_size) % self.capacity
        self._replay._size = min(self.capacity, len(self) + retained_count)

    def build_epoch_batches(self, horizon, logical_batch_size):
        """Return shuffled, disjoint CPU index/offset plans for one replay epoch."""
        for name, value in (("horizon", horizon), ("logical_batch_size", logical_batch_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        resident_indices = torch.nonzero(self._valid, as_tuple=False).flatten()
        row_count = resident_indices.numel()
        self.last_suffix_length_counts = torch.zeros(1, dtype=torch.int64)
        if row_count == 0:
            self.last_plan_metrics = self._empty_metrics()
            return []

        successors, predecessors, stale_link_count = self._validated_link_graph()
        self._stale_link_cuts_since_plan += stale_link_count
        ordered_indices, roots, depths = self._ordered_chain_rows(
            resident_indices, successors, predecessors
        )

        chain_start_mask = torch.ones(row_count, dtype=torch.bool)
        chain_start_mask[1:] = roots[1:] != roots[:-1]
        chain_ids = torch.cumsum(chain_start_mask.to(torch.int64), dim=0) - 1
        chain_starts = torch.nonzero(chain_start_mask, as_tuple=False).flatten()
        chain_ends = torch.cat((chain_starts[1:], torch.tensor([row_count])))
        chain_lengths = chain_ends - chain_starts

        ordered_depths = depths
        horizon_segment_ids = chain_ids * (self.capacity + 1) + torch.div(
            ordered_depths, horizon, rounding_mode="floor"
        )
        segment_start_mask = torch.ones(row_count, dtype=torch.bool)
        segment_start_mask[1:] = horizon_segment_ids[1:] != horizon_segment_ids[:-1]
        segment_starts = torch.nonzero(segment_start_mask, as_tuple=False).flatten()
        segment_ends = torch.cat((segment_starts[1:], torch.tensor([row_count])))
        segment_lengths = segment_ends - segment_starts

        horizon_cut_count = int(
            torch.sum((ordered_depths + 1).remainder(horizon) == 0).item()
        )
        horizon_cut_count -= int(
            torch.sum(
                (ordered_depths + 1).remainder(horizon).eq(0)
                & (successors[ordered_indices] < 0)
            ).item()
        )
        segment_count = segment_lengths.numel()
        segment_order = torch.randperm(segment_count)
        row_segment_ids = torch.cumsum(segment_start_mask.to(torch.int64), dim=0) - 1
        segment_ranks = torch.empty(segment_count, dtype=torch.int64)
        segment_ranks[segment_order] = torch.arange(segment_count, dtype=torch.int64)
        row_positions = torch.arange(row_count, dtype=torch.int64)
        row_segment_starts = segment_starts[row_segment_ids]
        within_segment = row_positions - row_segment_starts
        maximum_segment_length = int(segment_lengths.max())
        shuffle_keys = (
            segment_ranks[row_segment_ids] * (maximum_segment_length + 1) + within_segment
        )
        shuffled_order = torch.argsort(shuffle_keys)
        shuffled_indices = ordered_indices[shuffled_order]
        shuffled_segment_lengths = segment_lengths[segment_order]
        shuffled_segment_ends = torch.cumsum(shuffled_segment_lengths, dim=0)
        if logical_batch_size < row_count:
            logical_boundaries = torch.arange(
                logical_batch_size,
                row_count,
                logical_batch_size,
                dtype=torch.int64,
            )
        else:
            logical_boundaries = torch.zeros(0, dtype=torch.int64)
        all_boundaries = torch.unique(
            torch.cat((torch.tensor([0, row_count]), shuffled_segment_ends, logical_boundaries)),
            sorted=True,
        )
        logical_cut_count = int(
            torch.sum(~torch.isin(logical_boundaries, shuffled_segment_ends)).item()
        )

        plans = []
        for batch_start in range(0, row_count, logical_batch_size):
            batch_end = min(batch_start + logical_batch_size, row_count)
            batch_indices = shuffled_indices[batch_start:batch_end].clone()
            batch_cuts = all_boundaries[
                (all_boundaries >= batch_start) & (all_boundaries <= batch_end)
            ] - batch_start
            plans.append((batch_indices, batch_cuts.clone()))

        suffix_lengths = self._suffix_lengths(all_boundaries, row_count)
        self.last_suffix_length_counts = torch.bincount(
            suffix_lengths, minlength=int(suffix_lengths.max()) + 1
        ).to(dtype=torch.int64, device="cpu")
        age_values = self._insertion_clock - self._insertion_steps[resident_indices]
        chain_lengths_sorted = chain_lengths.to(dtype=torch.float32)
        suffix_lengths_sorted = suffix_lengths.to(dtype=torch.float32)
        terminals_count = int(self._terminals[resident_indices].sum())
        truncations_count = int(self._truncations[resident_indices].sum())
        chain_end_count = int(torch.sum(successors[resident_indices] < 0).item())

        self.last_plan_metrics = {
            "row_count": int(row_count),
            "segment_count": int(segment_count),
            "logical_batch_count": len(plans),
            "sequence_length_mean": float(chain_lengths_sorted.mean()),
            "sequence_length_median": float(torch.quantile(chain_lengths_sorted, 0.5)),
            "sequence_length_p90": float(torch.quantile(chain_lengths_sorted, 0.9)),
            "sequence_length_max": int(chain_lengths.max()),
            "segment_length_mean": float(segment_lengths.to(torch.float32).mean()),
            "segment_length_max": int(segment_lengths.max()),
            "target_suffix_length_mean": float(suffix_lengths_sorted.mean()),
            "target_suffix_length_median": float(torch.quantile(suffix_lengths_sorted, 0.5)),
            "target_suffix_length_p90": float(torch.quantile(suffix_lengths_sorted, 0.9)),
            "target_suffix_length_max": int(suffix_lengths.max()),
            "target_suffix_fraction_ge_32": float(torch.mean((suffix_lengths >= 32).float())),
            "target_suffix_fraction_ge_64": float(torch.mean((suffix_lengths >= 64).float())),
            "target_suffix_fraction_ge_128": float(torch.mean((suffix_lengths >= 128).float())),
            "cut_terminal_count": terminals_count,
            "cut_truncation_count": truncations_count,
            "cut_unlinked_chain_end_count": chain_end_count - terminals_count - truncations_count,
            "cut_overwrite_count": int(self._overwrite_cuts_since_plan),
            "cut_gap_count": int(self._gap_cuts_since_plan),
            "cut_stale_generation_count": int(self._stale_link_cuts_since_plan),
            "cut_horizon_count": horizon_cut_count,
            "cut_logical_batch_count": logical_cut_count,
            "replay_age_mean": float(age_values.to(torch.float32).mean()),
            "replay_age_median": float(torch.quantile(age_values.to(torch.float32), 0.5)),
            "replay_age_p90": float(torch.quantile(age_values.to(torch.float32), 0.9)),
            "replay_age_max": int(age_values.max()),
        }
        return plans

    def _write_storage(
        self,
        slots,
        rows,
        observations,
        actions,
        rewards,
        next_observations,
        discounts,
    ):
        destination = slots.to(device=self.device)
        source = rows.to(device=self.device)
        self._replay.observations[destination] = observations[source]
        self._replay.actions[destination] = actions[source]
        self._replay.rewards[destination] = rewards[source]
        self._replay.next_observations[destination] = next_observations[source]
        self._replay.discounts[destination] = discounts[source]

    def _evict_slots(self, slots):
        slots = torch.unique(slots.to(dtype=torch.int64, device="cpu"))
        if slots.numel() == 0:
            return
        active_slots = slots[self._valid[slots]]
        if active_slots.numel() == 0:
            return
        old_generations = self._generation[active_slots]
        previous_indices = self._previous_indices[active_slots]
        previous_generations = self._previous_generations[active_slots]
        next_indices = self._next_indices[active_slots]
        next_generations = self._next_generations[active_slots]

        previous_exists = previous_indices >= 0
        if bool(torch.any(previous_indices[previous_exists] >= self.capacity)):
            raise RuntimeError("sequence replay predecessor index is out of range")
        safe_previous = previous_indices.clamp_min(0)
        previous_generation_matches = previous_exists & (
            self._generation[safe_previous] == previous_generations
        )
        if bool(previous_generation_matches.any()):
            predecessor_slots = safe_previous[previous_generation_matches]
            expected_slots = active_slots[previous_generation_matches]
            expected_generations = old_generations[previous_generation_matches]
            if bool(
                torch.any(
                    (self._next_indices[predecessor_slots] != expected_slots)
                    | (self._next_generations[predecessor_slots] != expected_generations)
                )
            ):
                raise RuntimeError("malformed sequence replay predecessor link")
            self._next_indices[predecessor_slots] = -1
            self._next_generations[predecessor_slots] = -1
            predecessor_is_evicted = torch.isin(predecessor_slots, active_slots)
            self._overwrite_cuts_since_plan += int((~predecessor_is_evicted).sum())

        next_exists = next_indices >= 0
        if bool(torch.any(next_indices[next_exists] >= self.capacity)):
            raise RuntimeError("sequence replay successor index is out of range")
        safe_next = next_indices.clamp_min(0)
        next_generation_matches = next_exists & (
            self._generation[safe_next] == next_generations
        )
        if bool(next_generation_matches.any()):
            successor_slots = safe_next[next_generation_matches]
            expected_slots = active_slots[next_generation_matches]
            expected_generations = old_generations[next_generation_matches]
            if bool(
                torch.any(
                    (self._previous_indices[successor_slots] != expected_slots)
                    | (self._previous_generations[successor_slots] != expected_generations)
                )
            ):
                raise RuntimeError("malformed sequence replay successor link")
            self._previous_indices[successor_slots] = -1
            self._previous_generations[successor_slots] = -1
            self._overwrite_cuts_since_plan += int(next_generation_matches.sum())

        row_agents = self._agent_ids[active_slots]
        tail_matches = (self._tail_indices[row_agents] == active_slots) & (
            self._tail_generations[row_agents] == old_generations
        )
        if bool(tail_matches.any()):
            stale_tail_agents = row_agents[tail_matches]
            self._tail_indices[stale_tail_agents] = -1
            self._tail_generations[stale_tail_agents] = -1

        self._previous_indices[active_slots] = -1
        self._previous_generations[active_slots] = -1
        self._next_indices[active_slots] = -1
        self._next_generations[active_slots] = -1
        self._valid[active_slots] = False

    def _validated_link_graph(self):
        valid_indices = torch.nonzero(self._valid, as_tuple=False).flatten()
        successors = torch.full((self.capacity,), -1, dtype=torch.int64)
        predecessors = torch.full((self.capacity,), -1, dtype=torch.int64)
        stale_link_count = 0

        for linked_indices, linked_generations, direction in (
            (self._next_indices, self._next_generations, "successor"),
            (self._previous_indices, self._previous_generations, "predecessor"),
        ):
            row_links = linked_indices[valid_indices]
            row_generations = linked_generations[valid_indices]
            has_link = row_links >= 0
            if bool(torch.any(row_links < -1)) or bool(
                torch.any((~has_link) & (row_generations != -1))
            ):
                raise RuntimeError(f"malformed sequence replay {direction} sentinel")
            if bool(torch.any(row_links[has_link] >= self.capacity)):
                raise RuntimeError(f"sequence replay {direction} index is out of range")

            safe_links = row_links.clamp_min(0)
            generation_matches = has_link & (
                self._generation[safe_links] == row_generations
            )
            stale_links = has_link & ~generation_matches
            stale_link_count += int(stale_links.sum())
            if bool(stale_links.any()):
                stale_rows = valid_indices[stale_links]
                linked_indices[stale_rows] = -1
                linked_generations[stale_rows] = -1
            source_indices = valid_indices[generation_matches]
            destination_indices = row_links[generation_matches]
            destination_generations = row_generations[generation_matches]
            if destination_indices.numel() == 0:
                continue
            if bool(torch.any(~self._valid[destination_indices])):
                raise RuntimeError(f"sequence replay {direction} points to an empty row")
            if bool(torch.any(self._agent_ids[destination_indices] != self._agent_ids[source_indices])):
                raise RuntimeError(f"sequence replay {direction} crosses agent IDs")

            if direction == "successor":
                if bool(torch.any(self._terminals[source_indices] | self._truncations[source_indices])):
                    raise RuntimeError("sequence replay links through an episode boundary")
                reciprocal_indices = self._previous_indices[destination_indices]
                reciprocal_generations = self._previous_generations[destination_indices]
                expected_generations = self._generation[source_indices]
                if bool(
                    torch.any(
                        (reciprocal_indices != source_indices)
                        | (reciprocal_generations != expected_generations)
                    )
                ):
                    raise RuntimeError("malformed sequence replay successor reciprocity")
                destination_counts = torch.bincount(
                    destination_indices, minlength=self.capacity
                )
                if bool(torch.any(destination_counts > 1)):
                    raise RuntimeError("sequence replay row has multiple predecessors")
                successors[source_indices] = destination_indices
                predecessors[destination_indices] = source_indices
            else:
                if bool(torch.any(self._terminals[destination_indices] | self._truncations[destination_indices])):
                    raise RuntimeError("sequence replay links through an episode boundary")
                reciprocal_indices = self._next_indices[destination_indices]
                reciprocal_generations = self._next_generations[destination_indices]
                expected_generations = self._generation[source_indices]
                if bool(
                    torch.any(
                        (reciprocal_indices != source_indices)
                        | (reciprocal_generations != expected_generations)
                    )
                ):
                    raise RuntimeError("malformed sequence replay predecessor reciprocity")

        return successors, predecessors, stale_link_count

    def _ordered_chain_rows(self, resident_indices, successors, predecessors):
        row_count = resident_indices.numel()
        ancestor = predecessors.clone()
        roots = torch.arange(self.capacity, dtype=torch.int64)
        depths = torch.zeros(self.capacity, dtype=torch.int64)
        depths[resident_indices] = (predecessors[resident_indices] >= 0).to(dtype=torch.int64)
        for _ in range(max(1, math.ceil(math.log2(self.capacity + 1)))):
            active = ancestor >= 0
            safe_ancestor = ancestor.clamp_min(0)
            depths += torch.where(active, depths[safe_ancestor], torch.zeros_like(depths))
            roots = torch.where(active, roots[safe_ancestor], roots)
            ancestor = torch.where(active, ancestor[safe_ancestor], -1)
        if bool(torch.any(ancestor[resident_indices] >= 0)):
            raise RuntimeError("sequence replay contains a cycle or overlong chain")
        roots = roots[resident_indices]
        depths = depths[resident_indices]
        order = torch.argsort(roots * self.capacity + depths)
        ordered_indices = resident_indices[order]
        ordered_roots = roots[order]
        ordered_depths = depths[order]
        same_root = ordered_roots[1:] == ordered_roots[:-1]
        if row_count > 1 and bool(
            torch.any(same_root & (ordered_depths[1:] != ordered_depths[:-1] + 1))
        ):
            raise RuntimeError("sequence replay chain positions are not contiguous")
        if bool(
            torch.any(
                successors[ordered_indices[:-1]][same_root]
                != ordered_indices[1:][same_root]
            )
        ):
            raise RuntimeError("sequence replay contains a broken chain ordering")
        return ordered_indices, ordered_roots, ordered_depths

    def _suffix_lengths(self, boundaries, row_count):
        row_positions = torch.arange(row_count, dtype=torch.int64)
        segment_ends = boundaries[1:]
        segment_ids = torch.searchsorted(segment_ends, row_positions, right=True)
        return segment_ends[segment_ids] - row_positions

    def _empty_metrics(self):
        return {
            "row_count": 0,
            "segment_count": 0,
            "logical_batch_count": 0,
            "sequence_length_mean": 0.0,
            "sequence_length_median": 0.0,
            "sequence_length_p90": 0.0,
            "sequence_length_max": 0,
            "segment_length_mean": 0.0,
            "segment_length_max": 0,
            "target_suffix_length_mean": 0.0,
            "target_suffix_length_median": 0.0,
            "target_suffix_length_p90": 0.0,
            "target_suffix_length_max": 0,
            "target_suffix_fraction_ge_32": 0.0,
            "target_suffix_fraction_ge_64": 0.0,
            "target_suffix_fraction_ge_128": 0.0,
            "cut_terminal_count": 0,
            "cut_truncation_count": 0,
            "cut_unlinked_chain_end_count": 0,
            "cut_overwrite_count": int(self._overwrite_cuts_since_plan),
            "cut_gap_count": int(self._gap_cuts_since_plan),
            "cut_stale_generation_count": int(self._stale_link_cuts_since_plan),
            "cut_horizon_count": 0,
            "cut_logical_batch_count": 0,
            "replay_age_mean": 0.0,
            "replay_age_median": 0.0,
            "replay_age_p90": 0.0,
            "replay_age_max": 0,
        }

    def _agent_id_tensor(self, values):
        agent_ids = torch.as_tensor(values)
        if agent_ids.ndim != 1 or agent_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("agent_ids must be a one-dimensional integer tensor")
        agent_ids = agent_ids.detach().to(device="cpu", dtype=torch.int64)
        if agent_ids.numel() and bool(
            torch.any((agent_ids < 0) | (agent_ids >= self.num_agents))
        ):
            raise ValueError("agent_ids must be within the configured agent range")
        return agent_ids

    @staticmethod
    def _cpu_indices(indices, name):
        indices = torch.as_tensor(indices)
        if indices.ndim != 1 or indices.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"{name} must be a one-dimensional integer tensor")
        return indices.detach().to(device="cpu", dtype=torch.int64)
