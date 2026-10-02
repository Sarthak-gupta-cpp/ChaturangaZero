"""
Replay buffer and mirror augmentation.

- Position-level sampling, NOT game-level (§3.3)
- Deque-based with configurable max size
- Mirror augmentation doubles effective data for free (§3.5)
"""

import numpy as np
from collections import deque
from typing import Optional
import random


class ReplayBuffer:
    """
    Replay buffer storing (state_encoding, policy_target, value_target) triples.
    
    Samples at the POSITION level, not the game level — each position is
    equally likely to be drawn regardless of which game it came from.
    """
    
    def __init__(self, max_size: int = 1_000_000):
        self.max_size = max_size
        self.states: deque = deque(maxlen=max_size)
        self.policies: deque = deque(maxlen=max_size)
        self.values: deque = deque(maxlen=max_size)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add a single (state, policy, value) triple."""
        self.states.append(state)
        self.policies.append(policy)
        self.values.append(value)
    
    def add_game(
        self,
        states: list[np.ndarray],
        policies: list[np.ndarray],
        result: float,
        mirror_augment: bool = True,
        mirror_perm: Optional[np.ndarray] = None,
    ):
        """
        Add all positions from a completed game.
        
        Args:
            states: list of encoded board states (C, H, W)
            policies: list of policy targets π (action_size,)
            result: game outcome z (+1 win, -1 loss, 0 draw) from player 1's perspective
            mirror_augment: whether to also add mirrored positions
            mirror_perm: action index permutation for mirroring
        """
        # Assign values: z from the perspective of the side that moved at each position.
        # Positions alternate sides, so signs alternate.
        for i, (s, pi) in enumerate(zip(states, policies)):
            # Position i was played by the side who moved at ply i.
            # If i is even, it was player 1's move (result = z)
            # If i is odd, it was player 2's move (result = -z)
            z = result if i % 2 == 0 else -result
            self.add(s, pi, z)
            
            # Mirror augmentation
            if mirror_augment and mirror_perm is not None:
                mirrored_state = np.flip(s, axis=2).copy()  # Flip along file axis
                mirrored_policy = pi[mirror_perm]
                self.add(mirrored_state, mirrored_policy, z)
    
    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of positions.
        
        Returns:
            states: (B, C, H, W) float32
            policies: (B, action_size) float32
            values: (B, 1) float32
        """
        n = len(self)
        if n < batch_size:
            batch_size = n
        
        indices = random.sample(range(n), batch_size)
        
        batch_states = np.array([self.states[i] for i in indices], dtype=np.float32)
        batch_policies = np.array([self.policies[i] for i in indices], dtype=np.float32)
        batch_values = np.array([[self.values[i]] for i in indices], dtype=np.float32)
        
        return batch_states, batch_policies, batch_values
    
    def age_histogram(self, n_bins: int = 10) -> list[int]:
        """
        Return a histogram of buffer entry ages (for diagnostics).
        Newer entries have lower indices in the deque.
        """
        n = len(self)
        if n == 0:
            return [0] * n_bins
        
        bin_size = max(1, n // n_bins)
        counts = []
        for i in range(n_bins):
            start = i * bin_size
            end = min((i + 1) * bin_size, n)
            counts.append(end - start)
        return counts
    
    def save(self, path: str):
        """Save buffer to disk."""
        np.savez_compressed(
            path,
            states=np.array(list(self.states)),
            policies=np.array(list(self.policies)),
            values=np.array(list(self.values)),
        )
    
    def load(self, path: str):
        """Load buffer from disk."""
        data = np.load(path)
        self.states.clear()
        self.policies.clear()
        self.values.clear()
        
        for s, p, v in zip(data['states'], data['policies'], data['values']):
            self.states.append(s)
            self.policies.append(p)
            self.values.append(float(v))
