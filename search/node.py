"""
MCTS Node — per-edge statistics stored as flat arrays.

Each Node stores:
- N[a]: visit count for each child action
- W[a]: total value accumulated through action a
- Q[a]: mean value = W[a] / N[a]
- P[a]: prior probability from the policy network
"""

import numpy as np
from typing import Optional, Hashable


class Node:
    """
    MCTS tree node with per-edge statistics.
    
    Uses flat numpy arrays for N, W, P indexed by action index
    to minimize Python object overhead.
    """
    __slots__ = (
        'state', 'parent', 'parent_action',
        'N', 'W', 'P',
        'children', 'legal_actions', 'is_expanded',
        'is_terminal', 'terminal_value',
    )
    
    def __init__(self, state, parent: Optional['Node'] = None,
                 parent_action: int = -1, action_size: int = 4096):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        
        self.N = np.zeros(action_size, dtype=np.float32)
        self.W = np.zeros(action_size, dtype=np.float32)
        self.P = np.zeros(action_size, dtype=np.float32)
        
        self.children: dict[int, 'Node'] = {}
        self.legal_actions: list[int] = []
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0
    
    @property
    def total_visits(self) -> float:
        return float(np.sum(self.N))
    
    def Q(self, action: int) -> float:
        """Mean value for an action."""
        if self.N[action] == 0:
            return 0.0
        return self.W[action] / self.N[action]
    
    def best_child_by_visits(self) -> int:
        """Return the action with the most visits (for move selection)."""
        # Only consider legal actions
        best_action = self.legal_actions[0]
        best_n = self.N[best_action]
        for a in self.legal_actions[1:]:
            if self.N[a] > best_n:
                best_n = self.N[a]
                best_action = a
        return best_action
    
    def visit_distribution(self) -> np.ndarray:
        """
        Return visit counts as a distribution over the full action space.
        This becomes the training target π.
        """
        return self.N.copy()
