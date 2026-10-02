"""
Baseline opponents for evaluation.

1. Random player — uniform random over legal moves
2. Greedy material player — picks the move that maximizes material advantage
3. Alpha-beta with material + mobility — 3-ply lookahead
"""

import numpy as np
from typing import Optional


def random_player(state, action_size: int = 4096) -> int:
    """Play a random legal move."""
    if hasattr(state, 'legal_moves'):
        moves = state.legal_moves()
    else:
        from chaturanga.moves import generate_moves
        moves = generate_moves(state)
    
    if not moves:
        return -1
    return int(np.random.choice(moves))


def greedy_material_player(state, action_size: int = 4096) -> int:
    """
    Pick the move that maximizes material advantage.
    Greedy: look one move ahead, pick the best capture or any non-capture.
    """
    if hasattr(state, 'legal_moves'):
        moves = state.legal_moves()
        apply_fn = state.apply
    else:
        from chaturanga.moves import generate_moves, apply_move
        moves = generate_moves(state)
        apply_fn = lambda a: apply_move(state, a)
    
    if not moves:
        return -1
    
    best_action = moves[0]
    best_score = -float('inf')
    side = state.side_to_move
    
    for action in moves:
        child = apply_fn(action) if hasattr(state, 'apply') else apply_move(state, action)
        
        # Check if terminal
        done, z = child.is_terminal()
        if done:
            # z is from child's side to move (opponent), so negate
            score = -z * 1000  # Large bonus for winning
        else:
            # Evaluate by material difference from our perspective
            if hasattr(child, 'material_score'):
                our_mat = child.material_score(side)
                their_mat = child.material_score(-side)
                score = our_mat - their_mat
            else:
                score = 0.0
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action


def alpha_beta_player(state, depth: int = 3, action_size: int = 4096) -> int:
    """
    Alpha-beta search with material + mobility evaluation.
    Default depth = 3 plies.
    """
    if hasattr(state, 'legal_moves'):
        moves = state.legal_moves()
    else:
        from chaturanga.moves import generate_moves
        moves = generate_moves(state)
    
    if not moves:
        return -1
    
    best_action = moves[0]
    best_score = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    
    for action in moves:
        child = _apply(state, action)
        score = -_alpha_beta(child, depth - 1, -beta, -alpha)
        if score > best_score:
            best_score = score
            best_action = action
        alpha = max(alpha, score)
    
    return best_action


def _alpha_beta(state, depth: int, alpha: float, beta: float) -> float:
    """Negamax alpha-beta search."""
    done, z = state.is_terminal()
    if done:
        return z * 1000  # Scale terminal values
    
    if depth <= 0:
        return _evaluate_position(state)
    
    if hasattr(state, 'legal_moves'):
        moves = state.legal_moves()
    else:
        from chaturanga.moves import generate_moves
        moves = generate_moves(state)
    
    if not moves:
        return 1000  # No legal moves = stalemate = win for side to move
    
    best = -float('inf')
    for action in moves:
        child = _apply(state, action)
        score = -_alpha_beta(child, depth - 1, -beta, -alpha)
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # Cutoff
    
    return best


def _evaluate_position(state) -> float:
    """
    Static evaluation: material + mobility.
    Returns score from the side-to-move's perspective.
    """
    side = state.side_to_move
    
    if hasattr(state, 'material_score'):
        # Chaturanga
        our_mat = state.material_score(side)
        their_mat = state.material_score(-side)
        material = our_mat - their_mat
    else:
        material = 0.0
    
    # Mobility: count legal moves
    if hasattr(state, 'legal_moves'):
        mobility = len(state.legal_moves()) * 0.05
    else:
        from chaturanga.moves import generate_moves
        mobility = len(generate_moves(state)) * 0.05
    
    return material + mobility


def _apply(state, action):
    """Apply action to state."""
    if hasattr(state, 'apply'):
        return state.apply(action)
    else:
        from chaturanga.moves import apply_move
        return apply_move(state, action)
