"""
Pseudo-legal move generation for Chaturanga.

THERE IS NO CHECK. Every pseudo-legal move IS legal.
No is_square_attacked() function exists or should ever exist.
"""

import numpy as np
from .board import (
    State, WHITE, BLACK, EMPTY,
    PADATI, ASHVA, GAJA, RATHA, MANTRI, RAJA,
    rank_of, file_of, sq, on_board, MOVE_CAP,
)


def encode_action(from_sq: int, to_sq: int) -> int:
    """Encode a move as action = from_square * 64 + to_square."""
    return from_sq * 64 + to_sq


def decode_action(action: int) -> tuple[int, int]:
    """Decode action into (from_square, to_square)."""
    return action // 64, action % 64


def _gen_padati_moves(board: np.ndarray, s: int, side: int, moves: list[int]):
    """Generate Padati (pawn) moves. Forward one step, diagonal captures, auto-promote."""
    r, f = rank_of(s), file_of(s)
    forward = 1 if side == WHITE else -1
    nr = r + forward
    
    if not (0 <= nr < 8):
        return
    
    # Forward move (non-capture)
    target_sq = sq(nr, f)
    if board[target_sq] == EMPTY:
        moves.append(encode_action(s, target_sq))
    
    # Diagonal captures
    for df in (-1, 1):
        nf = f + df
        if 0 <= nf < 8:
            target_sq = sq(nr, nf)
            target_piece = board[target_sq]
            # Can capture enemy pieces only
            if target_piece != EMPTY and (target_piece > 0) != (side > 0):
                moves.append(encode_action(s, target_sq))


def _gen_ashva_moves(board: np.ndarray, s: int, side: int, moves: list[int]):
    """Generate Ashva (knight) moves. Standard knight leap."""
    r, f = rank_of(s), file_of(s)
    for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nr, nf = r + dr, f + df
        if on_board(nr, nf):
            target_sq = sq(nr, nf)
            target_piece = board[target_sq]
            if target_piece == EMPTY or (target_piece > 0) != (side > 0):
                moves.append(encode_action(s, target_sq))


def _gen_gaja_moves(board: np.ndarray, s: int, side: int, moves: list[int]):
    """Generate Gaja (elephant) moves. Exactly two squares diagonally, jumping."""
    r, f = rank_of(s), file_of(s)
    for dr, df in [(-2,-2),(-2,2),(2,-2),(2,2)]:
        nr, nf = r + dr, f + df
        if on_board(nr, nf):
            target_sq = sq(nr, nf)
            target_piece = board[target_sq]
            if target_piece == EMPTY or (target_piece > 0) != (side > 0):
                moves.append(encode_action(s, target_sq))


def _gen_ratha_moves(board: np.ndarray, s: int, side: int, moves: list[int]):
    """Generate Ratha (chariot/rook) moves. Sliding orthogonally."""
    r, f = rank_of(s), file_of(s)
    for dr, df in [(0,1),(0,-1),(1,0),(-1,0)]:
        nr, nf = r + dr, f + df
        while on_board(nr, nf):
            target_sq = sq(nr, nf)
            target_piece = board[target_sq]
            if target_piece == EMPTY:
                moves.append(encode_action(s, target_sq))
            else:
                # Blocked: capture if enemy, then stop
                if (target_piece > 0) != (side > 0):
                    moves.append(encode_action(s, target_sq))
                break
            nr += dr
            nf += df


def _gen_mantri_moves(board: np.ndarray, s: int, side: int, moves: list[int]):
    """Generate Mantri (minister) moves. One square diagonally."""
    r, f = rank_of(s), file_of(s)
    for dr, df in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nf = r + dr, f + df
        if on_board(nr, nf):
            target_sq = sq(nr, nf)
            target_piece = board[target_sq]
            if target_piece == EMPTY or (target_piece > 0) != (side > 0):
                moves.append(encode_action(s, target_sq))


def _gen_raja_moves(board: np.ndarray, s: int, side: int, moves: list[int]):
    """
    Generate Raja (king) moves. One square in any direction.
    NO CHECK FILTERING. The Raja may legally step onto guarded squares.
    """
    r, f = rank_of(s), file_of(s)
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr, nf = r + dr, f + df
            if on_board(nr, nf):
                target_sq = sq(nr, nf)
                target_piece = board[target_sq]
                if target_piece == EMPTY or (target_piece > 0) != (side > 0):
                    moves.append(encode_action(s, target_sq))


# Dispatch table: piece_type -> generator function
_GENERATORS = {
    PADATI: _gen_padati_moves,
    ASHVA: _gen_ashva_moves,
    GAJA: _gen_gaja_moves,
    RATHA: _gen_ratha_moves,
    MANTRI: _gen_mantri_moves,
    RAJA: _gen_raja_moves,
}


def generate_moves(state: State) -> list[int]:
    """
    Generate all pseudo-legal moves for the side to move.
    
    PSEUDO-LEGAL == LEGAL in this ruleset (no check).
    Returns a list of action IDs in [0, 4096).
    """
    board = state.board
    side = state.side_to_move
    moves: list[int] = []
    
    for s in range(64):
        piece = board[s]
        if piece == EMPTY:
            continue
        # Check if piece belongs to the side to move
        if (piece > 0 and side == WHITE) or (piece < 0 and side == BLACK):
            piece_type = abs(piece)
            _GENERATORS[piece_type](board, s, side, moves)
    
    return moves


def apply_move(state: State, action: int) -> 'State':
    """
    Apply a move and return a NEW state. Does not mutate the original.
    
    Handles:
    - Normal moves and captures
    - Padati promotion to Mantri on last rank (forced, no choice)
    """
    from_sq, to_sq = decode_action(action)
    
    new_board = state.board.copy()
    piece = new_board[from_sq]
    
    # Move piece
    new_board[to_sq] = piece
    new_board[from_sq] = EMPTY
    
    # Check for Padati promotion
    piece_type = abs(int(piece))
    if piece_type == PADATI:
        to_rank = rank_of(to_sq)
        # White promotes on rank 7, Black promotes on rank 0
        if (state.side_to_move == WHITE and to_rank == 7) or \
           (state.side_to_move == BLACK and to_rank == 0):
            # Promote to Mantri
            new_board[to_sq] = MANTRI * state.side_to_move
    
    new_state = State.__new__(State)
    new_state.board = new_board
    new_state._side = -state.side_to_move
    new_state._ply = state.ply_count + 1
    new_state._hash = None
    
    return new_state


def legal_mask(state: State) -> np.ndarray:
    """
    Return a boolean mask of shape (4096,) with True at legal action indices.
    """
    mask = np.zeros(4096, dtype=bool)
    for action in generate_moves(state):
        mask[action] = True
    return mask
