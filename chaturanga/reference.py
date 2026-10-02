"""
Deliberately slow, obviously-correct reference engine for differential testing.

Written independently from the production engine (moves.py).
Uses naive 8x8 iteration, one function per piece, no optimization.
Written directly from RULES.md.
"""

import numpy as np
from .board import (
    State, WHITE, BLACK, EMPTY,
    PADATI, ASHVA, GAJA, RATHA, MANTRI, RAJA,
    rank_of, file_of, sq, on_board,
)
from .moves import encode_action


def ref_generate_moves(state: State) -> list[int]:
    """
    Reference move generation — deliberately simple and obviously correct.
    One function per piece type, no optimization.
    """
    board = state.board
    side = state.side_to_move
    moves = []
    
    for square in range(64):
        piece = board[square]
        if piece == EMPTY:
            continue
        
        # Check ownership
        if side == WHITE and piece <= 0:
            continue
        if side == BLACK and piece >= 0:
            continue
        
        piece_type = abs(int(piece))
        
        if piece_type == PADATI:
            _ref_padati(board, square, side, moves)
        elif piece_type == ASHVA:
            _ref_ashva(board, square, side, moves)
        elif piece_type == GAJA:
            _ref_gaja(board, square, side, moves)
        elif piece_type == RATHA:
            _ref_ratha(board, square, side, moves)
        elif piece_type == MANTRI:
            _ref_mantri(board, square, side, moves)
        elif piece_type == RAJA:
            _ref_raja(board, square, side, moves)
    
    return sorted(moves)


def _is_enemy(board, sq_idx, side):
    """Check if the piece at sq_idx is an enemy piece."""
    p = board[sq_idx]
    if p == EMPTY:
        return False
    if side == WHITE:
        return p < 0
    else:
        return p > 0


def _is_friendly(board, sq_idx, side):
    """Check if the piece at sq_idx is a friendly piece."""
    p = board[sq_idx]
    if p == EMPTY:
        return False
    if side == WHITE:
        return p > 0
    else:
        return p < 0


def _ref_padati(board, s, side, moves):
    """Padati: one step forward, diagonal capture. No double step. No en passant."""
    r, f = rank_of(s), file_of(s)
    direction = 1 if side == WHITE else -1
    
    # Forward (non-capture)
    nr = r + direction
    if 0 <= nr < 8:
        target = sq(nr, f)
        if board[target] == EMPTY:
            moves.append(encode_action(s, target))
    
    # Diagonal captures
    for df in [-1, 1]:
        nf = f + df
        nr = r + direction
        if 0 <= nr < 8 and 0 <= nf < 8:
            target = sq(nr, nf)
            if _is_enemy(board, target, side):
                moves.append(encode_action(s, target))


def _ref_ashva(board, s, side, moves):
    """Ashva: standard knight leap."""
    r, f = rank_of(s), file_of(s)
    knight_offsets = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
    ]
    for dr, df in knight_offsets:
        nr, nf = r + dr, f + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            target = sq(nr, nf)
            if not _is_friendly(board, target, side):
                moves.append(encode_action(s, target))


def _ref_gaja(board, s, side, moves):
    """Gaja: exactly two squares diagonally, jumping. Cannot move one square."""
    r, f = rank_of(s), file_of(s)
    for dr, df in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
        nr, nf = r + dr, f + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            target = sq(nr, nf)
            if not _is_friendly(board, target, side):
                moves.append(encode_action(s, target))


def _ref_ratha(board, s, side, moves):
    """Ratha: any distance orthogonally, blocked by first occupied square."""
    r, f = rank_of(s), file_of(s)
    for dr, df in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        cr, cf = r + dr, f + df
        while 0 <= cr < 8 and 0 <= cf < 8:
            target = sq(cr, cf)
            p = board[target]
            if p == EMPTY:
                moves.append(encode_action(s, target))
            elif _is_enemy(board, target, side):
                moves.append(encode_action(s, target))
                break
            else:
                # Friendly piece, blocked
                break
            cr += dr
            cf += df


def _ref_mantri(board, s, side, moves):
    """Mantri: exactly one square diagonally."""
    r, f = rank_of(s), file_of(s)
    for dr, df in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nf = r + dr, f + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            target = sq(nr, nf)
            if not _is_friendly(board, target, side):
                moves.append(encode_action(s, target))


def _ref_raja(board, s, side, moves):
    """Raja: one square in any direction. NO CHECK FILTERING."""
    r, f = rank_of(s), file_of(s)
    for dr in [-1, 0, 1]:
        for df in [-1, 0, 1]:
            if dr == 0 and df == 0:
                continue
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                target = sq(nr, nf)
                if not _is_friendly(board, target, side):
                    moves.append(encode_action(s, target))


def ref_perft(state: State, depth: int) -> int:
    """Reference perft — uses ref_generate_moves instead of the production engine."""
    from .terminal import is_terminal
    from .moves import apply_move
    
    if depth == 0:
        return 1
    
    done, _ = is_terminal(state)
    if done:
        return 0
    
    moves = ref_generate_moves(state)
    if not moves:
        return 0
    
    count = 0
    for action in moves:
        child = apply_move(state, action)
        count += ref_perft(child, depth - 1)
    
    return count
