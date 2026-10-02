"""
Terminal condition detection for Chaturanga.

Terminal conditions evaluated in THIS EXACT ORDER (§2.1):
1. RAJA CAPTURED — S has no Raja → S loses (z = -1)
2. KING vs KING — both sides have only their Raja → draw (z = 0)
3. BARE KING — S has only their Raja, opponent has more → S loses (z = -1)
4. STALEMATE — S has no legal move → S WINS (z = +1)  ← THE SIGN TRAP
5. MOVE CAP — ply_count >= 300 → adjudicate on material

Order matters. Rule 2 must precede rule 3.
"""

from .board import State, WHITE, BLACK, RAJA, PIECE_VALUES, MOVE_CAP


def is_terminal(state: State) -> tuple[bool, float]:
    """
    Check if the position is terminal from the perspective of the side to move.
    
    Returns:
        (is_done, z) where z is the value for the side to move:
        +1 = side to move wins, -1 = side to move loses, 0 = draw.
    """
    side = state.side_to_move
    opp = -side
    
    # 1. RAJA CAPTURED — side to move has no Raja
    if not state.has_raja(side):
        return (True, -1.0)
    
    # 2. KING vs KING — both sides have only their Raja
    side_count = state.piece_count(side)
    opp_count = state.piece_count(opp)
    
    if side_count == 1 and opp_count == 1:
        # Both have exactly one piece, which must be the Raja (since we passed rule 1)
        return (True, 0.0)
    
    # 3. BARE KING — side to move has only their Raja, opponent has more
    if side_count == 1 and opp_count > 1:
        return (True, -1.0)
    
    # 4. STALEMATE — side to move has no legal move → WINS (z = +1)
    # Import here to avoid circular imports
    from .moves import generate_moves
    moves = generate_moves(state)
    if len(moves) == 0:
        return (True, 1.0)  # THE SIGN: +1 for the stalemated player
    
    # 5. MOVE CAP — training device
    if state.ply_count >= MOVE_CAP:
        z = _adjudicate_material(state)
        return (True, z)
    
    return (False, 0.0)


def _adjudicate_material(state: State) -> float:
    """
    Adjudicate on material when the move cap is reached.
    
    material = 5*Ratha + 3*Ashva + 1.5*Gaja + 1.5*Mantri + 1*Padati
    if |diff| >= 4: z = sign(diff) from side-to-move's perspective
    else: z = 0 (draw)
    """
    side = state.side_to_move
    opp = -side
    
    our_material = state.material_score(side)
    their_material = state.material_score(opp)
    
    diff = our_material - their_material
    
    if abs(diff) >= 4.0:
        return 1.0 if diff > 0 else -1.0
    else:
        return 0.0
