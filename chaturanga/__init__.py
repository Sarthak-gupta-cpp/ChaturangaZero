"""
Chaturanga game engine package.
"""

from .board import State, WHITE, BLACK, PADATI, ASHVA, GAJA, RATHA, MANTRI, RAJA
from .moves import generate_moves, apply_move, legal_mask, encode_action, decode_action
from .terminal import is_terminal
from .encoding import encode_state, mirror_state, MIRROR_PERM, NUM_PLANES

__all__ = [
    'State', 'WHITE', 'BLACK',
    'PADATI', 'ASHVA', 'GAJA', 'RATHA', 'MANTRI', 'RAJA',
    'generate_moves', 'apply_move', 'legal_mask',
    'encode_action', 'decode_action',
    'is_terminal',
    'encode_state', 'mirror_state', 'MIRROR_PERM', 'NUM_PLANES',
]
