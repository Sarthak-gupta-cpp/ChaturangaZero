"""Eval package — Elo, plots, human-play CLI."""

from .elo import compute_elo, round_robin_elo, generate_plots, plot_elo_curve

__all__ = ['compute_elo', 'round_robin_elo', 'generate_plots', 'plot_elo_curve']
