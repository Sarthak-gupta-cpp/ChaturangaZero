"""
Phase 0 — Connect-4 pipeline proof.

Runs the entire AlphaZero loop on Connect-4 to validate:
- MCTS works correctly
- Network trains and improves
- Self-play generates valid data
- Arena gating functions

Goal: agent beats random >95% in under 1 GPU-hour.
This script is designed to be fast — should show improvement in ~10 minutes.
"""

import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from connect4.game import Connect4State
from net.model import create_connect4_net
from net.replay import ReplayBuffer
from net.train import Trainer, create_evaluator
from search.mcts import search, select_action_with_temperature
from search.baselines import random_player
from search.arena import arena


def run_phase0(
    n_iterations: int = 30,
    games_per_iter: int = 50,
    n_sims: int = 25,
    steps_per_iter: int = 200,
    batch_size: int = 128,
):
    """
    Run the Phase 0 Connect-4 pipeline proof.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 0 — Connect-4 Pipeline Proof")
    print(f"Device: {device}")
    print(f"{'='*50}")
    
    # Create network
    net = create_connect4_net(n_blocks=3, channels=64)
    print(f"Network: {net.count_parameters():,} parameters")
    
    # Trainer
    trainer = Trainer(net, lr=2e-3, l2=1e-4, device=device)
    
    # Buffer
    buffer = ReplayBuffer(max_size=100_000)
    
    # Output dir
    os.makedirs('runs/connect4', exist_ok=True)
    
    total_start = time.time()
    
    for iteration in range(n_iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration+1}/{n_iterations} ---")
        
        # Self-play
        evaluate = create_evaluator(net, device=device, action_size=7)
        
        game_results = []
        for g in range(games_per_iter):
            state = Connect4State()
            states_enc = []
            policies = []
            
            while True:
                done, z = state.is_terminal()
                if done:
                    result = z if state.side_to_move == 1 else -z
                    break
                
                encoded = state.encode()
                vc = search(
                    state=state,
                    evaluate=evaluate,
                    n_sims=n_sims,
                    action_size=7,
                    c_puct=1.5,
                    dirichlet_alpha=1.0,
                    dirichlet_eps=0.25,
                    add_noise=True,
                )
                
                temperature = 1.0 if state.ply_count < 10 else 0.0
                action, pi = select_action_with_temperature(
                    vc, temperature, state.legal_moves()
                )
                
                states_enc.append(encoded)
                policies.append(pi)
                state = state.apply(action)
            
            # Add to buffer
            for i, (s, pi) in enumerate(zip(states_enc, policies)):
                v = result if i % 2 == 0 else -result
                buffer.add(s, pi, v)
                
                # Mirror augmentation
                mirrored_s = np.flip(s, axis=2).copy()
                mirrored_pi = pi[::-1].copy()  # Flip column order
                buffer.add(mirrored_s, mirrored_pi, v)
            
            game_results.append(result)
        
        wins = sum(1 for r in game_results if r > 0)
        losses = sum(1 for r in game_results if r < 0)
        draws = sum(1 for r in game_results if r == 0)
        print(f"  Self-play: {wins}W-{draws}D-{losses}L, buffer={len(buffer)}")
        
        # Training
        if len(buffer) >= batch_size:
            stats = trainer.train_epoch(buffer, steps=steps_per_iter, batch_size=batch_size)
            if stats:
                print(f"  Train: ploss={stats['mean_policy_loss']:.3f}, "
                      f"vloss={stats['mean_value_loss']:.3f}")
        
        # Evaluate vs random every 5 iterations
        if (iteration + 1) % 5 == 0:
            eval_fn = create_evaluator(net, device=device, action_size=7)
            
            def agent_player(state):
                vc = search(state, eval_fn, n_sims=n_sims,
                          action_size=7, c_puct=1.5, add_noise=False)
                return int(np.argmax(vc))
            
            def rand_player(state):
                return random_player(state, action_size=7)
            
            result = arena(
                agent_player, rand_player, Connect4State,
                n_games=20, action_size=7,
            )
            
            win_rate = result['p1_win_rate']
            print(f"  vs Random: {result['p1_wins']}-{result['draws']}-{result['p2_wins']} "
                  f"({win_rate:.1%})")
            
            if win_rate >= 0.95:
                elapsed = time.time() - total_start
                print(f"\n{'='*50}")
                print(f"PHASE 0 GATE PASSED!")
                print(f"Agent beats random >{95}% (actual: {win_rate:.1%})")
                print(f"Time: {elapsed/60:.1f} minutes")
                print(f"{'='*50}")
                
                # Save checkpoint
                trainer.save_checkpoint(
                    'runs/connect4/phase0_passed.pt',
                    iteration=iteration,
                )
                return True
        
        iter_time = time.time() - iter_start
        print(f"  Iter time: {iter_time:.1f}s")
    
    # Final evaluation
    eval_fn = create_evaluator(net, device=device, action_size=7)
    
    def agent_player(state):
        vc = search(state, eval_fn, n_sims=n_sims,
                  action_size=7, c_puct=1.5, add_noise=False)
        return int(np.argmax(vc))
    
    def rand_player(state):
        return random_player(state, action_size=7)
    
    result = arena(agent_player, rand_player, Connect4State,
                  n_games=40, action_size=7)
    
    elapsed = time.time() - total_start
    win_rate = result['p1_win_rate']
    
    print(f"\n{'='*50}")
    print(f"Final vs Random: {result['p1_wins']}-{result['draws']}-{result['p2_wins']} "
          f"({win_rate:.1%})")
    print(f"Total time: {elapsed/60:.1f} minutes")
    
    if win_rate >= 0.95:
        print("PHASE 0 GATE PASSED!")
    else:
        print(f"Phase 0 gate not yet passed (need >95%, got {win_rate:.1%})")
        print("Increase iterations or sims and re-run.")
    
    print(f"{'='*50}")
    
    # Save final checkpoint
    trainer.save_checkpoint('runs/connect4/final.pt', iteration=n_iterations-1)
    
    return win_rate >= 0.95


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--sims', type=int, default=25)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    
    run_phase0(
        n_iterations=args.iterations,
        games_per_iter=args.games,
        n_sims=args.sims,
        steps_per_iter=args.steps,
    )
