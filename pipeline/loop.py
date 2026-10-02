"""
Main training loop — orchestrates the AlphaZero self-play → train cycle.

1 iteration = N self-play games → M gradient steps → optional arena gate
"""

import os
import time
import json
import yaml
import numpy as np
import torch
from typing import Optional
from pathlib import Path

from net.model import DualHeadNet, create_chaturanga_net, create_connect4_net
from net.replay import ReplayBuffer
from net.train import Trainer, create_evaluator
from search.arena import gate_model
from search.baselines import random_player
from search.mcts import search, select_action_with_temperature
from pipeline.selfplay import run_self_play_batch


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def training_loop(config: dict, output_dir: str = 'runs', resume_from: str = None):
    """
    Main AlphaZero training loop.
    
    Args:
        config: configuration dict (from YAML)
        output_dir: directory for checkpoints and logs
        resume_from: path to checkpoint to resume from
    """
    # Setup
    game_type = config.get('game', 'chaturanga')
    seed = config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Game setup
    if game_type == 'connect4':
        from connect4.game import Connect4State
        
        net = create_connect4_net(
            n_blocks=config['net']['blocks'],
            channels=config['net']['channels'],
        )
        action_size = config['net']['action_space']
        create_state = Connect4State
        mirror_perm = None  # Connect-4 mirrors differently
        board_size = (6, 7)
    else:
        from chaturanga.board import State
        from chaturanga.encoding import MIRROR_PERM
        
        net = create_chaturanga_net(
            n_blocks=config['net']['blocks'],
            channels=config['net']['channels'],
        )
        action_size = config['net']['action_space']
        create_state = State
        mirror_perm = MIRROR_PERM
        board_size = (8, 8)
    
    print(f"Network: {net.count_parameters():,} parameters")
    
    # Create trainer
    trainer = Trainer(
        net=net,
        lr=config['train']['lr'],
        l2=config['train']['l2'],
        lr_milestones=config['train'].get('lr_milestones'),
        lr_gamma=config['train'].get('lr_gamma', 0.1),
        grad_clip=config['train'].get('grad_clip', 1.0),
        device=device,
    )
    
    # Create replay buffer
    buffer = ReplayBuffer(max_size=config['train']['buffer_size'])
    
    # Resume from checkpoint if specified
    start_iteration = 0
    best_net_path = None
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        checkpoint = trainer.load_checkpoint(resume_from)
        start_iteration = checkpoint.get('iteration', 0) + 1
        
        # Load buffer if available
        buffer_path = os.path.join(output_dir, 'buffer.npz')
        if os.path.exists(buffer_path):
            buffer.load(buffer_path)
            print(f"Loaded buffer with {len(buffer)} positions")
    
    # Training config
    n_iterations = config['loop']['iterations']
    games_per_iter = config['loop']['games_per_iter']
    steps_per_iter = config['train']['steps_per_iter']
    batch_size = config['train']['batch_size']
    n_sims = config['mcts']['sims']
    c_puct = config['mcts']['c_puct']
    temp_moves = config['play']['temp_moves']
    move_cap = config['play'].get('move_cap', 300)
    
    # Arena config
    arena_games = config['arena']['games']
    arena_every = config['arena']['every_n_iters']
    win_threshold = config['arena']['win_threshold']
    
    # Training metrics log
    metrics_log = []
    
    print(f"\n{'='*60}")
    print(f"Starting training: {n_iterations} iterations")
    print(f"  Games/iter: {games_per_iter}, Sims/move: {n_sims}")
    print(f"  Steps/iter: {steps_per_iter}, Batch: {batch_size}")
    print(f"  Buffer: {config['train']['buffer_size']:,}")
    print(f"{'='*60}\n")
    
    # Save initial model as the "best"
    best_net_path = os.path.join(output_dir, 'best.pt')
    trainer.save_checkpoint(best_net_path, iteration=0)
    
    for iteration in range(start_iteration, n_iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration+1}/{n_iterations} ---")
        
        # 1. Self-play
        print(f"Self-play: {games_per_iter} games, {n_sims} sims/move...")
        evaluate = create_evaluator(net, device=device, action_size=action_size)
        
        games = run_self_play_batch(
            n_games=games_per_iter,
            create_state=create_state,
            evaluate=evaluate,
            action_size=action_size,
            n_sims=n_sims,
            c_puct=c_puct,
            temp_moves=temp_moves,
            move_cap=move_cap,
            dirichlet_alpha=config['mcts']['dirichlet_alpha'],
            dirichlet_eps=config['mcts']['dirichlet_eps'],
            verbose=True,
        )
        
        # Add games to buffer
        mirror_augment = config['train'].get('mirror_augment', True)
        terminal_stats = {}
        total_plies = 0
        
        for game in games:
            buffer.add_game(
                states=game['states'],
                policies=game['policies'],
                result=game['result'],
                mirror_augment=mirror_augment,
                mirror_perm=mirror_perm,
            )
            reason = game['terminal_reason']
            terminal_stats[reason] = terminal_stats.get(reason, 0) + 1
            total_plies += game['ply_count']
        
        avg_plies = total_plies / len(games)
        cap_rate = terminal_stats.get('move_cap', 0) / len(games)
        
        print(f"  Buffer: {len(buffer)} positions")
        print(f"  Avg game length: {avg_plies:.1f} plies")
        print(f"  Terminal reasons: {terminal_stats}")
        print(f"  Cap rate: {cap_rate:.1%}")
        
        # 2. Training
        print(f"Training: {steps_per_iter} steps...")
        train_stats = trainer.train_epoch(
            buffer=buffer,
            steps=steps_per_iter,
            batch_size=batch_size,
        )
        
        if train_stats:
            print(f"  Policy loss: {train_stats['mean_policy_loss']:.4f}")
            print(f"  Value loss:  {train_stats['mean_value_loss']:.4f}")
            print(f"  Entropy:     {train_stats['mean_entropy']:.4f}")
        
        # Step LR scheduler
        trainer.step_scheduler()
        
        # 3. Save checkpoint
        ckpt_path = os.path.join(output_dir, f'checkpoint_{iteration:04d}.pt')
        trainer.save_checkpoint(ckpt_path, iteration=iteration, extra={
            'terminal_stats': terminal_stats,
            'avg_plies': avg_plies,
            'cap_rate': cap_rate,
        })
        
        # Save buffer periodically
        if (iteration + 1) % config.get('cloud', {}).get('buffer_snapshot_every', 10) == 0:
            buffer.save(os.path.join(output_dir, 'buffer.npz'))
        
        # 4. Arena gating
        if (iteration + 1) % arena_every == 0 and best_net_path:
            print(f"Arena: {arena_games} games...")
            
            # Load best network
            best_net = type(net)(
                input_planes=config['net']['input_planes'],
                board_h=board_size[0],
                board_w=board_size[1],
                action_size=action_size,
                n_blocks=config['net']['blocks'],
                channels=config['net']['channels'],
            )
            best_checkpoint = torch.load(best_net_path, map_location=device, weights_only=False)
            best_net.load_state_dict(best_checkpoint['model_state_dict'])
            best_net.to(device)
            
            best_eval = create_evaluator(best_net, device=device, action_size=action_size)
            challenger_eval = create_evaluator(net, device=device, action_size=action_size)
            
            def challenger_player(state):
                vc = search(state, challenger_eval, n_sims=n_sims // 2,
                          action_size=action_size, c_puct=c_puct, add_noise=False)
                return int(np.argmax(vc))
            
            def incumbent_player(state):
                vc = search(state, best_eval, n_sims=n_sims // 2,
                          action_size=action_size, c_puct=c_puct, add_noise=False)
                return int(np.argmax(vc))
            
            promoted, arena_stats = gate_model(
                challenger=challenger_player,
                incumbent=incumbent_player,
                create_state=create_state,
                n_games=arena_games,
                win_threshold=win_threshold,
                action_size=action_size,
            )
            
            print(f"  Win rate: {arena_stats['p1_win_rate']:.1%} "
                  f"({'PROMOTED' if promoted else 'rejected'})")
            
            if promoted:
                trainer.save_checkpoint(best_net_path, iteration=iteration)
        
        # Log metrics
        iter_elapsed = time.time() - iter_start
        metrics = {
            'iteration': iteration,
            'avg_plies': avg_plies,
            'cap_rate': cap_rate,
            'terminal_stats': terminal_stats,
            'buffer_size': len(buffer),
            'elapsed_seconds': iter_elapsed,
        }
        if train_stats:
            metrics.update(train_stats)
        metrics_log.append(metrics)
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_log, f, indent=2)
        
        print(f"  Iteration time: {iter_elapsed:.1f}s")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ChaturangaZero Training')
    parser.add_argument('--config', type=str, default='configs/local.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--output', type=str, default='runs/default',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    config = load_config(args.config)
    training_loop(config, output_dir=args.output, resume_from=args.resume)
