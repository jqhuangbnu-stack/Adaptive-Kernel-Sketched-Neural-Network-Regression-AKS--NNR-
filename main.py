"""
Main script for Kernel Sketching Dimension Estimation
Runs experiments for Model A, B, C with Scenario I
All computations on GPU using PyTorch
"""

import torch
import torch.nn as nn
import time
import os
import json
import csv
import numpy as np
import traceback
import sys
from datetime import datetime
from collections import Counter

# Import your modules
from data_generator_all_settings import DataGenerator
from Step1_initialization import KernelSketcher
from Step2_model_training import KernelModelTrainer
from Step3_dimension_estimation import DimensionEstimator


# 强制刷新输出
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


class ExperimentConfig:
    """Configuration for a single experiment"""
    def __init__(self, model='A', scenario='I', p=50, n=1000, m=100, 
                 d_max=10, lambda_c=0.1, h=1.0, 
                 lr=1e-3, L=3, hidden_dim=64, patience=20, val_split=0.2,
                 lambda_n=None, stop_threshold=0.1, max_epochs=2000, seed=42,
                 save_ghat=False):
        self.model = model
        self.scenario = scenario
        self.p = p
        self.n = n
        self.m = m
        self.d_max = d_max
        self.lambda_c = lambda_c
        self.h = h
        self.lr = lr
        self.L = L
        self.hidden_dim = hidden_dim
        self.patience = patience
        self.val_split = val_split
        self.stop_threshold = stop_threshold
        self.max_epochs = max_epochs
        self.lambda_n = lambda_n
        self.seed = seed
        self.save_ghat = save_ghat


def numpy_to_torch(X_np, Y_np, device='cuda'):
    """Convert numpy arrays to torch tensors on GPU"""
    X_torch = torch.from_numpy(X_np).float().to(device)
    Y_torch = torch.from_numpy(Y_np).float().to(device)
    return X_torch, Y_torch


def response_dim(Y):
    """Return response dimension for scalar or vector-valued responses."""
    return 1 if Y.dim() == 1 else Y.shape[1]


def median_heuristic_bandwidth(X, max_pairs=200000, seed=123):
    """
    Compute the RBF bandwidth by the median heuristic:
        h = sqrt(median(||X_i - X_j||^2) / 2), i < j.

    For large n, use a reproducible random subset of pairs to avoid building
    a full pairwise distance matrix.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)

    X = X.detach()
    n = X.shape[0]
    if n < 2:
        raise ValueError("Median heuristic requires at least two observations.")

    total_pairs = n * (n - 1) // 2

    with torch.no_grad():
        if total_pairs <= max_pairs:
            sq_dists = []
            for i in range(n - 1):
                diff = X[i + 1:] - X[i]
                sq_dists.append(torch.sum(diff * diff, dim=1))
            sq_dists = torch.cat(sq_dists)
        else:
            generator = torch.Generator(device=X.device)
            generator.manual_seed(int(seed))

            i = torch.randint(0, n, (max_pairs,), device=X.device, generator=generator)
            j = torch.randint(0, n, (max_pairs,), device=X.device, generator=generator)
            mask = i != j
            i = i[mask]
            j = j[mask]

            if i.numel() == 0:
                raise ValueError("No non-identical pairs sampled for median heuristic.")

            diff = X[i] - X[j]
            sq_dists = torch.sum(diff * diff, dim=1)

        median_sq_dist = torch.median(sq_dists).item()

    if not np.isfinite(median_sq_dist) or median_sq_dist <= 0:
        raise ValueError(f"Invalid median squared distance for bandwidth: {median_sq_dist}")

    return float(np.sqrt(median_sq_dist / 2.0))


def evaluate_on_test(model, sketcher, X_test, Y_test, device='cuda', return_ghat=False):
    """Evaluate trained model on test set"""
    model.eval()
    with torch.no_grad():
        n_test = X_test.shape[0]
        all_predictions = []
        all_ghat = []
        
        for i in range(n_test):
            x = X_test[i]
            x_norm = torch.sum(x**2)
            x_dot_X = x @ sketcher.X_train.T
            sq_dists = x_norm + sketcher.X_train_norm - 2 * x_dot_X
            sq_dists = sq_dists.clamp(min=0)
            k_vals = torch.exp(-sq_dists / (2 * sketcher.h**2))
            k_n = k_vals / sketcher.sqrt_n
            sketched = sketcher.S @ k_n
            sketched = sketched.unsqueeze(1)
            pred = model(sketched)
            all_predictions.append(pred)
            if return_ghat:
                all_ghat.append(model.transform(sketched))
        
        predictions = torch.cat(all_predictions)
        ghat = torch.cat(all_ghat) if return_ghat else None
        if Y_test.dim() == 1:
            mse = torch.mean((predictions.squeeze() - Y_test) ** 2).item()
        else:
            mse = torch.mean((predictions - Y_test) ** 2).item()
    
    return mse, predictions, ghat


def run_single_experiment(config, verbose=True, save_detailed=True, output_dir=None):
    """
    Run complete experiment for one configuration
    """
    try:
        if verbose:
            print(f"\n{'='*70}", flush=True)
            print(f"Starting Experiment: Model {config.model}, Scenario {config.scenario}", flush=True)
            print(f"p={config.p}, n={config.n}, m={config.m}, d_max={config.d_max}", flush=True)
            print(f"seed={config.seed}", flush=True)
            print(f"{'='*70}", flush=True)
        
        start_time = time.time()
        
        # Set seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if verbose:
            print(f"Device: {device}", flush=True)
        
        # Step 0: Generate Data
        if verbose:
            print("\n[Step 0] Generating data...", flush=True)
        
        data_gen = DataGenerator(seed=config.seed)
        
        train_data = data_gen.generate_dataset(
            p=config.p, n=config.n, m=config.n,
            model=config.model, scenario=config.scenario,
            random_state=config.seed
        )
        
        test_data = data_gen.generate_dataset(
            p=config.p, n=config.n, m=config.n,
            model=config.model, scenario=config.scenario,
            random_state=config.seed + 1000
        )
        
        X_train, Y_train = numpy_to_torch(train_data['X_train'], train_data['Y_train'], device)
        X_test, Y_test = numpy_to_torch(test_data['X_test'], test_data['Y_test'], device)
        
        if verbose:
            print(f"  Training: X {X_train.shape}, Y {Y_train.shape}", flush=True)
            print(f"  Test: X {X_test.shape}, Y {Y_test.shape}", flush=True)
        
        # Use the median heuristic when h <= 0.
        if config.h <= 0:
            estimated_h = median_heuristic_bandwidth(
                X_train, max_pairs=200000, seed=config.seed
            )
            if verbose:
                print(
                    f"  Using median heuristic bandwidth: h={estimated_h:.6f}",
                    flush=True,
                )
            config.h = estimated_h

        # Step 1: Initialize Kernel Sketcher
        if verbose:
            print("\n[Step 1] Initializing Kernel Sketcher...", flush=True)
        
        sketcher = KernelSketcher(
            m=config.m, n=config.n, p=config.p,
            h=config.h, device=device, seed=config.seed
        )
        sketcher.X_train = X_train
        sketcher.X_train_norm = torch.sum(X_train**2, dim=1)
        
        # Step 2: Iterative Model Training
        if verbose:
            print("\n[Step 2] Iterative model training...", flush=True)
            print(f"  lr={config.lr}, L={config.L}, hidden_dim={config.hidden_dim}, patience={config.patience}", flush=True)
        
        trainer = KernelModelTrainer(
            sketcher=sketcher, lambda_c=config.lambda_c,
            lr=config.lr, L=config.L, hidden_dim=config.hidden_dim,
            patience=config.patience, val_split=config.val_split,
            stop_threshold=config.stop_threshold, max_epochs=config.max_epochs
        )
        
        trained_models, final_d = trainer.run_iterative_training(
            X_train=X_train, Y_train=Y_train, d_max=config.d_max
        )
        
        # Step 3: Dimension Estimation
        if verbose:
            print("\n[Step 3] Estimating optimal dimension...", flush=True)
        
        estimator = DimensionEstimator(
            history_losses=trainer.history_losses,
            n=config.n, d_max=final_d, lambda_n=config.lambda_n
        )
        d_hat = estimator.estimate_dimension()
        if trained_models:
            d_hat = max(1, min(int(d_hat), max(trained_models.keys())))
        
        # Step 4: Evaluate on test set
        if verbose:
            print("\n[Step 4] Evaluating on test set...", flush=True)
        
        best_model = trained_models.get(d_hat, None)
        ghat_file = None
        ghat_shape = None
        if best_model is not None:
            test_mse, predictions, ghat_test = evaluate_on_test(
                best_model,
                sketcher,
                X_test,
                Y_test,
                device,
                return_ghat=config.save_ghat,
            )
            if config.save_ghat and output_dir and ghat_test is not None:
                ghat_file = os.path.join(output_dir, f'ghat_seed{config.seed}.npz')
                np.savez_compressed(
                    ghat_file,
                    Ghat_test=ghat_test.detach().cpu().numpy(),
                    Yhat_test=predictions.detach().cpu().numpy(),
                    Y_test=Y_test.detach().cpu().numpy(),
                    X_test=test_data['X_test'],
                    d_estimated=np.array([d_hat]),
                    true_dim=np.array([train_data.get('true_dim', DataGenerator.true_dimension(config.model))]),
                )
                ghat_shape = list(ghat_test.shape)
        else:
            test_mse = None
        
        initial_loss = sketcher.compute_initial_loss(Y_train).item()
        
        if test_mse is not None:
            relative_improvement = (initial_loss - test_mse) / initial_loss * 100
        else:
            relative_improvement = None
        
        elapsed_time = time.time() - start_time
        
        # Collect results
        results = {
            'model': config.model,
            'scenario': config.scenario,
            'setting_family': train_data.get('setting_family', 'unknown'),
            'true_dim': train_data.get('true_dim', DataGenerator.true_dimension(config.model)),
            'response_dim': response_dim(Y_train),
            'seed': config.seed,
            'p': config.p,
            'n': config.n,
            'm': config.m,
            'h': config.h,
            'd_max': config.d_max,
            'lambda_c': config.lambda_c,
            'lambda_n': config.lambda_n,
            'lr': config.lr,
            'L': config.L,
            'hidden_dim': config.hidden_dim,
            'patience': config.patience,
            'val_split': config.val_split,
            'stop_threshold': config.stop_threshold,
            'max_epochs': config.max_epochs,
            'final_d': final_d,
            'd_estimated': d_hat,
            'dimension_fallback_used': getattr(estimator, 'dimension_fallback_used', False),
            'r_hats': getattr(estimator, 'r_hats', []),
            'initial_loss': initial_loss,
            'test_mse': test_mse if test_mse is not None else -1,
            'relative_improvement': relative_improvement if relative_improvement is not None else -1,
            'save_ghat': config.save_ghat,
            'ghat_file': ghat_file if ghat_file is not None else '',
            'ghat_shape': ghat_shape if ghat_shape is not None else [],
            'history_losses': trainer.history_losses,
            'runtime_seconds': elapsed_time,
            'device': device,
            'status': 'success'
        }
        
        # Save results
        if output_dir:
            save_results_to_files(results, config, output_dir)
        
        if verbose:
            print(f"\n{'='*70}", flush=True)
            print(f"Experiment Complete!", flush=True)
            print(f"  Final training dimension: {final_d}", flush=True)
            print(f"  Estimated optimal dimension: {d_hat}", flush=True)
            print(f"  Initial loss: {initial_loss:.6f}", flush=True)
            if test_mse is not None:
                print(f"  Test MSE: {test_mse:.6f}", flush=True)
                print(f"  Improvement: {relative_improvement:.2f}%", flush=True)
            print(f"  Runtime: {elapsed_time:.2f}s", flush=True)
            print(f"{'='*70}", flush=True)
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        
        # Save error information
        if output_dir:
            error_file = os.path.join(output_dir, f'error_seed{config.seed}.txt')
            try:
                os.makedirs(output_dir, exist_ok=True)
                with open(error_file, 'w') as f:
                    f.write(f"Error: {e}\n")
                    f.write(traceback.format_exc())
                print(f"Error saved to {error_file}", flush=True)
            except Exception as e2:
                print(f"Failed to save error: {e2}", flush=True)
        
        # Return error results
        return {
            'model': config.model,
            'scenario': config.scenario,
            'seed': config.seed,
            'status': 'failed',
            'error': str(e)
        }


def save_results_to_files(results, config, output_dir):
    """Save results to both JSON and CSV files"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save as JSON (detailed)
        json_file = os.path.join(output_dir, f'results_seed{config.seed}.json')
        
        # Convert history_losses to list if it's a tensor
        results_copy = results.copy()
        if hasattr(results_copy['history_losses'], 'tolist'):
            results_copy['history_losses'] = results_copy['history_losses'].tolist()
        
        with open(json_file, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        print(f"[SAVE] JSON saved: {json_file}", flush=True)
        
        # 2. Save as CSV (summary)
        csv_file = os.path.join(output_dir, 'results_summary.csv')
        
        # Prepare row for CSV (flat structure)
        csv_row = {
            'model': config.model,
            'scenario': config.scenario,
            'seed': config.seed,
            'setting_family': results.get('setting_family', 'unknown'),
            'true_dim': results.get('true_dim', -1),
            'response_dim': results.get('response_dim', -1),
            'p': config.p,
            'n': config.n,
            'm': config.m,
            'h': config.h,
            'd_max': config.d_max,
            'lambda_c': config.lambda_c,
            'lambda_n': config.lambda_n,
            'lr': config.lr,
            'L': config.L,
            'hidden_dim': config.hidden_dim,
            'patience': config.patience,
            'val_split': config.val_split,
            'stop_threshold': config.stop_threshold,
            'max_epochs': config.max_epochs,
            'final_d': results.get('final_d', -1),
            'd_estimated': results.get('d_estimated', -1),
            'dimension_fallback_used': results.get('dimension_fallback_used', False),
            'initial_loss': results.get('initial_loss', -1),
            'test_mse': results.get('test_mse', -1),
            'relative_improvement': results.get('relative_improvement', -1),
            'save_ghat': results.get('save_ghat', False),
            'ghat_file': results.get('ghat_file', ''),
            'ghat_shape': results.get('ghat_shape', []),
            'runtime_seconds': results.get('runtime_seconds', -1),
            'device': results.get('device', 'unknown'),
            'status': results.get('status', 'unknown')
        }
        
        # Check if CSV file exists to write header
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)
        
        print(f"[SAVE] CSV appended: {csv_file}", flush=True)
        
        # 3. Save individual CSV per seed (optional, for easier debugging)
        individual_csv = os.path.join(output_dir, f'results_seed{config.seed}.csv')
        with open(individual_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())
            writer.writeheader()
            writer.writerow(csv_row)
        
        print(f"[SAVE] Individual CSV saved: {individual_csv}", flush=True)
        
    except Exception as e:
        print(f"[SAVE ERROR] Failed to save results: {e}", flush=True)
        print(traceback.format_exc(), flush=True)


def run_multiple_seeds(config, seeds=range(42, 142), verbose=True, save_detailed=True, output_dir=None):
    """Run the same configuration across multiple random seeds"""
    print(f"\n{'='*80}", flush=True)
    print(f"Running {len(seeds)} experiments for Model {config.model}, Scenario {config.scenario}", flush=True)
    print(f"{'='*80}", flush=True)
    
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Experiment {i+1}/{len(seeds)} (seed={seed}) ---", flush=True)
        
        config.seed = seed
        result = run_single_experiment(config, verbose=False, save_detailed=save_detailed, output_dir=output_dir)
        all_results.append(result)
        
        if result.get('status') == 'success':
            print(f"  d_hat={result['d_estimated']}, test_mse={result.get('test_mse', -1):.6f}, "
                  f"runtime={result['runtime_seconds']:.1f}s", flush=True)
        else:
            print(f"  FAILED: {result.get('error', 'Unknown error')}", flush=True)
    
    # Aggregate results for successful runs
    successful = [r for r in all_results if r.get('status') == 'success']
    
    if successful:
        d_estimates = [r['d_estimated'] for r in successful]
        test_mses = [r['test_mse'] for r in successful]
        
        stats = {
            'd_mean': np.mean(d_estimates),
            'd_std': np.std(d_estimates),
            'd_counts': dict(Counter(d_estimates)),
            'test_mse_mean': np.mean(test_mses),
            'test_mse_std': np.std(test_mses),
            'n_successful': len(successful),
            'n_total': len(seeds)
        }
        
        print(f"\n{'='*80}", flush=True)
        print(f"Aggregated Results for Model {config.model}, Scenario {config.scenario}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Successful: {len(successful)}/{len(seeds)}", flush=True)
        print(f"Mean estimated dimension: {stats['d_mean']:.2f} ± {stats['d_std']:.2f}", flush=True)
        print(f"Test MSE: {stats['test_mse_mean']:.6f} ± {stats['test_mse_std']:.6f}", flush=True)
        print("\nDistribution of d*:", flush=True)
        for d in sorted(stats['d_counts'].keys()):
            percentage = 100 * stats['d_counts'][d] / len(successful)
            print(f"  d={d}: {stats['d_counts'][d]} ({percentage:.1f}%)", flush=True)
        
        return {'stats': stats, 'all_results': all_results}
    else:
        print(f"\n❌ All {len(seeds)} experiments failed!", flush=True)
        return {'stats': None, 'all_results': all_results}


def main():
    """Main execution function"""
    import argparse
    parser = argparse.ArgumentParser(description='Kernel Sketching Dimension Estimation')
    
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--p', type=int, default=50, help='Predictor dimension')
    parser.add_argument('--n', type=int, default=1000, help='Sample size')
    parser.add_argument('--m', type=int, default=100, help='Sketch dimension')
    parser.add_argument('--scenario', type=str, default='I',
                       help='Scenario: I/II/III for deep SDR, GA/GB/GC for GSIR scalar settings')
    parser.add_argument('--d_max', type=int, default=10, help='Maximum dimension')
    parser.add_argument('--lambda_c', type=float, default=0.1, help='Orthogonality penalty')
    parser.add_argument('--h', type=float, default=1.0, help='Kernel bandwidth')
    parser.add_argument('--lambda_n', type=float, default=None, help='Custom lambda_n')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--L', type=int, default=3, help='Number of layers in h_d')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--max_epochs', type=int, default=2000, help='Maximum epochs for each fixed-d model')
    parser.add_argument('--seed_start', type=int, default=42, help='Starting seed')
    parser.add_argument('--n_seeds', type=int, default=1, help='Number of seeds')
    parser.add_argument('--stop_threshold', type=float, default=0.1, 
                       help='Stopping threshold for Delta_d. Larger -> smaller d_max')
    parser.add_argument('--model', type=str, default='A',
                       help='Model: A/B/C, D/E/F, or G1/G2/G3')
    parser.add_argument('--save_ghat', action='store_true',
                       help='Save Ghat_test, Yhat_test, Y_test, and X_test to ghat_seed*.npz')
    
    args = parser.parse_args()
    
    print(f"\n{'='*100}", flush=True)
    print(f"KERNEL SKETCHING DIMENSION ESTIMATION", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"Configuration:", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Data: p={args.p}, n={args.n}, m={args.m}, scenario={args.scenario}", flush=True)
    print(f"  Algorithm: d_max={args.d_max}, lambda_c={args.lambda_c}, h={args.h}, lambda_n={args.lambda_n}", flush=True)
    print(f"  Training: lr={args.lr}, L={args.L}, hidden_dim={args.hidden_dim}, patience={args.patience}", flush=True)
    print(f"  Seeds: {args.seed_start}-{args.seed_start+args.n_seeds-1}", flush=True)
    print(f"  Output dir: {args.output_dir}", flush=True)
    print(f"{'='*100}", flush=True)
    
    # Create config
    config = ExperimentConfig(
        model=args.model,
        scenario=args.scenario,
        p=args.p,
        n=args.n,
        m=args.m,
        h=args.h,
        d_max=args.d_max,
        lambda_c=args.lambda_c,
        lambda_n=args.lambda_n,
        lr=args.lr,
        L=args.L,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
        val_split=args.val_split,
        stop_threshold=args.stop_threshold,
        max_epochs=args.max_epochs,
        save_ghat=args.save_ghat
    )
    
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    
    # Run experiments
    results = run_multiple_seeds(config, seeds=seeds, save_detailed=True, output_dir=args.output_dir)
    
    print(f"\n{'='*100}", flush=True)
    print(f"ALL EXPERIMENTS COMPLETED", flush=True)
    print(f"{'='*100}", flush=True)
    
    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
