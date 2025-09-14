"""
Grid search for CMVec2Vec hyperparameter optimization
Tests different combinations of loss weights and architecture dimensions
"""

from cm_vec2vec.parse_args import parse_args
from cm_vec2vec import CMVec2VecTranslator, CMVec2VecTrainer, CMVec2VecEvaluator
from cm_vec2vec.data_loader import load_nl2cm_data
import json
import os
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import argparse
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')


class GridSearch:
    """
    Grid search class for hyperparameter optimization of CMVec2Vec.
    """

    def __init__(self, base_args, search_space: Dict[str, List]):
        """
        Initialize grid search.

        Args:
            base_args: Base arguments from parse_args()
            search_space: Dictionary defining search space for each parameter
            results_dir: Directory to save results
        """
        results_dir = os.path.join(base_args.save_dir, "grid_search_results")
        os.makedirs(results_dir, exist_ok=True)
        
        self.base_args = base_args
        self.search_space = search_space
        self.results_dir = results_dir
        self.results = []

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Generate all parameter combinations
        self.param_combinations = self._generate_combinations()
        print(
            f"Generated {len(self.param_combinations)} parameter combinations")

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible combinations of parameters."""
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())

        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)

        return combinations

    def _update_args(self, params: Dict[str, Any]) -> argparse.Namespace:
        """Update base args with current parameter combination."""
        # Create a copy of base args
        updated_args = argparse.Namespace(**vars(self.base_args))

        # Update with current parameters
        for key, value in params.items():
            setattr(updated_args, key, value)

        return updated_args

    def _run_single_experiment(self, params: Dict[str, Any], experiment_id: int) -> Dict[str, Any]:
        """Run a single experiment with given parameters."""
        print(f"\n{'='*60}")
        print(f"Experiment {experiment_id + 1}/{len(self.param_combinations)}")
        print(f"Parameters: {params}")
        print(f"{'='*60}")
        
        data_path = os.path.join(params['data_path'], params['dataset'])
        nl_cm_cols = [params['nl_col'], params['cm_col']]
        
        train_loader, val_loader, test_loader = load_nl2cm_data(
            data_path=data_path,
            nl_cm_cols=nl_cm_cols,
            test_size=params['test_size'],
            random_state=params['seed'],
            num_workers=params['num_workers']
        )

        try:
            # Update args with current parameters
            args = self._update_args(params)

            # Create model
            model = CMVec2VecTranslator(
                embedding_dim=args.embedding_dim,
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                adapter_depth=args.adapter_depth,
                backbone_depth=args.backbone_depth
            )

            # Create save directory for this experiment
            exp_save_dir = os.path.join(
                self.results_dir, f"experiment_{experiment_id:03d}")
            os.makedirs(exp_save_dir, exist_ok=True)
            with open(os.path.join(exp_save_dir, 'parameters.json'), 'w') as f:
                json.dump(params, f, indent=2, default=str)

            # Create evaluator and trainer
            evaluator = CMVec2VecEvaluator(model=model, save_dir=exp_save_dir)
            trainer = CMVec2VecTrainer(
                model=model,
                lr_generator=args.lr_generator,
                lr_discriminator=args.lr_discriminator,
                loss_weights={
                    'reconstruction': args.reconstruction_weight,
                    'cycle_consistency': args.cycle_consistency_weight,
                    'vsp': args.vsp_weight,
                    'adversarial': args.adversarial_weight,
                    'latent_adversarial': args.latent_adversarial_weight
                },
                save_dir=exp_save_dir,
                evaluator=evaluator
            )

            # Train model
            print(f"Training model...")
            train_fn = trainer.enhanced_train if args.enhanced_losses else trainer.train
            training_history = train_fn(
                train_loader, val_loader, epochs=args.epochs,
                save_every=args.save_every,
                early_stopping_patience=args.early_stopping_patience
            )

            # Evaluate model
            print(f"Evaluating model...")
            results = evaluator.evaluate_loader(
                test_loader,
                plot=args.save_plots,
                save_table=args.save_table
            )

            # Save results for this experiment
            experiment_result = {
                'experiment_id': experiment_id,
                'parameters': params,
                'training_history': training_history,
                'evaluation_results': results,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'timestamp': datetime.now().isoformat()
            }

            # Save individual experiment results
            with open(os.path.join(exp_save_dir, 'experiment_results.json'), 'w') as f:
                json.dump(experiment_result, f, indent=2, default=str)

            print(f"✓ Experiment {experiment_id + 1} completed successfully")
            return experiment_result

        except Exception as e:
            print(f"❌ Experiment {experiment_id + 1} failed: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'parameters': params,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run_search(self, max_experiments: int = None) -> List[Dict[str, Any]]:
        """
        Run the grid search.

        Args:
            max_experiments: Maximum number of experiments to run (None for all)

        Returns:
            List of experiment results
        """
        if max_experiments is None:
            max_experiments = len(self.param_combinations)

        print(
            f"Starting grid search with {min(max_experiments, len(self.param_combinations))} experiments")

        for i, params in tqdm(enumerate(self.param_combinations[:max_experiments]), total=max_experiments, desc="Running experiments"):
            result = self._run_single_experiment(params, i)
            self.results.append(result)

            # Save intermediate results
            self._save_results()

        # Generate summary and analysis
        self._generate_summary()
        self._analyze_results()
        self._save_parameter_analysis()

        return self.results

    def _save_results(self):
        """Save current results to file."""
        results_file = os.path.join(
            self.results_dir, 'grid_search_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

    def _generate_summary(self):
        """Generate and save summary of results."""
        successful_results = [r for r in self.results if 'error' not in r]
        failed_results = [r for r in self.results if 'error' in r]

        print(f"\n{'='*60}")
        print("GRID SEARCH SUMMARY")
        print(f"{'='*60}")
        print(f"Total experiments: {len(self.results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")

        if successful_results:
            # Find best performing experiments
            self._find_best_experiments(successful_results)

        # Save summary
        summary = {
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(failed_results),
            'search_space': self.search_space,
            'timestamp': datetime.now().isoformat()
        }

        if successful_results:
            summary['best_experiments'] = self._find_best_experiments(
                successful_results)

        with open(os.path.join(self.results_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def _find_best_experiments(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find best performing experiments based on key metrics."""
        best_experiments = {}

        # Key metrics to optimize
        metrics = [
            'mean_cycle_similarity',
            'mean_geometry_correlation',
            'nlt2cmt_top_1_accuracy_results',
            'cmt2nlt_top_1_accuracy_results',
            'nlt2cmt_mrr_results',
            'cmt2nlt_mrr_results'
        ]

        for metric in metrics:
            best_exp = None
            best_value = float('-inf')

            for result in results:
                if 'evaluation_results' in result and metric in result['evaluation_results']:
                    value = result['evaluation_results'][metric]
                    if isinstance(value, (int, float)) and value > best_value:
                        best_value = value
                        best_exp = result

            if best_exp:
                best_experiments[metric] = {
                    'experiment_id': best_exp['experiment_id'],
                    'parameters': best_exp['parameters'],
                    'value': best_value
                }

        return best_experiments

    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from results for easier analysis."""
        data = []

        for result in self.results:
            if 'error' not in result:
                row = result['parameters'].copy()
                row['experiment_id'] = result['experiment_id']
                row['model_parameters'] = result.get('model_parameters', 0)

                # Add evaluation metrics
                if 'evaluation_results' in result:
                    for key, value in result['evaluation_results'].items():
                        if isinstance(value, (int, float)):
                            row[f'eval_{key}'] = value

                data.append(row)

        return pd.DataFrame(data)

    def _analyze_results(self):
        """Perform detailed analysis of results."""
        df = self._create_results_dataframe()

        if df.empty:
            print("No successful experiments to analyze.")
            return

        print(f"\n{'='*60}")
        print("DETAILED RESULTS ANALYSIS")
        print(f"{'='*60}")

        # Basic statistics
        print(f"Total successful experiments: {len(df)}")
        print(f"Parameter combinations tested: {len(df)}")

        # Top performing experiments
        key_metrics = [
            'nlt2cmt_top_1_accuracy_results',
            'nlt2cmt_top_5_accuracy_results',
            'nlt2cmt_cosine_similarity_results',
            'nlt2cmt_mrr_results',
        ]

        for metric in key_metrics:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_params = df.loc[best_idx, [
                    col for col in df.columns if not col.startswith('eval_')]]
                print(f"\nBest {metric}: {df.loc[best_idx, metric]:.4f}")
                print(f"Parameters: {dict(best_params)}")

        # Save detailed results
        results_file = os.path.join(self.results_dir, 'detailed_results.csv')
        df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

        # Parameter importance analysis
        self._analyze_parameter_importance(df)

    def _analyze_parameter_importance(self, df: pd.DataFrame):
        """Analyze which parameters have the most impact on performance."""
        print(f"\n{'='*60}")
        print("PARAMETER IMPORTANCE ANALYSIS")
        print(f"{'='*60}")

        # Key metrics for analysis
        target_metrics = [
            'nlt2cmt_top_1_accuracy_results',
            'nlt2cmt_top_5_accuracy_results',
            'nlt2cmt_cosine_similarity_results',
            'nlt2cmt_mrr_results',
        ]

        param_columns = [col for col in df.columns if not col.startswith(
            'eval_') and col != 'experiment_id']

        for metric in target_metrics:
            if metric in df.columns:
                print(f"\nAnalysis for {metric}:")

                # Calculate correlation with each parameter
                correlations = []
                for param in param_columns:
                    if df[param].dtype in ['int64', 'float64']:
                        corr = df[param].corr(df[metric])
                        correlations.append((param, corr))

                # Sort by absolute correlation
                correlations.sort(key=lambda x: abs(x[1]), reverse=True)

                for param, corr in correlations[:5]:  # Top 5
                    print(f"  {param}: {corr:.3f}")


    def _save_parameter_analysis(self):
        """Save parameter analysis plots and summaries."""
        df = self._create_results_dataframe()

        if df.empty:
            return

        # Create parameter correlation heatmap
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Select numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True,
                        cmap='coolwarm', center=0)
            plt.title('Parameter Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir,
                        'parameter_correlations.png'))
            plt.close()

        except ImportError:
            print("Matplotlib/Seaborn not available for plotting")


def create_search_space() -> Dict[str, List]:
    """
    Define the search space for hyperparameters.

    Returns:
        Dictionary defining search space for each parameter
    """
    return {
        # Architecture parameters
        'latent_dim': [128, 512],
        'hidden_dim': [256, 512],
        'adapter_depth': [2, 3],
        'backbone_depth': [3, 4],

        # Loss weights
        'reconstruction_weight': [10.0, 15.0, 20.0],
        'cycle_consistency_weight': [10.0, 15.0, 20.0],
        'vsp_weight': [2.0, 3.0, 10.0],
        'adversarial_weight': [1.0, 2.0, 10.0],
        'latent_adversarial_weight': [1.0, 2.0, 10.0],
    }


def run_single_experiment(experiment_params: Dict[str, Any], base_args=None):
    """
    Run a single experiment with specific parameters.

    Args:
        experiment_params: Dictionary of parameters for the experiment
        base_args: Base arguments (if None, will parse from command line)
    """
    if base_args is None:
        base_args = parse_args()

    # Create a single-experiment grid search
    search_space = {key: [value] for key, value in experiment_params.items()}
    grid_search = GridSearch(base_args, search_space,
                             results_dir="single_experiment")

    print(f"Running single experiment with parameters: {experiment_params}")
    results = grid_search.run_search(max_experiments=1)

    return results[0] if results else None


def main():
    """Main function to run grid search."""
    import argparse

    parser = argparse.ArgumentParser(description='CMVec2Vec Grid Search')

    parser.add_argument('--max_experiments', type=int, default=None,
                        help='Maximum number of experiments to run')
    args = parser.parse_args()

    # Parse base arguments
    base_args = parse_args()

    # Create search space based on type
    search_space = create_search_space()
    
    # Create grid search
    grid_search = GridSearch(base_args, search_space)

    # Run grid search
    print("Starting CMVec2Vec Grid Search")
    print(f"Search space: {search_space}")
    print(f"Total combinations: {len(grid_search.param_combinations)}")

    # Ask user for confirmation
    max_exp = args.max_experiments
    response = input(f"\nRun {max_exp} experiments? (y/n): ")
    if response.lower() != 'y':
        print("Grid search cancelled.")
        return

    # Run the search
    results = grid_search.run_search(max_experiments=args.max_experiments)

    print(
        f"\nGrid search completed! Results saved to: {grid_search.results_dir}")

    # Print quick summary
    successful = [r for r in results if 'error' not in r]
    print(f"Successful experiments: {len(successful)}/{len(results)}")

    if successful:
        print("\nTop 3 experiments by top1 accuracy:")
        key_metric = 'nlt2cmt_top_1_accuracy_results'
        results = [(r, r['evaluation_results'].get(key_metric, -999))
                             for r in successful if 'evaluation_results' in r]
        results.sort(key=lambda x: x[1], reverse=True)

        for i, (result, score) in enumerate(results[:3]):
            print(f"{i+1}. Score: {score:.4f}, Params: {result['parameters']}")


if __name__ == "__main__":
    main()
