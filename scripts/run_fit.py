#!/usr/bin/env python3
"""
Script to run the parametric curve fitting pipeline.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_data
from optimizer import fit_params
from plotting import plot_fit, plot_residuals
from utils import save_params_json


def setup_logging(output_dir: str) -> None:
    """Configure logging to file and console."""
    log_file = Path(output_dir) / 'rd_pipeline.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parametric curve fitting optimization"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/xy_data.csv",
        help="Path to input CSV file containing x,y data"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of uniform t samples for optimization"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results and plots"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=10,
        help="Number of random restarts for optimization"
    )
    return parser.parse_args()


def main():
    """Main entry point for the fitting pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Parametric Curve Fitting Pipeline")
    logger.info("="*60)
    logger.info(f"Data file: {args.data}")
    logger.info(f"Number of samples: {args.n_samples}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Number of restarts: {args.n_restarts}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*60)
    
    try:
        # Load data
        logger.info("Loading data...")
        df = load_data(args.data)
        logger.info(f"Loaded {len(df)} data points")
        
        # Run optimization
        logger.info("Running optimization...")
        result = fit_params(
            df,
            n_samples=args.n_samples,
            n_restarts=args.n_restarts,
            seed=args.seed
        )
        
        # Print results
        logger.info("="*60)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("="*60)
        logger.info(f"θ = {result['theta_deg']:.4f}° ({result['theta_rad']:.6f} rad)")
        logger.info(f"M = {result['M']:.6f}")
        logger.info(f"X = {result['X']:.4f}")
        logger.info(f"L1 Score = {result['l1']:.6f}")
        logger.info("="*60)
        
        # Save parameters
        params_path = Path(args.output_dir) / 'params.json'
        save_params_json(result, str(params_path))
        logger.info(f"Saved parameters to {params_path}")
        
        # Generate and save plots
        logger.info("Generating plots...")
        fit_plot_path = Path(args.output_dir) / 'fit_plot.png'
        residuals_plot_path = Path(args.output_dir) / 'residuals_plot.png'
        
        plot_fit(df, result, output_path=str(fit_plot_path))
        plot_residuals(df, result, n_samples=args.n_samples, output_path=str(residuals_plot_path))
        
        logger.info("Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
