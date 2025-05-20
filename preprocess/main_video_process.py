"""
Module: main
Entry point for running the COLMAP reconstruction pipeline.
"""

import argparse
from colmap_pipeline import do_one, do_one_robust

def main(args):
    """
    Main function to parse arguments and run the pipeline.
    
    Parameters:
        args: Parsed command line arguments.
    """
    source_path = args.source_path
    n_images = args.number_of_frames
    clean = args.clean
    full = args.full
    
    if args.robust:
        print("Using robust reconstruction pipeline...")
        # Pass additional parameters to the robust pipeline
        do_one_robust(
            source_path,
            n_images,
            clean,
            minimal=args.minimal,
            full=full,
            random_ratio=args.random_ratio,
            pruning_threshold=args.pruning_threshold
        )
    else:
        print("Using standard reconstruction pipeline...")
        do_one(source_path, n_images, clean, minimal=args.minimal, full=full)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--number_of_frames", "-n", default=200, type=int)
    parser.add_argument("--clean", "-c", action='store_true')
    parser.add_argument("--minimal", "-m", action='store_true', help="Use minimal frame selection after final reconstruction")
    parser.add_argument("--full", "-f", action='store_true', help="Use all frame selection after final reconstruction")
    parser.add_argument("--robust", "-r", action='store_true', help="Use robust reconstruction pipeline with pose interpolation")
    
    # Additional parameters for the robust pipeline
    parser.add_argument("--random_ratio", type=float, default=0.2, help="Ratio of frames to select randomly (default: 0.2)")
    parser.add_argument("--pruning_threshold", type=float, default=0.05, help="Threshold for pruning low-contribution frames (default: 0.05)")
    parser.add_argument("--coverage_weight", type=float, default=0.4, help="Weight for coverage score in frame selection (default: 0.4)")
    parser.add_argument("--triangulation_weight", type=float, default=0.3, help="Weight for triangulation score in frame selection (default: 0.3)")
    parser.add_argument("--diversity_weight", type=float, default=0.2, help="Weight for diversity score in frame selection (default: 0.2)")
    parser.add_argument("--confidence_weight", type=float, default=0.1, help="Weight for confidence score in frame selection (default: 0.1)")
    args = parser.parse_args()
    main(args)
