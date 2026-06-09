"""
BCI Pipeline Main Entry
========================
BCI Signal Processing Pipeline - Engineering and Integration

Usage:
    python -m bci.main --help
    python -m bci.main data.edf --config config.yaml
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='BCI Signal Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m bci.main data.edf
  python -m bci.main data.edf --config config.yaml
  python -m bci.main --gui
        """
    )

    parser.add_argument('filepath', type=str, nargs='?', help='Path to EEG file')
    parser.add_argument('--config', '-c', type=str, help='YAML config file')
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    return parser.parse_args()


def run_cli(args):
    """Run pipeline in CLI mode"""
    if not args.filepath:
        print("error: filepath is required in CLI mode (use --gui for interactive file picker)", file=sys.stderr)
        return 1
    from bci.config import PipelineConfig
    from bci.pipeline import BCIPipeline

    logger = logging.getLogger(__name__)

    if args.config:
        config = PipelineConfig.from_yaml(Path(args.config))
        logger.info(f"Loaded config from {args.config}")
    else:
        config = PipelineConfig()

    logger.info(f"Processing: {args.filepath}")

    pipeline = BCIPipeline(config)
    result = pipeline.run(args.filepath)

    if result.success:
        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        if result.accuracy is not None:
            logger.info(f"Accuracy: {result.accuracy:.3f} ± {result.std:.3f}")
        logger.info(f"Steps: {result.steps_completed}")
        logger.info("=" * 50)

        saved = pipeline.save_results()
        logger.info(f"Results saved to: {saved}")

        return 0
    else:
        logger.error(f"Pipeline failed: {result.errors}")
        return 1


def run_gui(args):
    """Run pipeline in GUI mode"""
    import os
    if not os.environ.get('DISPLAY'):
        print("error: No X display detected. GUI requires an X server.", file=sys.stderr)
        print("  Set DISPLAY and ensure X11 forwarding: ssh -Y user@host", file=sys.stderr)
        sys.exit(1)
    from bci.gui import main as gui_main
    gui_main()


def main():
    """Main entry point — defaults to GUI when no arguments given"""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.gui or not args.filepath:
        run_gui(args)
    else:
        sys.exit(run_cli(args))


if __name__ == '__main__':
    main()