"""Main entry point for KubeMindNexus."""
import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

from .api.app import api_server
from .config.config import initialize_config
from .database.client import db_client
from .ui.app import ui
from .utils.logger import setup_logger, app_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="KubeMindNexus - K8s Cluster Management")
    
    # Mode selection
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["api", "ui", "all"],
        default="all",
        help="The mode to run in. Default is 'all'."
    )
    
    # API options
    parser.add_argument(
        "--api-host",
        type=str,
        help="Host for the API server."
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        help="Port for the API server."
    )
    
    # UI options
    parser.add_argument(
        "--ui-host",
        type=str,
        help="Host for the UI server."
    )
    
    parser.add_argument(
        "--ui-port",
        type=int,
        help="Port for the UI server."
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file."
    )
    
    return parser.parse_args()


def run_api(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Run the API server.
    
    Args:
        host: The host to bind to. If None, uses the default.
        port: The port to bind to. If None, uses the default.
    """
    app_logger.info("Starting API server")
    api_server.start(host=host, port=port)


def run_ui() -> None:
    """Run the UI server."""
    app_logger.info("Starting UI server")
    ui.run()


async def setup(config_path: Optional[str] = None) -> None:
    """Setup the application."""
    app_logger.info("Setting up KubeMindNexus")
    
    # Create configuration if it doesn't exist
    config = initialize_config(config_path)
    
    # Ensure database is initialized
    if not os.path.exists(db_client.db_path):
        app_logger.info(f"Initializing database at {db_client.db_path}")
        db_client._create_tables()


def main() -> None:
    """Main entry point."""
    # Configure logging
    logger = setup_logger()
    
    # Parse arguments
    args = parse_args()
    
    # Setup
    asyncio.run(setup(args.config))
    
    # Run in specified mode
    if args.mode == "api" or args.mode == "all":
        if args.mode == "all":
            # Start API in a separate process
            import multiprocessing
            api_process = multiprocessing.Process(
                target=run_api,
                args=(args.api_host, args.api_port)
            )
            api_process.start()
            logger.info(f"API server started in a separate process (PID: {api_process.pid})")
        else:
            # Start API only
            run_api(host=args.api_host, port=args.api_port)
    
    if args.mode == "ui" or args.mode == "all":
        # Start UI
        run_ui()


if __name__ == "__main__":
    main()
