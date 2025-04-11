#!/usr/bin/env python
"""KubeMindNexus UI Server - Kubernetes clusters management with Model Context Protocol.

This is the UI server for the KubeMindNexus application. It serves the Streamlit
UI and connects to the API server.
"""

import argparse
import logging
import os
import sys

from kubemindnexus.constants import Colors, DEFAULT_API_HOST, DEFAULT_API_PORT, DEFAULT_UI_PORT
from kubemindnexus.ui.app import run_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("kubemindnexus.ui")


def check_api_server(api_host: str, api_port: int) -> bool:
    """Check if the API server is running.
    
    Args:
        api_host: API server host.
        api_port: API server port.
        
    Returns:
        True if the API server is running, False otherwise.
    """
    try:
        import requests
        api_url = f"http://{api_host}:{api_port}/api"
        response = requests.get(api_url, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def print_banner() -> None:
    """Print application banner."""
    banner = f"""
{Colors.OKBLUE}
 ██ ▄█▀ █    ██  ▄▄▄▄   ▓█████  ███▄ ▄███▓ ██▓ ███▄    █ ▓█████▄  ███▄    █ ▓█████ ▒██   ██▒█    ██   ██████ 
 ██▄█▒  ██  ▓██▒▓█████▄ ▓█   ▀ ▓██▒▀█▀ ██▒▓██▒ ██ ▀█   █ ▒██▀ ██▌ ██ ▀█   █ ▓█   ▀ ▒▒ █ █ ▒░██  ▓██▒▒██    ▒ 
▓███▄░ ▓██  ▒██░▒██▒ ▄██▒███   ▓██    ▓██░▒██▒▓██  ▀█ ██▒░██   █▌▓██  ▀█ ██▒▒███   ░░  █   ░▓██  ▒██░░ ▓██▄   
▓██ █▄ ▓▓█  ░██░▒██░█▀  ▒▓█  ▄ ▒██    ▒██ ░██░▓██▒  ▐▌██▒░▓█▄   ▌▓██▒  ▐▌██▒▒▓█  ▄  ░ █ █ ▒ ▓▓█  ░██░  ▒   ██▒
▒██▒ █▄▒▒█████▓ ░▓█  ▀█▓░▒████▒▒██▒   ░██▒░██░▒██░   ▓██░░▒████▓ ▒██░   ▓██░░▒████▒▒██▒ ▒██▒▒▒█████▓ ▒██████▒▒
▒ ▒▒ ▓▒░▒▓▒ ▒ ▒ ░▒▓███▀▒░░ ▒░ ░░ ▒░   ░  ░░▓  ░ ▒░   ▒ ▒  ▒▒▓  ▒ ░ ▒░   ▒ ▒ ░░ ▒░ ░▒▒ ░ ░▓ ░░▒▓▒ ▒ ▒ ▒ ▒▓▒ ▒ ░
░ ░▒ ▒░░░▒░ ░ ░ ▒░▒   ░  ░ ░  ░░  ░      ░ ▒ ░░ ░░   ░ ▒░ ░ ▒  ▒ ░ ░░   ░ ▒░ ░ ░  ░░░   ░▒ ░░░▒░ ░ ░ ░ ░▒  ░ ░
░ ░░ ░  ░░░ ░ ░  ░    ░    ░   ░      ░    ▒ ░   ░   ░ ░  ░ ░  ░    ░   ░ ░    ░    ░    ░   ░░░ ░ ░ ░  ░  ░  
░  ░      ░      ░         ░  ░       ░    ░           ░    ░             ░    ░  ░ ░    ░     ░           ░  
                      ░                                    ░                                                   
{Colors.ENDC}
{Colors.OKGREEN}KubeMindNexus UI Server - Kubernetes clusters management with Model Context Protocol{Colors.ENDC}
{Colors.HEADER}Version: 0.1.0{Colors.ENDC}
"""
    print(banner)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="KubeMindNexus UI Server - Kubernetes clusters management with MCP"
    )
    
    parser.add_argument(
        "--api-host",
        type=str,
        default=DEFAULT_API_HOST,
        help=f"API server host (default: {DEFAULT_API_HOST})",
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=DEFAULT_API_PORT,
        help=f"API server port (default: {DEFAULT_API_PORT})",
    )
    
    parser.add_argument(
        "--ui-port",
        type=int,
        default=DEFAULT_UI_PORT,
        help=f"UI server port (default: {DEFAULT_UI_PORT})",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    parser.add_argument(
        "--check-api",
        action="store_true",
        help="Check if API server is running before starting the UI server",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the UI server."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if API server is running if requested
    if args.check_api:
        logger.info(f"Checking if API server is running at {args.api_host}:{args.api_port}...")
        if not check_api_server(args.api_host, args.api_port):
            logger.error("API server is not running. Please start it before running the UI server.")
            sys.exit(1)
        logger.info("API server is running.")
    
    # Run the app directly as requested
    logger.info(f"Starting UI with API at http://{args.api_host}:{args.api_port}")
    api_url = f"http://{args.api_host}:{args.api_port}"
    run_app(api_url)


if __name__ == "__main__":
    main()
