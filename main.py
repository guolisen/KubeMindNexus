#!/usr/bin/env python
"""KubeMindNexus - Kubernetes clusters management with Model Context Protocol.

This is the main entry point for the KubeMindNexus application. It delegates
to either the API server or the UI server based on the user's choice.
"""

import argparse
import logging
import os
import sys

from kubemindnexus.constants import Colors, DEFAULT_API_HOST, DEFAULT_API_PORT, DEFAULT_UI_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("kubemindnexus")


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
{Colors.OKGREEN}KubeMindNexus - Kubernetes clusters management with Model Context Protocol{Colors.ENDC}
{Colors.HEADER}Version: 0.1.0{Colors.ENDC}
"""
    print(banner)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="KubeMindNexus - Kubernetes clusters management with MCP"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
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
    
    # Add server mode selection argument
    parser.add_argument(
        "--mode",
        type=str,
        choices=["api", "ui"],
        required=True,
        help="Select which server to run: API server or UI server",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print_banner()
    
    # Run appropriate server based on mode
    if args.mode == "api":
        # Import and run API server
        import api_server
        
        # Override args for API server
        sys.argv = ["api_server.py"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.api_host:
            sys.argv.extend(["--host", args.api_host])
        if args.api_port:
            sys.argv.extend(["--port", str(args.api_port)])
        if args.debug:
            sys.argv.append("--debug")
            
        # Run API server
        api_server.main()
    
    elif args.mode == "ui":
        # Import and run UI server
        import ui_server
        
        # Override args for UI server
        sys.argv = ["ui_server.py"]
        if args.api_host:
            sys.argv.extend(["--api-host", args.api_host])
        if args.api_port:
            sys.argv.extend(["--api-port", str(args.api_port)])
        if args.ui_port:
            sys.argv.extend(["--ui-port", str(args.ui_port)])
        if args.debug:
            sys.argv.append("--debug")
            
        # Run UI server
        ui_server.main()


if __name__ == "__main__":
    main()
