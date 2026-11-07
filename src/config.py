"""
Database configuration management.

Supports multiple database modes:
- local: Local PostgreSQL (default)
- aws: AWS RDS
- custom: Custom configuration from environment variables
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_database_config(mode: Optional[str] = None) -> Dict[str, str]:
    """
    Get database configuration based on mode.

    Args:
        mode: Database mode - 'local', 'aws', or None (uses DB_MODE env var)

    Returns:
        Dictionary with database connection parameters

    Environment Variables:
        DB_MODE: Database mode (local/aws/custom) - default: local

        For local mode:
            POSTGRES_HOST: default localhost
            POSTGRES_PORT: default 5432
            POSTGRES_USER: default postgres
            POSTGRES_PASSWORD: default postgres
            POSTGRES_DATABASE: default postgres

        For AWS mode:
            AWS_DB_HOST: AWS RDS endpoint
            AWS_DB_PORT: default 5432
            AWS_DB_USER: default postgres
            AWS_DB_PASSWORD: AWS DB password
            AWS_DB_NAME: default postgres

        For custom mode:
            DB_HOST: Database host
            DB_PORT: Database port
            DB_USER: Database user
            DB_PASSWORD: Database password
            DB_NAME: Database name
    """
    # Determine mode
    if mode is None:
        mode = os.getenv('DB_MODE', 'local').lower()

    if mode == 'local':
        return _get_local_config()
    elif mode == 'aws':
        return _get_aws_config()
    elif mode == 'custom':
        return _get_custom_config()
    else:
        raise ValueError(f"Invalid database mode: {mode}. Must be 'local', 'aws', or 'custom'")


def _get_connect_timeout_seconds() -> int:
    try:
        return max(1, int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10")))
    except Exception:
        return 10


def _get_local_config() -> Dict[str, str]:
    """Get local PostgreSQL configuration."""
    return {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        'database': os.getenv('POSTGRES_DATABASE', 'postgres'),
        'connect_timeout': _get_connect_timeout_seconds(),
    }


def _get_aws_config() -> Dict[str, str]:
    """Get AWS RDS configuration from environment variables."""
    host = os.getenv('AWS_DB_HOST')
    if not host:
        raise ValueError(
            "AWS_DB_HOST environment variable is required for AWS mode. "
            "Please set it in your .env file."
        )

    password = os.getenv('AWS_DB_PASSWORD')
    if not password:
        raise ValueError(
            "AWS_DB_PASSWORD environment variable is required for AWS mode. "
            "Please set it in your .env file."
        )

    return {
        'host': host,
        'port': os.getenv('AWS_DB_PORT', '5432'),
        'user': os.getenv('AWS_DB_USER', 'postgres'),
        'password': password,
        'database': os.getenv('AWS_DB_NAME', 'postgres'),
        'connect_timeout': _get_connect_timeout_seconds(),
    }


def _get_custom_config() -> Dict[str, str]:
    """Get custom database configuration from environment variables."""
    host = os.getenv('DB_HOST')
    if not host:
        raise ValueError(
            "DB_HOST environment variable is required for custom mode. "
            "Please set it in your .env file."
        )

    password = os.getenv('DB_PASSWORD')
    if not password:
        raise ValueError(
            "DB_PASSWORD environment variable is required for custom mode. "
            "Please set it in your .env file."
        )

    return {
        'host': host,
        'port': os.getenv('DB_PORT', '5432'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': password,
        'database': os.getenv('DB_NAME', 'postgres'),
        'connect_timeout': _get_connect_timeout_seconds(),
    }


def get_database_mode() -> str:
    """Get current database mode from environment."""
    return os.getenv('DB_MODE', 'local').lower()


def print_database_config(hide_password: bool = True) -> None:
    """Print current database configuration (for debugging)."""
    mode = get_database_mode()
    config = get_database_config(mode)

    print(f"Database Mode: {mode}")
    print(f"Configuration:")
    for key, value in config.items():
        if key == 'password' and hide_password:
            print(f"  {key}: {'*' * len(value)}")
        else:
            print(f"  {key}: {value}")
