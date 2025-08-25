import logging
import time
from typing import Optional
from urllib.parse import urlparse

from sshfs import SSHFileSystem

logger = logging.getLogger(__name__)

# Global SSH filesystem cache to reuse connections
_ssh_fs_cache = {}


def parse_ssh_path(ssh_path: str) -> tuple[str, str, str, Optional[int]]:
    """
    Parse SSH path and extract components.
    
    Args:
        ssh_path: SSH path in format ssh://[user@]host[:port]/path
        
    Returns:
        tuple: (host, user, path, port)
    """
    if not (ssh_path.startswith("ssh://") or ssh_path.startswith("sftp://")):
        raise ValueError("SSH path must start with ssh:// or sftp://")
    
    parsed = urlparse(ssh_path)
    host = parsed.hostname
    if not host:
        raise ValueError("SSH path must contain a hostname")
    
    user = parsed.username or "root"  # Default to root if no user specified
    port = parsed.port  # None if not specified
    path = parsed.path or "/"
    
    return host, user, path, port


def get_ssh_filesystem(host: str, user: str, port: Optional[int] = None, **kwargs) -> SSHFileSystem:
    """
    Get or create an SSH filesystem connection with caching.
    
    Args:
        host: SSH hostname
        user: SSH username  
        port: SSH port (optional)
        **kwargs: Additional arguments for SSHFileSystem
        
    Returns:
        SSHFileSystem instance
    """
    # Create cache key from connection parameters
    cache_key = (host, user, port, tuple(sorted(kwargs.items())))
    
    if cache_key in _ssh_fs_cache:
        logger.debug(f"Reusing cached SSH connection to {user}@{host}:{port or 22}")
        return _ssh_fs_cache[cache_key]
    
    logger.info(f"Creating new SSH connection to {user}@{host}:{port or 22}")
    
    # Create connection parameters
    connect_kwargs = {"username": user}
    if port:
        connect_kwargs["port"] = port
    connect_kwargs.update(kwargs)
    
    # Create filesystem
    fs = SSHFileSystem(host, **connect_kwargs)
    _ssh_fs_cache[cache_key] = fs
    
    return fs


def get_ssh_bytes(ssh_path: str, start_index: Optional[int] = None, end_index: Optional[int] = None, **ssh_kwargs) -> bytes:
    """
    Download file content from SSH server.
    
    Args:
        ssh_path: SSH path in format ssh://[user@]host[:port]/path
        start_index: Optional start byte offset
        end_index: Optional end byte offset  
        **ssh_kwargs: Additional SSH connection parameters
        
    Returns:
        File content as bytes
    """
    if start_index is not None or end_index is not None:
        raise NotImplementedError("Range queries not implemented for SSH yet")
    
    host, user, path, port = parse_ssh_path(ssh_path)
    
    fs = get_ssh_filesystem(host, user, port, **ssh_kwargs)
    
    with fs.open(path, "rb") as f:
        return f.read()


def get_ssh_bytes_with_backoff(ssh_path: str, max_retries: int = 8, backoff_factor: int = 2, **ssh_kwargs) -> bytes:
    """
    Download file content from SSH server with exponential backoff retry logic.
    
    Args:
        ssh_path: SSH path in format ssh://[user@]host[:port]/path
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff multiplier for retry delays
        **ssh_kwargs: Additional SSH connection parameters
        
    Returns:
        File content as bytes
        
    Raises:
        Exception: If all retry attempts fail
    """
    attempt = 0
    
    while attempt < max_retries:
        try:
            return get_ssh_bytes(ssh_path, **ssh_kwargs)
        except FileNotFoundError as e:
            logger.error(f"File not found at {ssh_path}: {e}")
            raise
        except PermissionError as e:
            logger.error(f"Permission denied accessing {ssh_path}: {e}")  
            raise
        except Exception as e:
            wait_time = backoff_factor**attempt
            logger.warning(f"Attempt {attempt+1} failed to get_ssh_bytes for {ssh_path}: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            attempt += 1
    
    logger.error(f"Failed to get_ssh_bytes for {ssh_path} after {max_retries} retries.")
    raise Exception("Failed to get_ssh_bytes after retries")


def expand_ssh_glob(ssh_path: str, **ssh_kwargs) -> list[str]:
    """
    Expand SSH glob patterns to list of matching files.
    
    Args:
        ssh_path: SSH path potentially containing glob patterns
        **ssh_kwargs: Additional SSH connection parameters
        
    Returns:
        List of matching SSH paths
    """
    host, user, path, port = parse_ssh_path(ssh_path)
    
    fs = get_ssh_filesystem(host, user, port, **ssh_kwargs)
    
    try:
        # Use glob to find matching files
        matched_files = fs.glob(path)
        
        # Convert back to full SSH URLs
        base_url = f"ssh://{user}@{host}"
        if port:
            base_url += f":{port}"
            
        return [f"{base_url}/{file_path}" for file_path in matched_files]
    except Exception as e:
        logger.error(f"Failed to expand SSH glob {ssh_path}: {e}")
        raise


def cleanup_ssh_connections():
    """Close all cached SSH connections."""
    global _ssh_fs_cache
    for fs in _ssh_fs_cache.values():
        try:
            fs.close()
        except Exception as e:
            logger.warning(f"Error closing SSH connection: {e}")
    _ssh_fs_cache.clear()
    logger.info("Cleaned up all SSH connections")