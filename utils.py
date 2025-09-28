import logging
import time
import functools
import hashlib
import json
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
from rich.console import Console

LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)

def get_console() -> Console:
    """Get a rich console instance."""
    return Console()

def retry_with_backoff(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger = get_logger(func.__module__)
                        logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    
            raise last_exception
        return wrapper
    return decorator

def measure_time(func: Callable) -> Callable:
    """Decorator to measure and log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = get_logger(func.__module__)
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.2f}s")
        return result
    return wrapper

def safe_json_load(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Safely load JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        get_logger(__name__).warning(f"JSON file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        get_logger(__name__).error(f"Invalid JSON in {file_path}: {e}")
        return {}

def safe_json_save(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """Safely save data to JSON file."""
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        get_logger(__name__).error(f"Failed to save JSON to {file_path}: {e}")
        return False

def validate_file_extension(file_path: str, allowed_extensions: list) -> bool:
    """Validate if file extension is allowed."""
    if not file_path:
        return False
    
    extension = Path(file_path).suffix.lower()
    return extension in [ext.lower() for ext in allowed_extensions]

def get_file_hash(content: Union[str, bytes]) -> str:
    """Get MD5 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.md5(content).hexdigest()

class SimpleCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, default_ttl: float = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        if time.time() > entry['expires']:
            del self.cache[key]
            return None
            
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache with TTL."""
        if ttl is None:
            ttl = self.default_ttl
            
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items() 
            if current_time > entry['expires']
        ]
        for key in expired_keys:
            del self.cache[key]

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and size_index < len(size_names) - 1:
        size /= 1024
        size_index += 1
    
    return f"{size:.1f} {size_names[size_index]}"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe usage."""
    # Remove or replace dangerous characters
    import re
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized or "untitled"
