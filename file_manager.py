import shutil
import re
import fnmatch
from pathlib import Path
from typing import List, Union, Optional
from rich.console import Console
import os
import hashlib

from utils import get_logger

logger = get_logger(__name__)

class LocalFileManager:
    """Handle local file operations in a secure workspace directory."""
    
    def __init__(self, console: Console, base_dir: str = "workspace_local"):
        self.console = console
        self.base_dir = Path(base_dir).resolve() # Use resolved, absolute path
        self.base_dir.mkdir(exist_ok=True)
    
    def _get_full_path(self, path: str) -> Path:
        """
        Convert relative path to a secure, full local path.
        Prevents directory traversal attacks.
        """
        # Normalize path to prevent tricks like '..'
        rel_path = Path(os.path.normpath(path.replace("workspace/", "").lstrip("/")))
        
        # Disallow absolute paths from being passed
        if rel_path.is_absolute():
            raise ValueError("Absolute paths are not allowed.")
            
        full_path = (self.base_dir / rel_path).resolve()
        
        # Security check: Ensure the resolved path is within the base directory
        if self.base_dir not in full_path.parents and full_path != self.base_dir:
            raise PermissionError("Attempted to access a file outside the workspace.")
            
        return full_path
    
    def get_file_hash(self, path: str) -> Optional[str]:
        """Get the MD5 hash of a file."""
        try:
            full_path = self._get_full_path(path)
            if not full_path.is_file():
                return None
            
            hasher = hashlib.md5()
            with open(full_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except (FileNotFoundError, PermissionError):
            return None
    
    def read_file(self, path: str) -> bytes:
        """Read file from local workspace."""
        full_path = self._get_full_path(path)
        try:
            return full_path.read_bytes()
        except Exception as e:
            raise RuntimeError(f"Failed to read local file {path}: {e}")
    
    def write_file(self, path: str, content: Union[str, bytes]) -> None:
        """Write file to local workspace."""
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(content, str):
                full_path.write_text(content, encoding='utf-8')
            else:
                full_path.write_bytes(content)
        except Exception as e:
            raise RuntimeError(f"Failed to write local file {path}: {e}")
    
    def delete_file(self, path: str) -> None:
        """Delete file from local workspace."""
        full_path = self._get_full_path(path)
        try:
            if full_path.is_file():
                full_path.unlink()
            elif full_path.is_dir():
                shutil.rmtree(full_path)
        except Exception as e:
            raise RuntimeError(f"Failed to delete local file {path}: {e}")
    
    def move_file(self, src: str, dest: str) -> None:
        """Move file in local workspace."""
        src_path = self._get_full_path(src)
        dest_path = self._get_full_path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.move(str(src_path), str(dest_path))
        except Exception as e:
            raise RuntimeError(f"Failed to move local file {src} to {dest}: {e}")
    
    def list_files(self, path: str = "") -> List[str]:
        """List files in local workspace."""
        try:
            search_path = self._get_full_path(path)
        except ValueError: # Handle case where initial path is empty
             search_path = self.base_dir

        results = []
        try:
            if search_path.exists():
                for item in search_path.rglob("*"):
                    if item.is_file():
                        rel_path = item.relative_to(self.base_dir)
                        results.append(f"workspace/{str(rel_path).replace('\\', '/')}")
        except Exception as e:
            logger.exception("Failed to list local files")
        return results

    def search_content(self, query: str, include_pattern: Optional[str] = None, exclude_pattern: Optional[str] = None, case_sensitive: bool = False) -> List[str]:
        """Search for content in files in the local workspace."""
        results = []
        try:
            for file_path in self.base_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                rel_path_str = str(file_path.relative_to(self.base_dir))

                if include_pattern and not fnmatch.fnmatch(rel_path_str, include_pattern):
                    continue
                if exclude_pattern and fnmatch.fnmatch(rel_path_str, exclude_pattern):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    flags = 0 if case_sensitive else re.IGNORECASE
                    if re.search(query, content, flags):
                        results.append(f"workspace/{rel_path_str.replace('\\', '/')}")
                except (UnicodeDecodeError, re.error):
                    # Ignore binary files or regex errors
                    continue
        except Exception:
            logger.exception("Local content search failed")
        return results
