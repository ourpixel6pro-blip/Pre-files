# enhanced_lovable_agent.py
# Fixed version with local-first operations and proper error handling
import os
import re
import sys
import json
import time
import logging
import datetime
import tempfile
import fnmatch
import glob
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional, Union
from urllib.parse import urlparse
import base64
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Third-party clients
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from daytona import Daytona, DaytonaConfig, FileUpload, CreateSandboxFromSnapshotParams

from dotenv import load_dotenv
load_dotenv()

# ---------------------------
# Logging setup
# ---------------------------
LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("enhanced_lovable_agent")


@dataclass
class Config:
    gemini_api_key: str
    daytona_api_key: str
    stability_api_key: Optional[str] = None
    daytona_url: Optional[str] = None
    daytona_target: Optional[str] = None
    preview_port: int = 3000

    @staticmethod
    def load() -> "Config":
        """Load configuration from environment variables and validate required keys."""
        gemini_key = os.getenv("GEMINI_API_KEY")
        daytona_key = os.getenv("DAYTONA_API_KEY")

        if not gemini_key or not daytona_key:
            raise RuntimeError("Missing required API keys: GEMINI_API_KEY and DAYTONA_API_KEY")

        return Config(
            gemini_api_key=gemini_key,
            daytona_api_key=daytona_key,
            stability_api_key=os.getenv("STABILITY_API_KEY"),
            daytona_url=os.getenv("DAYTONA_API_URL"),
            daytona_target=os.getenv("DAYTONA_TARGET"),
            preview_port=int(os.getenv("PREVIEW_PORT", "3000")),
        )


class LocalFileManager:
    """Handle local file operations in workspace_local directory."""
    
    def __init__(self, console: Console, base_dir: str = "workspace_local"):
        self.console = console
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def _get_full_path(self, path: str) -> Path:
        """Convert relative path to full local path."""
        rel_path = path.replace("workspace/", "").lstrip("/")
        return self.base_dir / rel_path
    
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
    
    def search_files(self, query: str) -> List[str]:
        """Search for files in local workspace."""
        results = []
        try:
            for file_path in self.base_dir.rglob("*"):
                if file_path.is_file() and query.lower() in file_path.name.lower():
                    rel_path = file_path.relative_to(self.base_dir)
                    results.append(f"workspace/{rel_path}")
        except Exception as e:
            logger.exception("Local file search failed")
        return results
    
    def create_folder(self, path: str) -> None:
        """Create folder in local workspace."""
        full_path = self._get_full_path(path)
        full_path.mkdir(parents=True, exist_ok=True)
    
    def list_files(self, path: str = "") -> List[str]:
        """List files in local workspace."""
        if not path:
            search_path = self.base_dir
        else:
            search_path = self._get_full_path(path)
        
        results = []
        try:
            if search_path.exists():
                for item in search_path.rglob("*"):
                    if item.is_file():
                        rel_path = item.relative_to(self.base_dir)
                        results.append(f"workspace/{rel_path}")
        except Exception as e:
            logger.exception("Failed to list local files")
        return results


class DaytonaManager:
    """Encapsulate Daytona sandbox lifecycle and file operations."""

    def __init__(self, config: Config, console: Console):
        self.config = config
        self.console = console
        try:
            # Initialize Daytona client
            self.client = Daytona(DaytonaConfig(api_key=config.daytona_api_key, base_url=config.daytona_url))
        except Exception as e:
            logger.exception("Failed to initialize Daytona client")
            raise
        self.sandbox = None

    def create_sandbox(self) -> None:
        """Create sandbox with Python language and setup workspace."""
        with self.console.status("[bold green]Creating Daytona sandbox..."):
            try:
                # Create sandbox with Python language support
                self.sandbox = self.client.create()
            except Exception as e:
                logger.exception("Failed to create sandbox")
                raise RuntimeError(f"Daytona sandbox creation failed: {e}") from e

        self.console.print(Panel(f"Sandbox created: {getattr(self.sandbox, 'id', '<unknown>')}", title="Daytona"))

        # Create workspace directory
        try:
            self.sandbox.fs.create_folder("/home/daytona/workspace", "755")
            self.console.print("[green]Created workspace directory[/green]")
        except Exception as e:
            logger.debug("Workspace directory creation failed (might already exist): %s", e)

        # Upload files individually instead of bulk upload
        self.populate_from_local_folder("pre_files_sandbox")

        # Try to show preview link if available
        try:
            if hasattr(self.sandbox, "get_preview_link"):
                link_response = self.sandbox.get_preview_link(self.config.preview_port)
                if hasattr(link_response, 'url') and link_response.url:
                    url = link_response.url
                    self.console.print(Panel(f"[link={url}]{url}[/link]", title="Preview URL",
                                             subtitle="Daytona Live Preview"))
                else:
                    self.console.print(Panel("Preview URL not available", title="Preview URL", style="yellow"))
        except Exception as e:
            logger.warning("Could not fetch preview link: %s", e)
            self.console.print(Panel(f"Preview URL unavailable: {e}", title="Preview URL", style="yellow"))

    def populate_from_local_folder(self, folder: str) -> None:
        """Upload files from a local folder into the sandbox filesystem using FileUpload."""
        if not os.path.isdir(folder):
            logger.debug("No local folder %s to populate", folder)
            return

        if not self.sandbox:
            raise RuntimeError("Sandbox not available for upload")

        uploads = []
        successful_uploads = set()  # Track successfully uploaded files
        failed_uploads = []
        skipped_count = 0
        created_dirs = set()

        # First pass: collect all directories and create a directory tree
        dir_tree = {"/home/daytona/workspace": set()}
        for root, _, _ in os.walk(folder):
            rel_path = os.path.relpath(root, folder)
            if rel_path == ".":
                continue
            sandbox_dir = f"/home/daytona/workspace/{rel_path}".replace("\\", "/")
            parent_dir = os.path.dirname(sandbox_dir)
            if parent_dir not in dir_tree:
                dir_tree[parent_dir] = set()
            dir_tree[parent_dir].add(sandbox_dir)
            dir_tree[sandbox_dir] = set()

        # Create directories in parallel using a thread pool
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        def create_dir(dir_path):
            if dir_path not in created_dirs:
                try:
                    self.sandbox.fs.create_folder(dir_path, "755")
                    created_dirs.add(dir_path)
                    return True, dir_path
                except Exception as e:
                    logger.debug("Directory creation failed (might exist): %s", e)
                    return False, dir_path
            return True, dir_path

        # Process directories level by level to maintain parent-child relationship
        with ThreadPoolExecutor(max_workers=10) as executor:
            current_level = {"/home/daytona/workspace"}
            while current_level:
                next_level = set()
                futures = []
                for dir_path in current_level:
                    if dir_path not in created_dirs:
                        futures.append(executor.submit(create_dir, dir_path))
                    next_level.update(dir_tree[dir_path])
                
                for future in futures:
                    success, dir_path = future.result()
                    if success:
                        self.console.print(f"[blue]Created folder {dir_path}[/blue]")
                
                current_level = next_level

        # Collect all files with optimized filtering
        # Prepare file uploads
        for root, _, files in os.walk(folder):
            rel_path = os.path.relpath(root, folder)
            continue               
                    
            local_path = os.path.join(root, file)
            sandbox_path = f"/home/daytona/workspace/{rel_path}/{file}".replace("\\", "/")
                
                # Skip if already uploaded successfully
            if sandbox_path in successful_uploads:
                 continue

            try:
                    uploads.append(FileUpload(
                        source=str(Path(local_path).absolute()),
                        destination=sandbox_path
                    ))
            except Exception as e:
                    logger.warning("Failed to prepare upload for %s: %s", rel_path, e)
                    continue

        if not uploads:
            self.console.print("[yellow]No new files to upload[/yellow]")
            return

        # Parallel upload using batches
        def upload_batch(batch):
            results = []
            try:
                self.sandbox.fs.upload_files(batch)
                results.extend((True, upload) for upload in batch)
            except Exception as e:
                # Try individual uploads if batch fails
                for upload in batch:
                    try:
                        self.sandbox.fs.upload_file(
                            Path(upload.source).read_bytes(),
                            upload.destination
                        )
                        results.append((True, upload))
                    except Exception as e:
                        results.append((False, (upload, str(e))))
            return results

        batch_size = 50
        batches = [uploads[i:i + batch_size] for i in range(0, len(uploads), batch_size)]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(upload_batch, batch) for batch in batches]
            
            for future in futures:
                try:
                    for result in future.result():
                        if result[0]:  # Success
                            upload = result[1]
                            successful_uploads.add(upload.destination)
                            rel_path = upload.destination.replace("/home/daytona/workspace/", "")
                            self.console.print(f"[green]Copied {rel_path} to sandbox[/green]")
                        else:  # Failure
                            upload, error = result[1]
                            failed_uploads.append((upload.destination, error))
                except Exception as e:
                    logger.error("Error processing upload batch results: %s", e)

        # Print final summary
        self.console.print(f"[green]Successfully uploaded {len(successful_uploads)} files[/green]")
        if failed_uploads:
            self.console.print(f"[red]Failed to upload {len(failed_uploads)} files:[/red]")
            for path, error in failed_uploads:
                self.console.print(f"[red]- {path}: {error}[/red]")
        if skipped_count > 0:
            self.console.print(f"[yellow]Skipped {skipped_count} files[/yellow]")

    def _ensure_sandbox(self):
        if not self.sandbox:
            raise RuntimeError("Sandbox is not initialized")

    def _normalize_path(self, path: str) -> str:
        """Convert to Daytona absolute path format."""
        # Convert path to POSIX format
        posix_path = path.replace("\\", "/")
        
        # Remove workspace/ prefix if present
        if posix_path.startswith("workspace/"):
            posix_path = posix_path[10:]
            
        # Ensure path starts with /home/daytona/workspace/
        if not posix_path.startswith("/"):
            posix_path = f"/home/daytona/workspace/{posix_path}"
        elif not posix_path.startswith("/home/daytona/workspace/"):
            posix_path = f"/home/daytona/workspace/{posix_path.lstrip('/')}"
            
        return posix_path

    def read_file(self, path: str) -> bytes:
        """Read a file from the sandbox filesystem."""
        self._ensure_sandbox()
        norm_path = self._normalize_path(path)
        try:
            content = self.sandbox.fs.download_file(norm_path)
            if content is None:
                raise RuntimeError(f"File {norm_path} not found or empty")
            if isinstance(content, bytes):
                return content
            elif isinstance(content, str):
                return content.encode('utf-8')
            else:
                return str(content).encode('utf-8')
        except Exception as e:
            logger.exception("Failed to read file %s", norm_path)
            raise

    def upload_file(self, path: str, content: Union[str, bytes]) -> None:
        """Upload/overwrite a file to the sandbox filesystem."""
        self._ensure_sandbox()
        norm_path = self._normalize_path(path)
        
        try:
            # Create parent directory if needed
            parent_dir = os.path.dirname(norm_path)
            if parent_dir and parent_dir != "/home/daytona/workspace":
                try:
                    self.sandbox.fs.create_folder(parent_dir, "755")
                except Exception as e:
                    logger.debug("Directory creation failed (might exist): %s", e)

            # Convert content to bytes
            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
            elif isinstance(content, bytes):
                content_bytes = content
            else:
                content_bytes = str(content).encode('utf-8')

            # Try upload with FileUpload first
            try:
                # Save content to temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(content_bytes)
                    temp_path = temp_file.name

                upload = FileUpload(
                    source=temp_path,
                    destination=norm_path
                )
                self.sandbox.fs.upload_files([upload])

                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

            except Exception as e:
                logger.debug("FileUpload failed, falling back to direct upload: %s", e)
                # Fallback to direct upload
                self.sandbox.fs.upload_file(content_bytes, norm_path)

        except Exception as e:
            logger.exception("Failed to upload file %s", norm_path)
            raise RuntimeError(f"Failed to upload file {norm_path}: {e}")

    def execute_command(self, cmd: str, timeout: int = 30) -> str:
        """Execute a command within sandbox with timeout."""
        self._ensure_sandbox()
        try:
            import signal
            from threading import Timer
            
            def timeout_handler():
                raise TimeoutError(f"Command timed out after {timeout} seconds: {cmd}")
            
            # Set up timeout
            timer = Timer(timeout, timeout_handler)
            timer.start()
            
            try:
                response = self.sandbox.process.exec(cmd)
            finally:
                timer.cancel()
            
            # Handle response based on actual structure
            if hasattr(response, 'result'):
                return str(response.result)
            elif hasattr(response, 'output'):
                return str(response.output)
            elif hasattr(response, 'stdout'):
                stdout = response.stdout if isinstance(response.stdout, str) else response.stdout.decode(errors="replace")
                stderr = ""
                if hasattr(response, 'stderr') and response.stderr:
                    stderr = response.stderr if isinstance(response.stderr, str) else response.stderr.decode(errors="replace")
                return (stdout + stderr).strip() if (stdout + stderr).strip() else "Command executed successfully"
            else:
                return "Command executed successfully"
                
        except Exception as e:
            logger.exception("Command failed: %s", cmd)
            return f"Command failed: {e}"

    def delete_file(self, path: str) -> None:
        """Delete a file from the sandbox filesystem."""
        self._ensure_sandbox()
        norm_path = self._normalize_path(path)
        try:
            self.sandbox.fs.delete_file(norm_path)
        except Exception:
            logger.exception("Failed to delete file %s", norm_path)
            raise

    def move_file(self, src: str, dest: str) -> None:
        """Move/rename a file in the sandbox filesystem."""
        self._ensure_sandbox()
        norm_src = self._normalize_path(src)
        norm_dest = self._normalize_path(dest)
        try:
            content = self.read_file(norm_src)
            self.upload_file(norm_dest, content)
            self.delete_file(norm_src)
        except Exception:
            logger.exception("Failed to move %s -> %s", norm_src, norm_dest)
            raise

    def search_files(self, query: str) -> List[str]:
        """Search for files in the sandbox filesystem."""
        self._ensure_sandbox()
        try:
            all_files = self._get_all_files()
            matching_files = [f for f in all_files if query.lower() in f.lower()]
            return matching_files
        except Exception as e:
            logger.exception("File search failed for query %s", query)
            return []

    def _get_all_files(self, path: str = "") -> List[str]:
        """Recursively get all files in the sandbox."""
        files = []
        try:
            if hasattr(self.sandbox.fs, 'list_files'):
                items = self.sandbox.fs.list_files(path=path or ".")
                if items:
                    for item in items:
                        if hasattr(item, 'name'):
                            item_path = f"{path}/{item.name}".strip('/') if path else item.name
                            if hasattr(item, 'is_directory') and item.is_directory:
                                files.extend(self._get_all_files(item_path))
                            else:
                                files.append(f"workspace/{item_path}")
                        elif isinstance(item, str):
                            item_path = f"{path}/{item}".strip('/') if path else item
                            files.append(f"workspace/{item_path}")
        except Exception as e:
            logger.warning(f"Failed to list files in {path}: {e}")
        return files

    def create_folder(self, path: str, mode: str = "755") -> None:
        """Create a folder in the sandbox filesystem."""
        self._ensure_sandbox()
        norm_path = self._normalize_path(path)
        try:
            self.sandbox.fs.create_folder(norm_path, mode)
        except Exception:
            logger.exception("Failed to create folder %s", norm_path)
            raise

    def destroy_sandbox(self) -> None:
        if not self.sandbox:
            logger.debug("No sandbox to destroy")
            return
        try:
            # Stop the dev server if running
            try:
                self.execute_command("pkill -f 'npm run dev' || true")
                self.console.print("[yellow]Stopped development server[/yellow]")
            except Exception as e:
                logger.warning("Failed to stop dev server: %s", e)
            
            # Delete the sandbox
            self.sandbox.delete()
            self.console.print("[yellow]Sandbox destroyed[/yellow]")
        except Exception:
            logger.exception("Error destroying sandbox")
            self.console.print("[red]Error destroying sandbox[/red]")


# ---------------------------
# Load tools from JSON
# ---------------------------
def load_tools_config() -> List[Dict]:
    """Load tool configurations from tools.json file or return Lovable standard tools."""
    try:
        with open("tools.json", "r", encoding="utf-8") as f:
            tools_config = json.load(f)
        logger.info(f"Loaded {len(tools_config)} tool configurations")
        return tools_config
    except FileNotFoundError:
        logger.warning("tools.json file not found, using Lovable standard tool set")
        return 
    except Exception as e:
        logger.error(f"Failed to load tools.json: {e}")
        raise RuntimeError(f"Failed to load tools configuration: {e}")

       # Global reference to executor - will be set in main()
    _executor = None
    
    def set_executor(executor):
        nonlocal _executor
        _executor = executor
    


def load_system_prompt() -> str:
    """Load system prompt from file."""
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning("prompt.txt not found, using default")
        return
    except Exception as e:
        logger.error("Failed to load system prompt: %s", e)
        return "You are a helpful AI assistant."


class ToolExecutor:
    """Dispatch and implement available tool functions with local-first operations."""

    def __init__(self, dm: DaytonaManager, lfm: LocalFileManager, console: Console, config: Config):
        self.dm = dm
        self.lfm = lfm
        self.console = console
        self.config = config

    def _apply_local_first(self, operation: str, local_func: Callable, sandbox_func: Callable, *args, **kwargs):
        """Apply operation locally first, then to sandbox."""
        results = []
        
        # Apply locally first
        try:
            local_result = local_func(*args, **kwargs)
            results.append(f"✓ Local {operation} successful")
            if local_result and isinstance(local_result, str):
                results.append(local_result)
        except Exception as e:
            results.append(f"✗ Local {operation} failed: {e}")
        
        # Then apply to sandbox
        try:
            sandbox_result = sandbox_func(*args, **kwargs)
            results.append(f"✓ Sandbox {operation} successful")
            if sandbox_result and isinstance(sandbox_result, str):
                results.append(sandbox_result)
        except Exception as e:
            results.append(f"✗ Sandbox {operation} failed: {e}")
        
        return "\n".join(results)

    def lov_view(self, args):
        """View file contents (try local first, then sandbox)."""
        path = args.get("path")
        try:
            # Try local first
            content = self.lfm.read_file(path)
            return f"[Local] {content.decode('utf-8', errors='replace')}"
        except:
            try:
                # Fallback to sandbox
                content = self.dm.read_file(path)
                return f"[Sandbox] {content.decode('utf-8', errors='replace')}"
            except Exception as e:
                return f"Error reading file {path}: {e}"

    def lov_write(self, args):
        """Write content to file (local first, then sandbox)."""
        path = args.get("path")
        content = args.get("content", "")
        
        def local_write():
            self.lfm.write_file(path, content)
            return f"Wrote {len(content)} characters to local {path}"
        
        def sandbox_write():
            self.dm.upload_file(path, content)
            return f"Wrote {len(content)} characters to sandbox {path}"
        
        return self._apply_local_first("write", local_write, sandbox_write)

    def lov_line_replace(self, args):
        """Line-based search and replace (local first, then sandbox)."""
        file_path = args.get("file_path")
        search = args.get("search")
        first_line = args.get("first_replaced_line", 1)
        last_line = args.get("last_replaced_line", 1)
        replace = args.get("replace", "")
        
        def perform_replace(read_func, write_func):
            content = read_func(file_path).decode('utf-8', errors='replace')
            lines = content.splitlines()
            if first_line < 1 or last_line > len(lines) or first_line > last_line:
                raise RuntimeError(f"Invalid line range: {first_line}-{last_line} for file with {len(lines)} lines")
            start_idx = first_line - 1
            end_idx = last_line
            new_lines = lines[:start_idx] + replace.splitlines() + lines[end_idx:]
            new_content = "\n".join(new_lines)
            write_func(file_path, new_content)
            return f"Replaced lines {first_line}-{last_line}"
        
        def local_replace():
            return perform_replace(self.lfm.read_file, self.lfm.write_file)
        
        def sandbox_replace():
            return perform_replace(self.dm.read_file, self.dm.upload_file)
        
        return self._apply_local_first("line replace", local_replace, sandbox_replace)

    def lov_search_files(self, args):
        """Search for files (both local and sandbox)."""
        query = args.get("query")
        results = []
        
        try:
            local_results = self.lfm.search_files(query)
            if local_results:
                results.append(f"Local files ({len(local_results)}):")
                results.extend(local_results)
        except Exception as e:
            results.append(f"Local search failed: {e}")
        
        try:
            sandbox_results = self.dm.search_files(query)
            if sandbox_results:
                results.append(f"Sandbox files ({len(sandbox_results)}):")
                results.extend(sandbox_results)
        except Exception as e:
            results.append(f"Sandbox search failed: {e}")
        
        return "\n".join(results) if results else "No files found matching the query"

    def lov_rename(self, args):
        """Rename/move file (local first, then sandbox)."""
        original_file_path = args.get("original_file_path")
        new_file_path = args.get("new_file_path")
        
        def local_rename():
            self.lfm.move_file(original_file_path, new_file_path)
            return f"Moved {original_file_path} to {new_file_path}"
        
        def sandbox_rename():
            self.dm.move_file(original_file_path, new_file_path)
            return f"Moved {original_file_path} to {new_file_path}"
        
        return self._apply_local_first("rename", local_rename, sandbox_rename)

    def lov_delete(self, args):
        """Delete file (local first, then sandbox)."""
        file_path = args.get("file_path")
        
        def local_delete():
            self.lfm.delete_file(file_path)
            return f"Deleted {file_path}"
        
        def sandbox_delete():
            self.dm.delete_file(file_path)
            return f"Deleted {file_path}"
        
        return self._apply_local_first("delete", local_delete, sandbox_delete)

    def lov_add_dependency(self, args):
        """Add npm dependency (sandbox only)."""
        package = args.get("package")
        try:
            result = self.dm.execute_command(f"cd /home/daytona/workspace && npm install {package}")
            return f"Installed {package}:\n{result}"
        except Exception as e:
            return f"Error installing {package}: {e}"

    def lov_remove_dependency(self, args):
        """Remove npm dependency (sandbox only)."""
        package = args.get("package")
        try:
            result = self.dm.execute_command(f"cd /home/daytona/workspace && npm uninstall {package}")
            return f"Removed {package}:\n{result}"
        except Exception as e:
            return f"Error removing {package}: {e}"

    def lov_download_to_repo(self, args):
        """Download file from URL (local first, then sandbox)."""
        source_url = args.get("source_url")
        target_path = args.get("target_path")
        
        try:
            response = requests.get(source_url, timeout=30)
            response.raise_for_status()
            content = response.content
            
            def local_download():
                self.lfm.write_file(target_path, content)
                return f"Downloaded {source_url} to local {target_path}"
            
            def sandbox_download():
                self.dm.upload_file(target_path, content)
                return f"Downloaded {source_url} to sandbox {target_path}"
            
            return self._apply_local_first("download", local_download, sandbox_download)
            
        except Exception as e:
            return f"Error downloading {source_url}: {e}"

    def execute_function_calls(self, function_calls: List) -> List[Dict[str, Any]]:
        """Execute function calls from Gemini's function calling API."""
        results = []
        
        # Categorize functions as local-first or sandbox-only
        local_first_operations = {
            "lov_write",
            "lov_line_replace",
            "lov_rename",
            "lov_delete",
            "lov_download_to_repo",
            "lov_view",
            "lov_search_files",
        }
        
        sandbox_only_operations = {
            "lov_add_dependency",
            "lov_remove_dependency",
        }
        
        
        dispatch = {
            "lov_view": self.lov_view,
            "lov_write": self.lov_write,
            "lov_line_replace": self.lov_line_replace,
            "lov_search_files": self.lov_search_files,
            "lov_rename": self.lov_rename,
            "lov_delete": self.lov_delete,
            "lov_add_dependency": self.lov_add_dependency,
            "lov_remove_dependency": self.lov_remove_dependency,
            "lov_download_to_repo": self.lov_download_to_repo,
        }
        
        for call in function_calls:
            name = call.name
            args = dict(call.args) if hasattr(call, 'args') else {}
            
            self.console.print(f"[yellow]Executing tool: {name}[/yellow]")
            
            handler = dispatch.get(name)
            if not handler:
                result = {"name": name, "error": f"Unknown tool: {name}"}
                self.console.print(f"[red]Unknown tool: {name}[/red]")
                results.append(result)
                continue

            try:
                if name in local_first_operations:
                    # For local-first operations, try local first then sandbox
                    try:
                        # Apply locally first
                        if name == "lov_write":
                            self.lfm.write_file(args["path"], args["content"])
                        elif name == "lov_line_replace":
                            content = self.lfm.read_file(args["file_path"]).decode('utf-8')
                            lines = content.splitlines()
                            start_idx = args["first_replaced_line"] - 1
                            end_idx = args["last_replaced_line"]
                            new_lines = lines[:start_idx] + args["replace"].splitlines() + lines[end_idx:]
                            self.lfm.write_file(args["file_path"], "\n".join(new_lines))
                        elif name == "lov_rename":
                            self.lfm.move_file(args["original_file_path"], args["new_file_path"])
                        elif name == "lov_delete":
                            self.lfm.delete_file(args["file_path"])
                        elif name == "lov_download_to_repo":
                            response = requests.get(args["source_url"], timeout=30)
                            response.raise_for_status()
                            self.lfm.write_file(args["target_path"], response.content)
                        
                        self.console.print(f"[green]Local operation successful: {name}[/green]")
                        
                        # Then apply to sandbox
                        output = handler(args)
                        result = {"name": name, "output": str(output)}
                        
                    except Exception as local_e:
                        self.console.print(f"[red]Local operation failed: {local_e}, trying sandbox...[/red]")
                        output = handler(args)
                        result = {"name": name, "output": str(output)}
                
                elif name in sandbox_only_operations:
                    # For sandbox-only operations, just execute normally
                    output = handler(args)
                    result = {"name": name, "output": str(output)}
                
                else:
                    # For hybrid operations (view/search), execute normally as they already handle both
                    output = handler(args)
                    result = {"name": name, "output": str(output)}
                
                self.console.print(f"[green]Tool {name} completed successfully[/green]")
                results.append(result)
                
            except Exception as e:
                logger.exception("Tool call failed: %s", name)
                result = {"name": name, "error": str(e)}
                self.console.print(f"[red]Tool {name} failed: {e}[/red]")
                results.append(result)
                
        return results


def create_tool_functions() -> List[Callable]:
    """Create the actual Python functions that will be passed to Gemini as tools."""
    
    # Global reference to executor - will be set in main()
    _executor = None
    
    def set_executor(executor):
        nonlocal _executor
        _executor = executor
    



def main():
    console = Console()
    console.print(Panel("[bold cyan]Enhanced Lovable Agent with Local-First Operations[/bold cyan]"))

    try:
        config = Config.load()
    except Exception as e:
        logger.exception("Configuration load failed")
        console.print(Panel(f"[red]Configuration error: {e}[/red]"))
        sys.exit(1)

    # Load system prompt and tools config
    system_prompt = load_system_prompt()
    tools_config = load_tools_config()
    
    console.print(f"[green]Loaded system prompt ({len(system_prompt)} chars)[/green]")
    console.print(f"[green]Loaded {len(tools_config)} tool configurations[/green]")

    # Initialize local file manager
    lfm = LocalFileManager(console)
    
    # Copy files from pre_files_sandbox to workspace_local
    pre_folder = "pre_files_sandbox"
    if os.path.isdir(pre_folder):
        for root, _, files in os.walk(pre_folder):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, pre_folder)
                try:
                    with open(src_path, "rb") as src_f:
                        content = src_f.read()
                    lfm.write_file(f"workspace/{rel_path}", content)
                    console.print(f"[green]Copied {rel_path} to workspace_local[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to copy {rel_path}: {e}[/red]")

    # Create Daytona manager and sandbox
    try:
        dm = DaytonaManager(config, console)
        dm.create_sandbox()
    except Exception as e:
        console.print(Panel(f"[red]Failed to create Daytona sandbox: {e}[/red]"))
        sys.exit(1)

    executor = ToolExecutor(dm, lfm, console, config)

    # Initialize Gemini client
    if genai is None or types is None:
        console.print(Panel("[red]google-genai library not available. Please install it.[/red]"))
        sys.exit(1)
    
    try:
        client = genai.Client(api_key=config.gemini_api_key)
    except Exception as e:
        console.print(Panel(f"[red]Failed to initialize Gemini client: {e}[/red]"))
        sys.exit(1)

    # Create tool functions and set up the executor reference
    tool_functions, set_executor_func = create_tool_functions()
    set_executor_func(executor)

    # Create chat with proper system instruction and tools
    try:
        # First, attempt to use the primary model
        console.print("[cyan]Attempting to initialize chat with gemini-2.5-pro...[/cyan]")
        chat = client.chats.create(
            model="gemini-2.5-pro",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=tool_functions,
                # Disable thinking for faster responses in interactive mode
                thinking_config=types.ThinkingConfig(thinking_budget=32768)
            )
        )
        console.print("[green]Chat initialized successfully with gemini-2.5-pro[/green]")

    except Exception as e:
        # This block runs if the primary model fails
        console.print(Panel(f"[yellow]Failed to use gemini-2.5-pro: {e}.\nFalling back to gemini-2.5-flash...[/yellow]"))

            # Second, attempt to use the fallback model
        console.print("[cyan]Attempting to initialize chat with gemini-2.5-flash...[/cyan]")
        chat = client.chats.create(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=tool_functions,
                    thinking_config=types.ThinkingConfig(thinking_budget=32768)
                )
            )
        console.print("[green]Chat initialized successfully with gemini-2.5-flash[/green]")
        
    except Exception as e_fallback:
        # This block runs if the fallback model also fails
        console.print(Panel(f"[red]Failed to create chat with fallback model: {e_fallback}[/red]"))
        sys.exit(1)
    # Main conversation loop
    try:
        # After sandbox creation, check if package.json exists and run npm commands
        console.print(Panel("[bold green]Setting up development environment...[/bold green]"))
        
        # Check for package.json in workspace
        try:
            package_json_content = dm.read_file("workspace/package.json")
            console.print("[green]Found package.json in sandbox[/green]")
            
            # Install dependencies and start the dev server on port 3000
            console.print("[cyan]Running npm install and starting dev server...[/cyan]")
            setup_cmd = "cd /home/daytona/workspace && npm install && pkill -f 'npm run dev' || true && npm run dev -- --port 3000 > /tmp/dev-server.log 2>&1 &"
            dm.execute_command(setup_cmd)

            # Give the server a moment to start after install and launch
            console.print("[cyan]Waiting for dev server to initialize...[/cyan]")
            time.sleep(10) # Increased sleep time for install
                
                # Check if the server started by looking for the process
            check_cmd = "pgrep -f 'npm run dev'"
            check_output = dm.execute_command(check_cmd)
                
            if check_output:
                    # Get the server logs
                    log_output = dm.execute_command("tail -n 20 /tmp/dev-server.log")
                    console.print(Panel(log_output, title="Development Server Startup Log", border_style="green"))
                    console.print("[green]Development server is running on port 3000[/green]")
            else:
                    console.print("[red]Failed to start development server[/red]")
                    
        except Exception as e:
            console.print(f"[red]Error starting development server: {e}[/red]")

        # List current files in sandbox
        try:
            files = dm._get_all_files()
            console.print(Panel("\n".join(files), title="Sandbox Files", border_style="cyan"))
        except Exception as e:
            console.print(Panel(f"[red]Failed to list sandbox files: {e}[/red]"))

        # List local files
        try:
            local_files = lfm.list_files()
            console.print(Panel("\n".join(local_files), title="Local Files", border_style="magenta"))
        except Exception as e:
            console.print(Panel(f"[red]Failed to list local files: {e}[/red]"))

        # Clear separation after setup
        console.print("\n" + "="*80 + "\n")
        console.print("[bold green]Setup complete! You can now interact with the chat.[/bold green]")
        console.print("[cyan]Type your questions or commands below, or 'exit' to quit.[/cyan]\n")

        while True:
            user_input = Prompt.ask("[bold cyan]You")
            if not user_input:
                continue
            if user_input.strip().lower() in {"exit", "quit", "bye"}:
                console.print(Panel("[green]Goodbye![green]"))
                break

            console.print(Panel("Lovable is thinking...", style="magenta"))

            try:
                # Send message to chat
                response = chat.send_message(user_input)
                
                # Check if there are function calls to execute
                if hasattr(response, 'function_calls') and response.function_calls:
                    console.print(f"[yellow]Executing {len(response.function_calls)} tool call(s)...[yellow]")
                    
                    # Execute the function calls
                    results = executor.execute_function_calls(response.function_calls)
                    
                    console.print("[green]Tool execution completed[green]")
                
                # Display the response text
                if hasattr(response, 'text') and response.text:
                    console.print(Panel(response.text, title="Lovable", border_style="blue"))
                elif not (hasattr(response, 'function_calls') and response.function_calls):
                    console.print("[yellow]No text response received[yellow]")

            except KeyboardInterrupt:
                console.print("\n[red]Interrupted by user[red]")
                break
            except Exception as e:
                logger.exception("Error in chat interaction")
                console.print(f"[red]Chat error: {e}[red]")
                continue

    except KeyboardInterrupt:
        console.print("\n[yellow]Chat interrupted by user[yellow]")
    finally:
        try:
            dm.destroy_sandbox()
        except Exception:
            logger.exception("Error during sandbox cleanup")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception:
        logger.exception("Unhandled exception in main")
        sys.exit(1)