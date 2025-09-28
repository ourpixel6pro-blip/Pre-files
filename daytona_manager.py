import os
import tempfile
import time
import hashlib
from pathlib import Path
from typing import List, Union, Optional
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console
from rich.panel import Panel
try:
    from daytona import Daytona, DaytonaConfig, FileUpload
except ImportError as e:
    print(f"Warning: Daytona SDK import failed: {e}")
    print("Continuing without Daytona functionality...")
    Daytona = None
    DaytonaConfig = None
    FileUpload = None
from rich.progress import Progress

from config import Config
from utils import get_logger

logger = get_logger(__name__)

class DaytonaManager:
    """Encapsulate Daytona sandbox lifecycle and file operations."""

    def __init__(self, config: Config, console: Console):
        self.config = config
        self.console = console
        self.sandbox = None
        self.client = None
        
        # Initialize Daytona client if available
        if Daytona is not None:
            try:
                self.client = Daytona(DaytonaConfig(api_key=config.daytona_api_key, base_url=config.daytona_url))
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to initialize Daytona client: {e}[/yellow]")
                self.client = None
        else:
            console.print("[yellow]Daytona SDK not available - running in local-only mode[/yellow]")

    def create_sandbox(self, retries=3, delay=5) -> None:
        """
        Create a sandbox, trying to use backup/snapshot first for speed.
        If no backup exists, create fresh sandbox and make a backup for future use.
        """
        if self.client is None:
            self.console.print("[yellow]Daytona client not available - skipping sandbox creation[/yellow]")
            return
            
        backup_name = "lovable-agent-base"
        
        try:
            # First, try to find an existing sandbox backup/snapshot
            self.console.print("[cyan]Looking for existing sandbox backup...[/cyan]")
            existing_backup = self._find_existing_backup(backup_name)
            
            if existing_backup:
                self.console.print(f"[green]Found backup '{backup_name}'. Creating sandbox from it...[/green]")
                # Create sandbox from the backup (this would be implemented if we find the right API)
                # For now, fall back to fresh creation
                self._create_fresh_sandbox_and_backup(backup_name, retries, delay)
            else:
                self.console.print("[yellow]No backup found. Creating fresh sandbox and backup for future runs...[/yellow]")
                self._create_fresh_sandbox_and_backup(backup_name, retries, delay)
                
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise RuntimeError(f"Daytona sandbox creation failed: {e}") from e

    def _find_existing_backup(self, backup_name: str) -> Optional[str]:
        """Look for existing backup. This is a placeholder for now."""
        # TODO: Implement backup lookup using Daytona API
        return None

    def _create_fresh_sandbox_and_backup(self, backup_name: str, retries: int, delay: int) -> None:
        """Create a fresh sandbox, populate it, and create a backup."""
        for attempt in range(retries):
            try:
                self.console.print("[cyan]Creating fresh Daytona sandbox...[/cyan]")
                self.sandbox = self.client.create()
                self.console.print(Panel(f"Sandbox created: {getattr(self.sandbox, 'id', '<unknown>')}", title="Daytona"))
                
                self.sandbox.fs.create_folder("/home/daytona/workspace", "755")
                self.console.print("[green]Created workspace directory[/green]")
                self.populate_from_local_folder(self.config.workspace_dir)
                self._show_preview_link()
                
                # Create backup for future runs
                self._create_sandbox_backup(backup_name)
                return
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{retries} failed to create sandbox: {e}")
                if attempt < retries - 1:
                    self.console.print(f"[yellow]Retrying in {delay} seconds...[/yellow]")
                    time.sleep(delay)
                else:
                    logger.exception("Failed to create sandbox after multiple retries")
                    raise RuntimeError(f"Daytona sandbox creation failed: {e}") from e

    def _create_sandbox_backup(self, backup_name: str) -> None:
        """Create a backup of the current sandbox using the Daytona API."""
        if not self.config.backup_enabled:
            logger.info("Backup creation disabled in configuration")
            return
            
        try:
            self.console.print(f"[cyan]Creating backup '{backup_name}' for faster future startups...[/cyan]")
            
            # Use the backup API endpoint
            import requests
            headers = {
                "Authorization": f"Bearer {self.config.daytona_api_key}",
                "Content-Type": "application/json"
            }
            
            # Try different possible API endpoints
            base_urls = [
                self.config.daytona_url,
                "https://api.daytona.io",
                "https://app.daytona.io/api"
            ]
            
            backup_created = False
            for base_url in base_urls:
                if not base_url:
                    continue
                    
                try:
                    backup_url = f"{base_url.rstrip('/')}/sandboxes/{self.sandbox.id}/backup"
                    logger.debug(f"Trying backup URL: {backup_url}")
                    
                    response = requests.post(backup_url, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    self.console.print("[green]Backup created successfully![/green]")
                    self.console.print("[cyan]Future runs will start much faster using this backup.[/cyan]")
                    backup_created = True
                    break
                    
                except requests.exceptions.RequestException as url_error:
                    logger.debug(f"Backup failed for URL {backup_url}: {url_error}")
                    continue
            
            if not backup_created:
                logger.info("Backup creation not available with current Daytona setup")
                self.console.print("[yellow]Backup creation not available with current setup.[/yellow]")
            
        except Exception as e:
            # Don't fail the entire process if backup creation fails
            logger.warning(f"Failed to create backup (non-critical): {e}")
            self.console.print(f"[yellow]Could not create backup: {e}[/yellow]")
            self.console.print("[yellow]This won't affect functionality, but future startups will be slower.[/yellow]")

    def create_backup(self, backup_name: str) -> str:
        
        """Create a backup of the current sandbox using the Daytona API.
        
        Args:
            backup_name (str): Name for the backup
            
        Returns:
            str: Status message about backup creation
        """
        if not self.config.backup_enabled:
            return "Backup creation disabled in configuration"
            
        if not hasattr(self, 'sandbox') or not self.sandbox:
            return "No active sandbox to backup"
            
        try:
            self.console.print(f"[cyan]Creating backup '{backup_name}' for faster future startups...[/cyan]")
            
            # Use the backup API endpoint
            import requests
            headers = {
                "Authorization": f"Bearer {self.config.daytona_api_key}",
                "Content-Type": "application/json"
            }
            
            # Try different possible API endpoints
            base_urls = [
                self.config.daytona_url,
                "https://api.daytona.io",
                "https://app.daytona.io/api"
            ]
            
            backup_created = False
            for base_url in base_urls:
                if not base_url:
                    continue
                    
                try:
                    backup_url = f"{base_url.rstrip('/')}/sandboxes/{self.sandbox.id}/backup"
                    logger.debug(f"Trying backup URL: {backup_url}")
                    
                    response = requests.post(backup_url, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    self.console.print("[green]Backup created successfully![/green]")
                    self.console.print("[cyan]Future runs will start much faster using this backup.[/cyan]")
                    backup_created = True
                    return f"Backup '{backup_name}' created successfully"
                    
                except requests.exceptions.RequestException as url_error:
                    logger.debug(f"Backup failed for URL {backup_url}: {url_error}")
                    continue
            
            if not backup_created:
                logger.info("Backup creation not available with current Daytona setup")
                return "Backup creation not available with current setup"
            
        except Exception as e:
            # Don't fail the entire process if backup creation fails
            logger.warning(f"Failed to create backup (non-critical): {e}")
            return f"Could not create backup: {e}"

    def get_backup_status(self) -> str:
        """Get the current backup status of the sandbox.
        
        Returns:
            str: Status message about backup state
        """
        if not hasattr(self, 'sandbox') or not self.sandbox:
            return "No active sandbox"
            
        try:
            # Check if sandbox has backup information
            if hasattr(self.sandbox, 'backup_state'):
                return f"Backup state: {self.sandbox.backup_state}"
            else:
                return "Backup status not available"
                
        except Exception as e:
            logger.warning(f"Failed to get backup status: {e}")
            return f"Could not get backup status: {e}"

    def _show_preview_link(self):
        try:
            if hasattr(self.sandbox, "get_preview_link"):
                link_response = self.sandbox.get_preview_link(self.config.preview_port)
                if hasattr(link_response, 'url') and link_response.url:
                    url = link_response.url
                    self.console.print(Panel(f"[link={url}]{url}[/link]", title="Preview URL", subtitle="Daytona Live Preview"))
                else:
                    self.console.print(Panel("Preview URL not available", title="Preview URL", style="yellow"))
        except Exception as e:
            logger.warning("Could not fetch preview link: %s", e)

    def populate_from_local_folder(self, folder: str) -> None:
        """Upload files from a local folder into the sandbox filesystem."""
        if not self.sandbox or FileUpload is None:
            if FileUpload is None:
                self.console.print("[yellow]FileUpload not available - skipping file population[/yellow]")
            return

        # Auto-populate workspace_local if it's empty
        if not os.path.isdir(folder) or not os.listdir(folder):
            if os.path.isdir("workspace") and os.listdir("workspace"):
                self.console.print(f"[cyan]Auto-populating {folder} from workspace...[/cyan]")
                self._copy_directory("workspace", folder)
            elif os.path.isdir("pre_files_sandbox") and os.listdir("pre_files_sandbox"):
                self.console.print(f"[cyan]Auto-populating {folder} from pre_files_sandbox...[/cyan]")
                self._copy_directory("pre_files_sandbox", folder)
            else:
                self.console.print(f"[yellow]No files to upload from {folder}[/yellow]")
                return

        uploads = []
        for root, _, files in os.walk(folder):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, folder)
                sandbox_path = f"/home/daytona/workspace/{rel_path}".replace("\\", "/")
                uploads.append(FileUpload(source=str(Path(local_path).absolute()), destination=sandbox_path))
        
        if not uploads:
            self.console.print(f"[yellow]No new files to upload from {folder}[/yellow]")
            return
        
        with Progress(console=self.console) as progress:
            task = progress.add_task("[green]Populating sandbox...", total=len(uploads))
            
            def upload_and_update(upload):
                self.sandbox.fs.upload_files([upload])
                progress.update(task, advance=1)

            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(upload_and_update, uploads)

    def _copy_directory(self, src: str, dst: str) -> None:
        """Copy directory contents from src to dst."""
        import shutil
        try:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            self.console.print(f"[green]Successfully copied files from {src} to {dst}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to copy directory: {e}[/red]")
            logger.exception("Failed to copy directory")

    def _normalize_path(self, path: str) -> str:
        """Convert to Daytona absolute path format."""
        posix_path = path.replace("\\", "/").replace("workspace/", "")
        if not posix_path.startswith("/"):
            posix_path = f"/home/daytona/workspace/{posix_path}"
        elif not posix_path.startswith("/home/daytona/workspace/"):
            posix_path = f"/home/daytona/workspace/{posix_path.lstrip('/')}"
        return posix_path
    
    def get_file_hash(self, path: str) -> Optional[str]:
        """Get the MD5 hash of a file in the sandbox."""
        self._ensure_sandbox()
        norm_path = self._normalize_path(path)
        # The command returns "HASH  FILENAME", we just want the hash
        cmd = f"md5sum {norm_path}"
        try:
            result = self.execute_command(cmd, timeout=10)
            if "No such file" in result or not result:
                return None
            return result.split()[0]
        except Exception:
            return None

    def read_file(self, path: str) -> bytes:
        self._ensure_sandbox()
        norm_path = self._normalize_path(path)
        try:
            content = self.sandbox.fs.download_file(norm_path)
            if content is None: raise RuntimeError(f"File {norm_path} not found or empty")
            return content if isinstance(content, bytes) else str(content).encode('utf-8')
        except Exception as e:
            logger.exception("Failed to read file %s", norm_path)
            raise

    def upload_file(self, path: str, content: Union[str, bytes]) -> None:
        self._ensure_sandbox()
        norm_path = self._normalize_path(path)
        try:
            parent_dir = os.path.dirname(norm_path)
            if parent_dir and parent_dir != "/home/daytona/workspace":
                try: self.sandbox.fs.create_folder(parent_dir, "755")
                except: pass

            content_bytes = content if isinstance(content, bytes) else str(content).encode('utf-8')
            self.sandbox.fs.upload_file(content_bytes, norm_path)
        except Exception as e:
            logger.exception("Failed to upload file %s", norm_path)
            raise

    def execute_command(self, cmd: str, timeout: int = 60) -> str:
        """Execute a command within sandbox."""
        if self.client is None:
            return "Daytona client not available"
        self._ensure_sandbox()
        try:
            response = self.sandbox.process.exec(cmd, timeout=timeout)
            stdout = getattr(response, 'stdout', b'').decode(errors="replace")
            stderr = getattr(response, 'stderr', b'').decode(errors="replace")
            return (stdout + stderr).strip() if (stdout + stderr).strip() else "Command executed successfully"
        except Exception as e:
            logger.exception("Command failed: %s", cmd)
            return f"Command failed: {e}"

    def delete_file(self, path: str) -> None:
        self._ensure_sandbox()
        norm_path = self._normalize_path(path)
        try:
            self.sandbox.fs.delete_file(norm_path)
        except Exception:
            logger.exception("Failed to delete file %s", norm_path)
            raise

    def destroy_sandbox(self) -> None:
        if not self.sandbox: return
        try:
            self.execute_command("pkill -f 'npm run dev' || true")
            self.sandbox.delete()
            self.console.print("[yellow]Sandbox destroyed[/yellow]")
        except Exception:
            logger.exception("Error destroying sandbox")

    def _ensure_sandbox(self):
        if self.client is None:
            raise RuntimeError("Daytona client not available")
        if not self.sandbox:
            raise RuntimeError("Sandbox is not initialized")
            
    def wait_for_server(self, timeout: int = 120) -> bool:
        """Poll the dev server log for a ready signal."""
        log_file = "/tmp/dev-server.log"
        start_time = time.time()
        self.console.print(f"[cyan]Waiting for dev server (max {timeout}s)...[/cyan]")
        
        # Common ready signals for development servers
        ready_signals = [
            "ready", "server running", "local:", "localhost:", 
            "http://", "compiled successfully", "running at",
            "dev server running", "listening on", "started server"
        ]
        
        with self.console.status("...") as status:
            while time.time() - start_time < timeout:
                try:
                    logs = self.execute_command(f"tail -n 20 {log_file}")
                    if logs:
                        status.update(f"...\n[grey50]{logs[-200:]}[/grey50]")  # Show last 200 chars
                        for signal in ready_signals:
                            if signal.lower() in logs.lower():
                                self.console.print("[green]Dev server is ready![/green]")
                                return True
                except Exception:
                    pass
                time.sleep(2)
        self.console.print(f"[yellow]Timeout waiting for dev server after {timeout}s.[/yellow]")
        return False
