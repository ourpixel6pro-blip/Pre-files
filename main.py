import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Google GenAI library not found. Please run 'pip install google-generativeai'")
    sys.exit(1)

from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import Config
from daytona_manager import DaytonaManager
from file_manager import LocalFileManager
from tool_executor import ToolExecutor, create_tool_functions, load_tools_config
from utils import get_console, get_logger

logger = get_logger("main")
console = get_console()

class LovableAgent:
    def __init__(self):
        self.config = None
        self.dm = None
        self.lfm = None
        self.executor = None
        self.chat = None
        self.history_path = "history.json"

    def setup(self):
        console.print(Panel("[bold cyan]Enhanced Lovable Agent[/bold cyan]"))
        try:
            self.config = Config.load()
        except Exception as e:
            console.print(Panel(f"[red]Configuration error: {e}[/red]"))
            sys.exit(1)

        self.lfm = LocalFileManager(console, self.config.workspace_dir)
        self.dm = DaytonaManager(self.config, console)
        self.executor = ToolExecutor(self.dm, self.lfm, console, self.config)

        try:
            self.dm.create_sandbox()
            self._setup_dev_environment()
        except Exception as e:
            console.print(Panel(f"[yellow]Warning: Could not create Daytona sandbox: {e}[/yellow]"))
            console.print("[yellow]Continuing in local-only mode. Some features may be limited.[/yellow]")
        
        self._initialize_chat()

    def _initialize_chat(self):
        system_prompt = self._load_prompt("prompt.txt")
        tools_config = load_tools_config()
        tool_functions = create_tool_functions(tools_config)

        try:
            client = genai.Client(api_key=self.config.gemini_api_key)
            model_name = self.config.models.get('primary', 'gemini-1.5-pro-latest')
            console.print(f"[cyan]Initializing chat with {model_name}...[/cyan]")
            
            chat_history = self._load_history()

            # Create chat config with tools if available
            if tool_functions:
                chat_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=tool_functions
                )
                console.print(f"[cyan]Loaded {len(tool_functions)} tools[/cyan]")
            else:
                chat_config = types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
                console.print("[yellow]No tools loaded - running in chat-only mode[/yellow]")

            self.chat = client.chats.create(
                model=model_name,
                history=chat_history,
                config=chat_config
            )
            console.print("[green]Chat initialized successfully.[/green]")
        except Exception as e:
            console.print(Panel(f"[red]Failed to initialize chat: {e}[/red]"))
            # Try to initialize without tools as fallback
            try:
                console.print("[yellow]Attempting to initialize without tools...[/yellow]")
                self.chat = client.chats.create(
                    model=model_name,
                    history=chat_history,
                    config=types.GenerateContentConfig(system_instruction=system_prompt)
                )
                console.print("[green]Chat initialized successfully (without tools).[/green]")
            except Exception as fallback_error:
                console.print(Panel(f"[red]Failed to initialize chat even without tools: {fallback_error}[/red]"))
                sys.exit(1)

    def _setup_dev_environment(self):
        console.print(Panel("[bold green]Setting up development environment...[/bold green]"))
        try:
            # Check if package.json exists in the sandbox
            package_check_cmd = "test -f /home/daytona/workspace/package.json && echo 'EXISTS' || echo 'NOT_FOUND'"
            package_exists = self.dm.execute_command(package_check_cmd).strip()
            
            if package_exists == "EXISTS":
                console.print("[cyan]Running npm install...")
                setup_cmd = "cd /home/daytona/workspace && npm install"
                install_logs = self.dm.execute_command(setup_cmd)
                console.print(Panel(install_logs, title="NPM Install Logs", border_style="green"))
                
                self.restart_dev_server()
            else:
                console.print("[yellow]No package.json found in sandbox, skipping npm install.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error during dev setup: {e}[/red]")

    def restart_dev_server(self):
        console.print("[cyan]Starting/Restarting dev server...[/cyan]")
        restart_cmd = f"cd /home/daytona/workspace && pkill -f 'npm run dev' || true && npm run dev -- --port {self.config.preview_port} > /tmp/dev-server.log 2>&1 &"
        try:
            self.dm.execute_command(restart_cmd)
            # Wait briefly for server startup, but don't block for too long
            if self.dm.wait_for_server(timeout=10):
                console.print("[green]Dev server started successfully![/green]")
            else:
                console.print("[yellow]Dev server is starting in background. Check logs if needed.[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to restart dev server: {e}[/red]")

    def run(self):
        self.setup()
        console.print("\n[bold green]Setup complete! You can now interact with the chat.[/bold green]")
        try:
            while True:
                user_input = Prompt.ask("[bold cyan]You")
                if user_input.strip().lower() in {"exit", "quit"}:
                    break
                
                with console.status("[magenta]Lovable is thinking..."):
                    self.executor.reset_changes()
                    try:
                        response = self.chat.send_message(user_input)
                    except Exception as e:
                        if "getaddrinfo failed" in str(e) or "ConnectError" in str(e):
                            console.print("[red]Network connection error. Please check your internet connection.[/red]")
                            console.print("[yellow]The application will continue in local-only mode.[/yellow]")
                            continue
                        else:
                            raise

                if hasattr(response, 'function_calls') and response.function_calls:
                    results = self.executor.execute_function_calls(response.function_calls)
                    self._sync_changes()
                    
                    # Show execution metrics if in debug mode
                    if self.config.debug_mode:
                        metrics = self.executor.get_metrics()
                        console.print(f"[dim]Tools executed: {metrics['tools_executed']}, "
                                    f"Files processed: {metrics['files_processed']}, "
                                    f"Errors: {metrics['errors']}, "
                                    f"Cache hits: {metrics['cache_hits']}[/dim]")

                if hasattr(response, 'text') and response.text:
                    console.print(Panel(response.text, title="Lovable", border_style="blue"))
                elif hasattr(response, 'function_calls') and response.function_calls:
                    # If there's only function calls without text, show a brief message
                    console.print(Panel("Tools executed successfully.", title="Lovable", border_style="blue"))

        except KeyboardInterrupt:
            console.print("\n[yellow]Chat interrupted by user.[/yellow]")
        finally:
            self._cleanup()

    def _sync_changes(self):
        if not self.executor.changed_files and not self.executor.deleted_files:
            return

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            sync_task = progress.add_task("Syncing changes to sandbox...", total=None)
            
            # Deletions
            for file_path in self.executor.deleted_files:
                try:
                    self.dm.delete_file(file_path)
                except Exception as e:
                    console.print(f"[red]Failed to delete {file_path} from sandbox: {e}[/red]")
            
            # Uploads with differential check
            files_to_upload = []
            for file_path in self.executor.changed_files:
                local_hash = self.lfm.get_file_hash(file_path)
                sandbox_hash = self.dm.get_file_hash(file_path)
                if local_hash != sandbox_hash:
                    files_to_upload.append(file_path)
            
            if files_to_upload:
                for file_path in files_to_upload:
                    try:
                        content = self.lfm.read_file(file_path)
                        self.dm.upload_file(file_path, content)
                    except Exception as e:
                        console.print(f"[red]Failed to upload {file_path}: {e}[/red]")

            progress.update(sync_task, completed=True, description="[green]Sync complete.[/green]")

        # Intelligent restart
        if self.config.intelligent_restart['enabled']:
            skip_ext = self.config.intelligent_restart['skip_extensions']
            should_restart = any(not file_path.endswith(tuple(skip_ext)) for file_path in self.executor.changed_files)
            if should_restart:
                self.restart_dev_server()
            else:
                console.print("[yellow]Skipping server restart based on file types.[/yellow]")
        else:
            self.restart_dev_server()

    def _load_prompt(self, path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return "You are a helpful AI assistant."

    def _load_history(self) -> List[Dict]:
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        # Empty file, return empty history
                        return []
                    return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not load history (creating new): {e}")
            # Initialize empty history file
            self._save_empty_history()
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
        return []
        
    def _save_empty_history(self):
        """Initialize an empty history file."""
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            logger.warning(f"Could not initialize history file: {e}")

    def _save_history(self):
        try:
            if self.chat and hasattr(self.chat, 'history'):
                 with open(self.history_path, "w") as f:
                    # The Gemini history object needs to be converted to a serializable format.
                    # This is a simplification. The actual structure might be more complex.
                    serializable_history = [
                        {'role': msg.role, 'parts': [part.text for part in msg.parts]}
                        for msg in self.chat.history
                    ]
                    json.dump(serializable_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")
            
    def _cleanup(self):
        console.print("[yellow]Cleaning up...[/yellow]")
        self._save_history()
        if self.dm:
            self.dm.destroy_sandbox()
        console.print("[bold green]Goodbye![/bold green]")

if __name__ == "__main__":
    agent = LovableAgent()
    agent.run()
