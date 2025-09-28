import requests
import json
import time
import base64
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Any, Callable, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import track
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from config import Config
from daytona_manager import DaytonaManager
from file_manager import LocalFileManager
from utils import get_logger, retry_with_backoff, measure_time, SimpleCache, validate_file_extension

logger = get_logger(__name__)

class ToolExecutor:
    """Enhanced tool executor with validation, caching, and advanced features."""

    def __init__(self, dm: DaytonaManager, lfm: LocalFileManager, console: Console, config: Config):
        self.dm = dm
        self.lfm = lfm
        self.console = console
        self.config = config
        self.changed_files = set()
        self.deleted_files = set()
        self.execution_cache = SimpleCache(default_ttl=600)  # 10-minute cache
        self.metrics = {
            'tools_executed': 0,
            'files_processed': 0,
            'execution_times': {},
            'errors': 0,
            'cache_hits': 0
        }
        
        # Tool validation rules
        self.tool_validators = {
            'lov_write': self._validate_write_args,
            'lov_line_replace': self._validate_line_replace_args,
            'lov_search_files': self._validate_search_args,
            'generate_image': self._validate_image_args,
            'edit_image': self._validate_edit_image_args,
            'web_search': self._validate_web_search_args,
            'lov_screenshot_website': self._validate_screenshot_args,
            'lov_crawl_website': self._validate_url_args,
            'lov_map_website': self._validate_url_args,
            'lov_extract_data': self._validate_extract_args,
            'lov_batch_crawl': self._validate_batch_crawl_args
        }

    def reset_changes(self):
        """Reset the tracked file changes and clean up cache."""
        self.changed_files.clear()
        self.deleted_files.clear()
        self.execution_cache.cleanup_expired()
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for monitoring."""
        return self.metrics.copy()
        
    def _validate_write_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments for write operations."""
        file_path = args.get('file_path')
        content = args.get('content', '')
        
        if not file_path:
            return "file_path is required"
        if not validate_file_extension(file_path, self.config.security.allowed_file_extensions):
            return f"File extension not allowed: {Path(file_path).suffix}"
        if len(content.encode('utf-8')) > self.config.security.max_file_size_mb * 1024 * 1024:
            return f"Content too large (max: {self.config.security.max_file_size_mb}MB)"
        return None
        
    def _validate_line_replace_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments for line replace operations."""
        try:
            first_line = int(args.get('first_replaced_line', 0))
            last_line = int(args.get('last_replaced_line', 0))
            if first_line < 1 or last_line < 1 or first_line > last_line:
                return "Invalid line numbers"
        except (ValueError, TypeError):
            return "Line numbers must be integers"
        return None
        
    def _validate_search_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments for search operations."""
        query = args.get('query')
        if not query or len(query.strip()) < 2:
            return "Search query must be at least 2 characters"
        return None
        
    def _validate_image_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments for image generation."""
        prompt = args.get('prompt')
        if not prompt or len(prompt.strip()) < 5:
            return "Image prompt must be at least 5 characters"
        return None
        
    def _validate_edit_image_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments for image editing."""
        source_image = args.get('source_image')
        prompt = args.get('prompt')
        
        if not source_image:
            return "Source image path is required for image editing"
        if not prompt or len(prompt.strip()) < 5:
            return "Edit prompt must be at least 5 characters"
        return None
        
    def _validate_url_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments for URL-based tools."""
        url = args.get('url')
        if not url or not url.strip():
            return "URL is required"
        if not url.startswith(('http://', 'https://')):
            return "URL must start with http:// or https://"
        return None
        
    def _validate_extract_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments for data extraction."""
        url = args.get('url')
        if not url or not url.strip():
            return "URL is required for data extraction"
        if not url.startswith(('http://', 'https://')):
            return "URL must start with http:// or https://"
        return None
        
    def _validate_batch_crawl_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments for batch crawling."""
        urls = args.get('urls', [])
        if not urls or not isinstance(urls, list):
            return "URLs list is required for batch crawling"
        if len(urls) == 0:
            return "At least one URL is required for batch crawling"
        for url in urls:
            if not url.startswith(('http://', 'https://')):
                return f"Invalid URL format: {url}"
        return None
        
    def _validate_web_search_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments for web search."""
        query = args.get('query')
        if not query or len(query.strip()) < 2:
            return "Search query must be at least 2 characters"
        
        num_results = args.get('numResults', 5)
        if not isinstance(num_results, int) or num_results < 1 or num_results > 20:
            return "numResults must be an integer between 1 and 20"
        
        return None
        
    def _validate_screenshot_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments for screenshot operations."""
        url = args.get('url')
        if not url or not url.strip():
            return "URL is required"
        
        # Basic URL validation
        if not (url.startswith('http://') or url.startswith('https://')):
            return "URL must start with http:// or https://"
        
        # Validate viewport dimensions if provided
        viewport = args.get('viewport', {})
        if viewport:
            width = viewport.get('width', 1920)
            height = viewport.get('height', 1080)
            if not (100 <= width <= 4000) or not (100 <= height <= 4000):
                return "Viewport dimensions must be between 100 and 4000 pixels"
        
        return None

    @measure_time
    @retry_with_backoff(max_retries=2)
    def lov_view(self, args):
        path = args.get("file_path")
        
        # Check cache first
        cache_key = f"view:{path}"
        cached = self.execution_cache.get(cache_key)
        if cached:
            self.metrics['cache_hits'] += 1
            return cached
            
        try:
            content = self.lfm.read_file(path)
            result = f"[Local]\n{content.decode('utf-8', errors='replace')}"
        except Exception:
            try:
                content = self.dm.read_file(path)
                result = f"[Sandbox]\n{content.decode('utf-8', errors='replace')}"
            except Exception as e:
                self.metrics['errors'] += 1
                return f"Error reading file {path}: {e}"
        
        # Cache the result
        self.execution_cache.set(cache_key, result, ttl=300)
        return result

    @measure_time
    def lov_write(self, args):
        # Validate arguments
        validation_error = self._validate_write_args(args)
        if validation_error:
            self.metrics['errors'] += 1
            return f"Validation error: {validation_error}"
            
        path = args.get("file_path")
        content = args.get("content", "")
        
        try:
            self.lfm.write_file(path, content)
            self.changed_files.add(path)
            self.metrics['files_processed'] += 1
            
            # Invalidate cache for this file
            self.execution_cache.set(f"view:{path}", None, ttl=0)
            
            return f"Successfully wrote to local file {path}"
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Failed to write file {path}: {e}")
            return f"Error writing file {path}: {e}"

    def lov_line_replace(self, args):
        file_path = args.get("file_path")
        try:
            first_line = int(args.get("first_replaced_line", 1))
            last_line = int(args.get("last_replaced_line", 1))
        except (ValueError, TypeError):
            return "Error: Line numbers must be integers."

        replace = args.get("replace", "")
        
        try:
            content = self.lfm.read_file(file_path).decode('utf-8', 'replace')
            lines = content.splitlines()
            if not (0 < first_line <= last_line <= len(lines)):
                return f"Error: Invalid line range for file with {len(lines)} lines."

            new_lines = lines[:first_line - 1] + replace.splitlines() + lines[last_line:]
            self.lfm.write_file(file_path, "\n".join(new_lines))
            self.changed_files.add(file_path)
            return f"Replaced lines in local file {file_path}"
        except Exception as e:
            return f"Error during line replace: {e}"

    def lov_search_files(self, args):
        query = args.get("query")
        # Local search
        local_results = self.lfm.search_content(
            query, args.get("include_pattern"), args.get("exclude_pattern"), args.get("case_sensitive", False)
        )
        # Sandbox search
        sandbox_results = self.dm.execute_command(
            f"grep -r {'-i' if not args.get('case_sensitive') else ''} '{query}' /home/daytona/workspace"
        ).splitlines()

        table = Table(title="File Search Results")
        table.add_column("Location", style="cyan")
        table.add_column("Path", style="magenta")
        
        for r in local_results: table.add_row("Local", r)
        for r in sandbox_results: table.add_row("Sandbox", r)
            
        self.console.print(table)
        return "Search results displayed above."

    def lov_rename(self, args):
        original = args.get("original_file_path")
        new = args.get("new_file_path")
        self.lfm.move_file(original, new)
        self.deleted_files.add(original)
        self.changed_files.add(new)
        return f"Renamed local file from {original} to {new}"

    def lov_delete(self, args):
        file_path = args.get("file_path")
        self.lfm.delete_file(file_path)
        self.deleted_files.add(file_path)
        if file_path in self.changed_files:
            self.changed_files.remove(file_path)
        return f"Deleted local file {file_path}"

    def lov_add_dependency(self, args):
        package = args.get("package")
        # This will be synced and installed when the server restarts
        try:
            package_json_path = "workspace/package.json"
            content = self.lfm.read_file(package_json_path).decode()
            data = json.loads(content)
            data['dependencies'][package] = "latest" # simplistic
            self.lfm.write_file(package_json_path, json.dumps(data, indent=2))
            self.changed_files.add(package_json_path)
            return f"Added {package} to local package.json. It will be installed on next sync."
        except Exception as e:
            return f"Error adding dependency: {e}"


    def lov_download_to_repo(self, args):
        source_url = args.get("source_url")
        target_path = args.get("target_path")
        try:
            response = requests.get(source_url, timeout=30)
            response.raise_for_status()
            self.lfm.write_file(target_path, response.content)
            self.changed_files.add(target_path)
            return f"Downloaded to local {target_path}"
        except Exception as e:
            return f"Error downloading {source_url}: {e}"
            
    def lov_read_console_logs(self, args):
        """Read recent logs from the dev server in the sandbox."""
        return self.dm.execute_command("tail -n 50 /tmp/dev-server.log")

    @measure_time
    @retry_with_backoff(max_retries=2)
    def lov_fetch_website(self, args):
        """Enhanced website fetching with Firecrawl screenshot and scraping capabilities."""
        url = args.get("url")
        formats = args.get("formats", "markdown").split(",")
        
        if not url:
            return "Error: URL is required"
            
        # Clean and validate formats
        valid_formats = ['markdown', 'html', 'screenshot']
        formats = [f.strip().lower() for f in formats if f.strip().lower() in valid_formats]
        
        if not formats:
            formats = ['markdown']  # Default fallback
            
        results = {}
        
        # If Firecrawl is available and configured, use it for enhanced scraping
        if self.config.firecrawl_api_key and "YOUR_" not in self.config.firecrawl_api_key:
            try:
                results = self._firecrawl_scrape_website(url, formats)
                if results:
                    return self._format_fetch_results(url, results, formats)
            except Exception as e:
                logger.warning(f"Firecrawl failed, falling back to basic scraping: {e}")
        
        # Fallback to basic scraping for non-screenshot formats
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if 'html' in formats:
                results['html'] = response.text
                
            if 'markdown' in formats:
                soup = BeautifulSoup(response.text, 'html.parser')
                results['markdown'] = soup.get_text(separator='\n', strip=True)
                
            if 'screenshot' in formats:
                results['screenshot'] = "Screenshot not available (Firecrawl API key not configured)"
                
            return self._format_fetch_results(url, results, formats)
            
        except Exception as e:
            return f"Error fetching {url}: {e}"
    
    def _firecrawl_scrape_website(self, url: str, formats: List[str]) -> Dict[str, str]:
        """Use Firecrawl API to scrape website with enhanced capabilities."""
        scrape_url = "https://api.firecrawl.dev/v1/scrape"
        headers = {
            "Authorization": f"Bearer {self.config.firecrawl_api_key}",
            "Content-Type": "application/json"
        }
        
        # Configure scraping options based on requested formats
        page_options = {
            "onlyMainContent": True,
            "includeHtml": 'html' in formats,
            "includeRawHtml": False,
            "screenshot": 'screenshot' in formats,
            "fullPageScreenshot": 'screenshot' in formats,
            "waitFor": 2000  # Wait 2 seconds for page to load
        }
        
        payload = {
            "url": url,
            "pageOptions": page_options,
            "extractorOptions": {
                "mode": "markdown" if 'markdown' in formats else "text"
            }
        }
        
        logger.info(f"Scraping website with Firecrawl: {url}")
        response = requests.post(scrape_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("success"):
            raise Exception(f"Firecrawl scraping failed: {data.get('error', 'Unknown error')}")
        
        result_data = data.get("data", {})
        results = {}
        
        # Extract different formats
        if 'markdown' in formats and 'markdown' in result_data:
            results['markdown'] = result_data['markdown']
        elif 'markdown' in formats and 'content' in result_data:
            results['markdown'] = result_data['content']
            
        if 'html' in formats and 'html' in result_data:
            results['html'] = result_data['html']
            
        if 'screenshot' in formats and 'screenshot' in result_data:
            # Save screenshot to local file
            screenshot_data = result_data['screenshot']
            if screenshot_data:
                # Handle base64 encoded screenshot
                if screenshot_data.startswith('data:image'):
                    # Extract base64 data
                    header, data = screenshot_data.split(',', 1)
                    screenshot_bytes = base64.b64decode(data)
                else:
                    screenshot_bytes = base64.b64decode(screenshot_data)
                
                # Generate filename
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.replace('.', '_')
                screenshot_filename = f"screenshots/{domain}_{int(time.time())}.png"
                
                # Save screenshot locally
                try:
                    self.lfm.write_file(screenshot_filename, screenshot_bytes)
                    self.changed_files.add(screenshot_filename)
                    results['screenshot'] = f"Screenshot saved to: {screenshot_filename}"
                except Exception as e:
                    results['screenshot'] = f"Screenshot captured but failed to save: {e}"
        
        return results
    
    def _format_fetch_results(self, url: str, results: Dict[str, str], formats: List[str]) -> str:
        """Format the fetching results for display."""
        output = [f"🌐 Website Content from: {url}"]
        output.append("=" * 60)
        
        for format_type in formats:
            if format_type in results:
                content = results[format_type]
                if format_type == 'markdown':
                    output.append(f"\n📄 **Markdown Content:**")
                    # Truncate very long content
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (content truncated)"
                    output.append(content)
                elif format_type == 'html':
                    output.append(f"\n🏷️ **HTML Content:**")
                    # Show only first part of HTML
                    if len(content) > 1000:
                        content = content[:1000] + "\n... (HTML truncated)"
                    output.append(content)
                elif format_type == 'screenshot':
                    output.append(f"\n📸 **Screenshot:**")
                    output.append(content)
        
        return "\n".join(output)

    def generate_image(self, args):
        """Generate images using Gemini 2.5 Flash with Imagen model."""
        prompt = args.get("prompt")
        target_path = args.get("target_path", "generated_image.png")
        num_images = args.get("num_images", 1)
        aspect_ratio = args.get("aspect_ratio", "1:1")  # Default square
        safety_level = args.get("safety_level", "BLOCK_MEDIUM_AND_ABOVE")
        
        try:
            import google.genai as genai
            from google.genai import types
            
            # Initialize client with API key from config
            client = genai.Client(api_key=self.config.gemini_api_key)
            
            # Use Gemini 2.5 Flash Image for conversational editing and speed
            model = "gemini-2.5-flash-image-preview"
            
            # Generate images using the correct Gemini 2.5 Flash Image API
            logger.info(f"Generating image with prompt: {prompt}")
            logger.info(f"Target path: {target_path}")
            
            response = client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    temperature=0.7
                )
            )
            
            logger.info(f"Response received: {response}")
            
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning("No candidates or parts in response")
                return "No images were generated. Please try a different prompt."
            
            # Save images locally
            saved_files = []
            for i, part in enumerate(response.candidates[0].content.parts):
                logger.info(f"Processing part {i}: {part}")
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Create filename for multiple images
                    if len(response.candidates[0].content.parts) > 1:
                        file_path = target_path.replace('.png', f'_{i+1}.png')
                    else:
                        file_path = target_path
                    
                    logger.info(f"Saving image to: {file_path}")
                    
                    # Write image bytes to file
                    self.lfm.write_file(file_path, part.inline_data.data, mode='wb')
                    self.changed_files.add(file_path)
                    saved_files.append(file_path)
                    logger.info(f"Successfully saved image to: {file_path}")
                else:
                    logger.warning(f"Part {i} does not have inline_data: {part}")
            
            if saved_files:
                result = f"Generated {len(saved_files)} image(s) using Gemini Imagen:\n"
                for file_path in saved_files:
                    result += f"  • {file_path}\n"
                return result.strip()
            else:
                return "Images were generated but could not be saved to files."
                
        except ImportError:
            return "Google GenAI library is not available. Please install it with: pip install google-genai"
        except Exception as e:
            return f"Failed to generate image with Gemini: {e}"

    def edit_image(self, args):
        """Edit images using Gemini 2.5 Flash with Imagen model."""
        source_image = args.get("source_image")
        prompt = args.get("prompt")
        target_path = args.get("target_path", "edited_image.png")
        edit_mode = args.get("edit_mode", "EDIT_MODE_INPAINT_INSERTION")
        mask_mode = args.get("mask_mode", "MASK_MODE_BACKGROUND")
        
        if not source_image:
            return "Source image path is required for image editing."
        
        try:
            import google.genai as genai
            from google.genai import types
            
            # Initialize client with API key from config
            client = genai.Client(api_key=self.config.gemini_api_key)
            
            # Use Gemini 2.5 Flash Image for conversational editing
            model = "gemini-2.5-flash-image-preview"
            
            # Load the source image
            try:
                from PIL import Image
                original_image = Image.open(source_image)
                
            except Exception as e:
                return f"Failed to load source image from {source_image}: {e}"
            
            # Edit image using the correct Gemini 2.5 Flash Image API
            response = client.models.generate_content(
                model=model,
                contents=[original_image, prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    temperature=0.7
                )
            )
            
            if not response.candidates or not response.candidates[0].content.parts:
                return "No edited images were generated. Please try a different prompt or edit mode."
            
            # Save the edited image
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    self.lfm.write_file(target_path, part.inline_data.data, mode='wb')
                    self.changed_files.add(target_path)
                    return f"Successfully edited image using Gemini 2.5 Flash Image and saved to: {target_path}"
            
            return "Image was edited but could not be saved to file."
                
        except ImportError:
            return "Google GenAI library is not available. Please install it with: pip install google-genai"
        except Exception as e:
            return f"Failed to edit image with Gemini: {e}"

    def lov_crawl_website(self, args):
        """Crawl a website using Firecrawl API."""
        url = args.get('url')
        if not url:
            return "URL is required for website crawling"
        
        try:
            response = requests.post(
                f"{self.config.firecrawl_base_url}/crawl",
                headers={"Authorization": f"Bearer {self.config.firecrawl_api_key}"},
                json={"url": url, "formats": ["markdown", "html"]}
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success') and data.get('data'):
                crawl_data = data['data']
                return f"Successfully crawled {url}. Found {len(crawl_data.get('data', []))} pages."
            else:
                return f"Failed to crawl {url}: {data.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"Error crawling website: {e}"

    def lov_map_website(self, args):
        """Map a website structure using Firecrawl API."""
        url = args.get('url')
        if not url:
            return "URL is required for website mapping"
        
        try:
            response = requests.post(
                f"{self.config.firecrawl_base_url}/map",
                headers={"Authorization": f"Bearer {self.config.firecrawl_api_key}"},
                json={"url": url}
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success') and data.get('data'):
                map_data = data['data']
                return f"Successfully mapped {url}. Found {len(map_data.get('links', []))} links."
            else:
                return f"Failed to map {url}: {data.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"Error mapping website: {e}"

    def lov_extract_data(self, args):
        """Extract structured data from a website using Firecrawl API."""
        url = args.get('url')
        schema = args.get('schema', {})
        
        if not url:
            return "URL is required for data extraction"
        
        try:
            response = requests.post(
                f"{self.config.firecrawl_base_url}/extract",
                headers={"Authorization": f"Bearer {self.config.firecrawl_api_key}"},
                json={"url": url, "schema": schema}
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success') and data.get('data'):
                extracted_data = data['data']
                return f"Successfully extracted data from {url}: {extracted_data}"
            else:
                return f"Failed to extract data from {url}: {data.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"Error extracting data: {e}"

    def lov_batch_crawl(self, args):
        """Batch crawl multiple URLs using Firecrawl API."""
        urls = args.get('urls', [])
        if not urls:
            return "URLs list is required for batch crawling"
        
        try:
            response = requests.post(
                f"{self.config.firecrawl_base_url}/batch",
                headers={"Authorization": f"Bearer {self.config.firecrawl_api_key}"},
                json={"urls": urls, "formats": ["markdown", "html"]}
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success') and data.get('data'):
                batch_data = data['data']
                return f"Successfully batch crawled {len(urls)} URLs. Job ID: {batch_data.get('jobId', 'Unknown')}"
            else:
                return f"Failed to batch crawl URLs: {data.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"Error in batch crawling: {e}"

    @measure_time
    @retry_with_backoff(max_retries=2)
    def web_search(self, args):
        """Perform web search using Firecrawl API."""
        if not self.config.firecrawl_api_key:
            return "Firecrawl API key is not configured."
        
        query = args.get("query", "")
        num_results = args.get("numResults", 5)
        category = args.get("category")
        links = args.get("links", 0)
        image_links = args.get("imageLinks", 0)
        
        if not query.strip():
            return "Error: Search query is required"
        
        try:
            # Firecrawl search endpoint
            url = "https://api.firecrawl.dev/v1/search"
            headers = {
                "Authorization": f"Bearer {self.config.firecrawl_api_key}",
                "Content-Type": "application/json"
            }
            
            # Build search parameters
            search_params = {
                "query": query,
                "pageOptions": {
                    "onlyMainContent": True,
                    "includeHtml": False,
                    "includeRawHtml": False
                },
                "searchOptions": {
                    "limit": num_results
                }
            }
            
            # Add category-specific search parameters
            if category:
                search_params["searchOptions"]["category"] = category
            
            logger.info(f"Performing web search for: {query}")
            response = requests.post(url, headers=headers, json=search_params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("success"):
                return f"Search failed: {data.get('error', 'Unknown error')}"
            
            search_results = data.get("data", [])
            
            if not search_results:
                return f"No results found for query: {query}"
            
            # Format results
            formatted_results = []
            formatted_results.append(f"🔍 Search Results for: '{query}' ({len(search_results)} results)")
            formatted_results.append("=" * 60)
            
            for i, result in enumerate(search_results, 1):
                title = result.get("metadata", {}).get("title", "No Title")
                url = result.get("metadata", {}).get("sourceURL", "No URL")
                content = result.get("markdown", "")
                
                # Truncate content for readability
                if len(content) > 300:
                    content = content[:300] + "..."
                
                formatted_results.append(f"\n{i}. {title}")
                formatted_results.append(f"   URL: {url}")
                formatted_results.append(f"   Content: {content}")
                
                # Add additional links if requested
                if links > 0:
                    formatted_results.append(f"   🔗 Link: {url}")
                
                # Add image links if requested
                if image_links > 0:
                    images = result.get("metadata", {}).get("images", [])
                    if images:
                        for j, img_url in enumerate(images[:image_links]):
                            formatted_results.append(f"   🖼️ Image {j+1}: {img_url}")
            
            # Cache the search results
            cache_key = f"search:{query}:{num_results}"
            self.execution_cache.set(cache_key, "\n".join(formatted_results), ttl=600)
            
            self.metrics['tools_executed'] += 1
            return "\n".join(formatted_results)
            
        except requests.exceptions.RequestException as e:
            self.metrics['errors'] += 1
            logger.error(f"Firecrawl API request failed: {e}")
            return f"Network error during search: {e}"
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Web search failed: {e}")
            return f"Error performing web search: {e}"

    @measure_time
    @retry_with_backoff(max_retries=2)
    def lov_screenshot_website(self, args):
        """Capture advanced screenshots with device emulation and custom options."""
        if not self.config.firecrawl_api_key or "YOUR_" in self.config.firecrawl_api_key:
            return "Firecrawl API key is not configured. Screenshot functionality requires Firecrawl."
        
        url = args.get("url", "")
        target_path = args.get("target_path", "")
        viewport = args.get("viewport", {"width": 1920, "height": 1080})
        device_type = args.get("device_type", "desktop")  # desktop, mobile, tablet
        full_page = args.get("full_page", True)
        wait_for = args.get("wait_for", 3000)  # Wait time in milliseconds
        
        if not url.strip():
            return "Error: URL is required"
        
        try:
            # Configure device-specific settings
            device_configs = {
                "desktop": {"width": 1920, "height": 1080, "mobile": False},
                "mobile": {"width": 375, "height": 667, "mobile": True},
                "tablet": {"width": 768, "height": 1024, "mobile": False}
            }
            
            device_config = device_configs.get(device_type, device_configs["desktop"])
            
            # Override with custom viewport if provided
            final_viewport = {
                "width": viewport.get("width", device_config["width"]),
                "height": viewport.get("height", device_config["height"])
            }
            
            # Firecrawl scrape endpoint with screenshot options
            scrape_url = "https://api.firecrawl.dev/v1/scrape"
            headers = {
                "Authorization": f"Bearer {self.config.firecrawl_api_key}",
                "Content-Type": "application/json"
            }
            
            # Advanced page options for screenshot
            page_options = {
                "screenshot": True,
                "fullPageScreenshot": full_page,
                "viewport": final_viewport,
                "mobile": device_config["mobile"],
                "waitFor": wait_for,
                "onlyMainContent": False,  # Get full page for screenshots
                "includeHtml": False,
                "includeRawHtml": False
            }
            
            payload = {
                "url": url,
                "pageOptions": page_options
            }
            
            logger.info(f"Taking screenshot of {url} with {device_type} viewport ({final_viewport['width']}x{final_viewport['height']})")
            response = requests.post(scrape_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("success"):
                return f"Screenshot failed: {data.get('error', 'Unknown error')}"
            
            result_data = data.get("data", {})
            screenshot_data = result_data.get("screenshot")
            
            if not screenshot_data:
                return "No screenshot data received from Firecrawl"
            
            # Handle base64 encoded screenshot
            if screenshot_data.startswith('data:image'):
                # Extract base64 data
                header, data_part = screenshot_data.split(',', 1)
                screenshot_bytes = base64.b64decode(data_part)
            else:
                screenshot_bytes = base64.b64decode(screenshot_data)
            
            # Generate filename if not provided
            if not target_path:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.replace('.', '_')
                timestamp = int(time.time())
                target_path = f"screenshots/{domain}_{device_type}_{timestamp}.png"
            
            # Ensure screenshots directory exists
            screenshots_dir = "screenshots"
            if not target_path.startswith(screenshots_dir + "/"):
                target_path = f"{screenshots_dir}/{target_path}"
            
            # Save screenshot locally
            try:
                self.lfm.write_file(target_path, screenshot_bytes)
                self.changed_files.add(target_path)
                
                # Get file size for reporting
                file_size = len(screenshot_bytes)
                size_kb = file_size / 1024
                
                result = [
                    f"📸 Screenshot captured successfully!",
                    f"URL: {url}",
                    f"Device: {device_type} ({final_viewport['width']}x{final_viewport['height']})",
                    f"Type: {'Full page' if full_page else 'Viewport only'}",
                    f"File: {target_path}",
                    f"Size: {size_kb:.1f} KB"
                ]
                
                self.metrics['tools_executed'] += 1
                return "\n".join(result)
                
            except Exception as e:
                self.metrics['errors'] += 1
                return f"Screenshot captured but failed to save to {target_path}: {e}"
                
        except requests.exceptions.RequestException as e:
            self.metrics['errors'] += 1
            logger.error(f"Firecrawl screenshot request failed: {e}")
            return f"Network error during screenshot: {e}"
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Screenshot failed: {e}")
            return f"Error taking screenshot: {e}"

    def _not_implemented(self, tool_name):
        return f"Tool '{tool_name}' is not implemented yet."
        
    @measure_time
    def lov_run_code(self, args):
        """Execute code in sandbox and capture artifacts (charts, outputs, etc.)."""
        code = args.get("code", "")
        language = args.get("language", "python")
        
        if not code.strip():
            return "Error: No code provided"
            
        try:
            # Execute code in sandbox
            if language.lower() == "python":
                # For Python, we can capture matplotlib charts
                enhanced_code = f"""
import sys
import traceback
try:
{code}
except Exception as e:
    print(f"Error: {{e}}")
    traceback.print_exc()
"""
                result = self.dm.execute_command(f"cd /home/daytona/workspace && python3 -c '{enhanced_code}'")
            else:
                # For other languages, execute directly
                result = self.dm.execute_command(f"cd /home/daytona/workspace && {code}")
            
            self.metrics['tools_executed'] += 1
            return f"Code execution result:\n{result}"
            
        except Exception as e:
            self.metrics['errors'] += 1
            return f"Error executing code: {e}"
    
    @measure_time
    def lov_batch_operations(self, args):
        """Perform multiple file operations in batch for efficiency."""
        operations = args.get("operations", [])
        
        if not operations:
            return "Error: No operations provided"
            
        results = []
        successful_ops = 0
        
        for i, op in enumerate(track(operations, description="Processing operations...")):
            try:
                op_type = op.get("type")
                op_args = op.get("args", {})
                
                if op_type == "write":
                    result = self.lov_write(op_args)
                elif op_type == "delete":
                    result = self.lov_delete(op_args)
                elif op_type == "rename":
                    result = self.lov_rename(op_args)
                else:
                    result = f"Unknown operation type: {op_type}"
                
                results.append(f"Operation {i+1}: {result}")
                if not result.startswith("Error"):
                    successful_ops += 1
                    
            except Exception as e:
                results.append(f"Operation {i+1} failed: {e}")
                self.metrics['errors'] += 1
        
        summary = f"Batch operations completed: {successful_ops}/{len(operations)} successful"
        return f"{summary}\n\n" + "\n".join(results)
    
    @measure_time  
    def lov_search_and_replace(self, args):
        """Advanced search and replace across multiple files."""
        search_pattern = args.get("search_pattern", "")
        replace_text = args.get("replace_text", "")
        file_pattern = args.get("file_pattern", "*")
        case_sensitive = args.get("case_sensitive", False)
        
        if not search_pattern:
            return "Error: Search pattern is required"
            
        try:
            # Use sandbox grep for powerful search and replace
            grep_flags = "" if case_sensitive else "-i"
            find_cmd = f"find /home/daytona/workspace -name '{file_pattern}' -type f"
            files_result = self.dm.execute_command(find_cmd)
            
            if "No such file" in files_result:
                return "No matching files found"
                
            files = [f.strip() for f in files_result.split('\n') if f.strip()]
            modified_files = []
            
            for file_path in files:
                # Check if file contains the pattern
                grep_cmd = f"grep {grep_flags} -l '{search_pattern}' '{file_path}'"
                grep_result = self.dm.execute_command(grep_cmd)
                
                if file_path in grep_result:
                    # Perform replacement
                    sed_flags = "i" if not case_sensitive else "I"
                    replace_cmd = f"sed -{sed_flags} 's/{search_pattern}/{replace_text}/g' '{file_path}'"
                    self.dm.execute_command(replace_cmd)
                    modified_files.append(file_path.replace("/home/daytona/workspace/", ""))
            
            if modified_files:
                # Mark files as changed for sync
                for file_path in modified_files:
                    self.changed_files.add(file_path)
                    
                return f"Successfully replaced '{search_pattern}' with '{replace_text}' in {len(modified_files)} files:\n" + "\n".join(modified_files)
            else:
                return f"Pattern '{search_pattern}' not found in any files"
                
        except Exception as e:
            self.metrics['errors'] += 1
            return f"Error during search and replace: {e}"
    
    @measure_time
    def lov_project_analytics(self, args):
        """Generate project analytics and insights."""
        try:
            # Get project statistics
            stats_cmd = """
cd /home/daytona/workspace && {
    echo "=== PROJECT ANALYTICS ==="
    echo "Files by type:"
    find . -type f | grep -E '\\.' | sed 's/.*\\.//' | sort | uniq -c | sort -nr | head -10
    echo ""
    echo "Total files:" $(find . -type f | wc -l)
    echo "Total directories:" $(find . -type d | wc -l)
    echo "Lines of code:"
    find . -name '*.js' -o -name '*.ts' -o -name '*.py' -o -name '*.go' | xargs wc -l | tail -1
    echo ""
    echo "Largest files:"
    find . -type f -exec ls -la {} \\; | sort -k5 -nr | head -5
    echo ""
    echo "Recent files:"
    find . -type f -mtime -1 | head -10
}
"""
            result = self.dm.execute_command(stats_cmd)
            return f"Project Analytics:\n{result}"
            
        except Exception as e:
            return f"Error generating analytics: {e}"

    def execute_function_calls(self, function_calls: List) -> List[Dict[str, Any]]:
        dispatch = {
            "lov-view": self.lov_view,
            "lov-write": self.lov_write,
            "lov-line-replace": self.lov_line_replace,
            "lov-search-files": self.lov_search_files,
            "lov-rename": self.lov_rename,
            "lov-delete": self.lov_delete,
            "lov-add-dependency": self.lov_add_dependency,
            "lov-download-to-repo": self.lov_download_to_repo,
            "lov-read-console-logs": self.lov_read_console_logs,
            "lov-fetch-website": self.lov_fetch_website,
            "generate_image": self.generate_image,
            # New advanced tools
            "lov-run-code": self.lov_run_code,
            "lov-batch-operations": self.lov_batch_operations,
            "lov-search-and-replace": self.lov_search_and_replace,
            "lov-project-analytics": self.lov_project_analytics,
            # Firecrawl tools
            "web_search": self.web_search,
            "lov-screenshot-website": self.lov_screenshot_website,
            "lov-crawl-website": self.lov_crawl_website,
            "lov-map-website": self.lov_map_website,
            "lov-extract-data": self.lov_extract_data,
            "lov-batch-crawl": self.lov_batch_crawl,
            # Placeholders for tools not yet implemented
            "lov-read-network-requests": lambda args: self._not_implemented("lov-read-network-requests"),
            "edit_image": self.edit_image,
        }
        
        results = []
        start_time = time.time()
        
        for call in function_calls:
            call_start = time.time()
            name = call.name.replace('_', '-')
            args = dict(call.args) if hasattr(call, 'args') else {}
            handler = dispatch.get(name)
            
            self.console.print(f"[yellow]Executing tool: {name}[/yellow]")
            
            # Pre-execution validation
            validator = self.tool_validators.get(name.replace('-', '_'))
            if validator:
                validation_error = validator(args)
                if validation_error:
                    self.metrics['errors'] += 1
                    output = f"Validation failed: {validation_error}"
                    results.append({"name": name, "output": output, "execution_time": 0})
                    continue
            
            if not handler:
                output = f"Unknown tool: {name}"
                self.metrics['errors'] += 1
            else:
                try:
                    output = handler(args)
                    self.metrics['tools_executed'] += 1
                except Exception as e:
                    logger.exception(f"Tool call failed: {name}")
                    output = f"Error executing tool {name}: {e}"
                    self.metrics['errors'] += 1
            
            call_time = time.time() - call_start
            self.metrics['execution_times'][name] = call_time
            
            results.append({
                "name": name, 
                "output": str(output),
                "execution_time": call_time
            })
        
        total_time = time.time() - start_time
        logger.info(f"Executed {len(function_calls)} tools in {total_time:.2f}s")
        return results

def create_tool_functions(tools_config: List[Dict]) -> List[Any]:
    """Create tool declarations for Gemini from tools.json config."""
    if not tools_config:
        return []
    
    try:
        import copy
        # Create a clean copy and patch for Gemini compatibility
        patched_tools_config = copy.deepcopy(tools_config)
        
        gemini_tools = []
        for tool in patched_tools_config:
            # Convert kebab-case to snake_case for function names
            if 'name' in tool:
                tool['name'] = tool['name'].replace('-', '_')
            
            # Remove 'example' fields from parameters as they're not supported
            if 'parameters' in tool and 'properties' in tool['parameters']:
                for param_name, param_props in tool['parameters']['properties'].items():
                    if isinstance(param_props, dict) and 'example' in param_props:
                        del param_props['example']
            
            # Format for Gemini API - wrap in function_declarations
            gemini_tool = types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool['name'],
                        description=tool.get('description', ''),
                        parameters=tool.get('parameters', {})
                    )
                ]
            )
            gemini_tools.append(gemini_tool)
        
        return gemini_tools
        
    except Exception as e:
        logger.error(f"Failed to process tools for Gemini: {e}")
        logger.debug(f"Tools config sample: {tools_config[:1] if tools_config else 'Empty'}")
        # Return empty list on error to allow chat to work without tools
        return []

def load_tools_config(path: str = "Tools.json") -> List[Dict]:
    """Load tool configurations from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"{path} not found, no tools will be loaded.")
        return []
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")
