import os
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml

from utils import get_logger

logger = get_logger(__name__)

@dataclass
class PerformanceConfig:
    """Performance-related configuration."""
    max_parallel_uploads: int = 10
    sync_batch_size: int = 50
    command_timeout: int = 60
    server_startup_timeout: int = 30
    enable_file_caching: bool = True
    cache_ttl_seconds: int = 300

@dataclass
class SecurityConfig:
    """Security-related configuration."""
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        '.js', '.ts', '.jsx', '.tsx', '.css', '.html', '.json', '.md', '.txt', 
        '.py', '.go', '.rs', '.java', '.cpp', '.c', '.h', '.sh', '.yml', '.yaml'
    ])
    max_file_size_mb: int = 10
    enable_path_validation: bool = True
    audit_logging: bool = True

@dataclass
class Config:
    """Main configuration class with validation and defaults."""
    gemini_api_key: str
    daytona_api_key: str
    stability_api_key: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    daytona_url: Optional[str] = None
    daytona_target: Optional[str] = None
    preview_port: int = 3000
    workspace_dir: str = "workspace_local"
    models: Dict[str, str] = field(default_factory=lambda: {
        'primary': 'gemini-2.5-pro',
        'fallback': 'gemini-2.5-flash'
    })
    intelligent_restart: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'skip_extensions': ['.css', '.md', '.txt', '.jpg', '.jpeg', '.png', '.svg']
    })
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    debug_mode: bool = False
    backup_enabled: bool = True
    auto_save_history: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_api_keys()
        self._validate_ports()
        self._validate_models()
        
    def _validate_api_keys(self):
        """Validate API keys are not placeholders."""
        if not self.gemini_api_key or "YOUR_" in self.gemini_api_key:
            raise ValueError("Invalid or missing Gemini API key")
        if not self.daytona_api_key or "YOUR_" in self.daytona_api_key:
            raise ValueError("Invalid or missing Daytona API key")
        if self.stability_api_key and "YOUR_" in self.stability_api_key:
            logger.warning("Stability API key appears to be a placeholder")
        if self.firecrawl_api_key and "YOUR_" in self.firecrawl_api_key:
            logger.warning("Firecrawl API key appears to be a placeholder")
            
    def _validate_ports(self):
        """Validate port numbers."""
        if not (1000 <= self.preview_port <= 65535):
            raise ValueError(f"Invalid preview port: {self.preview_port}")
            
    def _validate_models(self):
        """Validate model configuration."""
        if not self.models.get('primary'):
            self.models['primary'] = 'gemini-2.5-pro'
        if not self.models.get('fallback'):
            self.models['fallback'] = 'gemini-2.5-flash'

    @staticmethod
    def load(path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file with comprehensive error handling."""
        config_data = {}
        
        # Load from file if it exists
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {path}")
            except Exception as e:
                logger.error(f"Error loading configuration file: {e}")
                raise RuntimeError(f"Failed to load {path}: {e}")
        else:
            logger.warning(f"Configuration file {path} not found, using defaults")

        # Override with environment variables
        env_overrides = {
            'gemini_api_key': os.getenv("GEMINI_API_KEY"),
            'daytona_api_key': os.getenv("DAYTONA_API_KEY"),
            'stability_api_key': os.getenv("STABILITY_API_KEY"),
            'firecrawl_api_key': os.getenv("FIRECRAWL_API_KEY"),
            'daytona_url': os.getenv("DAYTONA_API_URL"),
            'daytona_target': os.getenv("DAYTONA_TARGET"),
            'preview_port': os.getenv("PREVIEW_PORT"),
            'debug_mode': os.getenv("DEBUG_MODE", "").lower() in ("true", "1", "yes"),
        }
        
        # Apply non-None environment overrides
        for key, value in env_overrides.items():
            if value is not None:
                if key == 'preview_port':
                    try:
                        config_data[key] = int(value)
                    except ValueError:
                        logger.warning(f"Invalid port in environment: {value}")
                else:
                    config_data[key] = value

        # Create nested config objects
        if 'performance' in config_data:
            config_data['performance'] = PerformanceConfig(**config_data['performance'])
        if 'security' in config_data:
            config_data['security'] = SecurityConfig(**config_data['security'])

        try:
            return Config(**config_data)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise RuntimeError(f"Invalid configuration: {e}")

    def save(self, path: str = "config.yaml"):
        """Save current configuration to file."""
        try:
            # Convert to dict for serialization
            config_dict = {
                'gemini_api_key': self.gemini_api_key,
                'daytona_api_key': self.daytona_api_key,
                'stability_api_key': self.stability_api_key,
                'firecrawl_api_key': self.firecrawl_api_key,
                'daytona_url': self.daytona_url,
                'daytona_target': self.daytona_target,
                'preview_port': self.preview_port,
                'workspace_dir': self.workspace_dir,
                'models': self.models,
                'intelligent_restart': self.intelligent_restart,
                'debug_mode': self.debug_mode,
                'backup_enabled': self.backup_enabled,
                'auto_save_history': self.auto_save_history,
            }
            
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_dict, f, indent=2, default_flow_style=False)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
