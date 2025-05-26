"""
Plugin architecture for scryptorum extensions.

This module provides the foundation for extending scryptorum with additional
functionality like HTTP servers, custom storage backends, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ScryptorumPlugin(ABC):
    """Base class for scryptorum plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name identifier."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass


class PluginRegistry:
    """Registry for managing scryptorum plugins."""

    def __init__(self):
        self._plugins: Dict[str, ScryptorumPlugin] = {}

    def register(self, plugin: ScryptorumPlugin) -> None:
        """Register a plugin."""
        self._plugins[plugin.name] = plugin

    def unregister(self, plugin_name: str) -> None:
        """Unregister a plugin."""
        if plugin_name in self._plugins:
            self._plugins[plugin_name].cleanup()
            del self._plugins[plugin_name]

    def get_plugin(self, plugin_name: str) -> Optional[ScryptorumPlugin]:
        """Get a registered plugin by name."""
        return self._plugins.get(plugin_name)

    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def initialize_all(self, config: Dict[str, Any]) -> None:
        """Initialize all registered plugins."""
        for plugin in self._plugins.values():
            plugin_config = config.get(plugin.name, {})
            plugin.initialize(plugin_config)

    def cleanup_all(self) -> None:
        """Clean up all registered plugins."""
        for plugin in self._plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                print(f"Error cleaning up plugin {plugin.name}: {e}")


# Global plugin registry
registry = PluginRegistry()


# Plugin interface for HTTP server (future implementation)
class HTTPServerPlugin(ScryptorumPlugin):
    """Base class for HTTP server plugins."""

    @abstractmethod
    def start_server(self, host: str = "localhost", port: int = 8000) -> None:
        """Start the HTTP server."""
        pass

    @abstractmethod
    def stop_server(self) -> None:
        """Stop the HTTP server."""
        pass

    @abstractmethod
    def add_route(self, path: str, handler: Any) -> None:
        """Add a route to the server."""
        pass
