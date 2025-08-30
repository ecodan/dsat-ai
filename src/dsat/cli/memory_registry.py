"""
Registry system for memory management strategies and plugins.

This module provides centralized registration and discovery of memory strategies,
plugins, and custom memory managers.
"""

import logging
from typing import Dict, List, Optional, Type, Any

from .memory_interfaces import BaseMemoryStrategy, BaseMemoryManager, MemoryPlugin

# Entry points for plugin discovery
try:
    from importlib.metadata import entry_points
    ENTRY_POINTS_AVAILABLE = True
except ImportError:
    try:
        from importlib_metadata import entry_points
        ENTRY_POINTS_AVAILABLE = True
    except ImportError:
        ENTRY_POINTS_AVAILABLE = False


class MemoryStrategyRegistry:
    """Registry for memory management strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseMemoryStrategy]] = {}
        self._plugins: Dict[str, MemoryPlugin] = {}
        self._memory_managers: Dict[str, Type[BaseMemoryManager]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load built-in strategies
        self._register_builtin_strategies()
        
        # Discover and load plugins
        self._discover_plugins()
    
    def register_strategy(self, name: str, strategy_class: Type[BaseMemoryStrategy]) -> None:
        """
        Register a memory strategy.
        
        :param name: Strategy name identifier
        :param strategy_class: Strategy class
        """
        if not issubclass(strategy_class, BaseMemoryStrategy):
            raise ValueError("Strategy class must inherit from BaseMemoryStrategy")
        
        self._strategies[name] = strategy_class
        self.logger.debug(f"Registered memory strategy: {name}")
    
    def register_memory_manager(self, name: str, manager_class: Type[BaseMemoryManager]) -> None:
        """
        Register a custom memory manager.
        
        :param name: Manager name identifier
        :param manager_class: Manager class
        """
        if not issubclass(manager_class, BaseMemoryManager):
            raise ValueError("Manager class must inherit from BaseMemoryManager")
        
        self._memory_managers[name] = manager_class
        self.logger.debug(f"Registered memory manager: {name}")
    
    def register_plugin(self, plugin: MemoryPlugin) -> None:
        """
        Register a memory plugin.
        
        :param plugin: Memory plugin instance
        """
        plugin_name = plugin.name
        
        # Register the plugin
        self._plugins[plugin_name] = plugin
        
        # Register strategies from the plugin
        for strategy_name, strategy_class in plugin.get_strategies().items():
            full_name = f"{plugin_name}.{strategy_name}"
            self.register_strategy(full_name, strategy_class)
        
        # Register memory manager if provided
        manager_class = plugin.get_memory_manager_class()
        if manager_class:
            self.register_memory_manager(plugin_name, manager_class)
        
        self.logger.info(f"Registered memory plugin: {plugin_name} v{plugin.version}")
    
    def get_strategy(self, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseMemoryStrategy]:
        """
        Get a memory strategy by name.
        
        :param name: Strategy name
        :param config: Strategy configuration
        :return: Strategy instance or None
        """
        strategy_class = self._strategies.get(name)
        if not strategy_class:
            return None
        
        try:
            strategy = strategy_class(config)
            
            # Validate configuration if provided
            if config and not strategy.validate_config():
                self.logger.warning(f"Invalid configuration for strategy: {name}")
                return None
            
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to create strategy {name}: {e}")
            return None
    
    def get_memory_manager(self, name: str, **kwargs) -> Optional[BaseMemoryManager]:
        """
        Get a memory manager by name.
        
        :param name: Manager name
        :param kwargs: Manager initialization arguments
        :return: Manager instance or None
        """
        manager_class = self._memory_managers.get(name)
        if not manager_class:
            return None
        
        try:
            return manager_class(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create memory manager {name}: {e}")
            return None
    
    def list_strategies(self) -> List[str]:
        """
        List all registered strategy names.
        
        :return: List of strategy names
        """
        return list(self._strategies.keys())
    
    def list_memory_managers(self) -> List[str]:
        """
        List all registered memory manager names.
        
        :return: List of memory manager names
        """
        return list(self._memory_managers.keys())
    
    def list_plugins(self) -> List[Dict[str, str]]:
        """
        List all registered plugins.
        
        :return: List of plugin info dicts
        """
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description
            }
            for plugin in self._plugins.values()
        ]
    
    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a strategy.
        
        :param name: Strategy name
        :return: Strategy info dict or None
        """
        strategy_class = self._strategies.get(name)
        if not strategy_class:
            return None
        
        try:
            # Create temporary instance to get info
            temp_strategy = strategy_class()
            return {
                "name": temp_strategy.name,
                "description": temp_strategy.description,
                "config_schema": temp_strategy.get_config_schema(),
                "class": strategy_class.__name__,
                "module": strategy_class.__module__
            }
        except Exception as e:
            self.logger.error(f"Failed to get info for strategy {name}: {e}")
            return None
    
    def _register_builtin_strategies(self) -> None:
        """Register built-in memory strategies."""
        # Import built-in strategies here to avoid circular imports
        try:
            from .memory_strategies import (
                PruningMemoryStrategy,
                CompactingMemoryStrategy,
                SlidingWindowMemoryStrategy
            )
            
            self.register_strategy("pruning", PruningMemoryStrategy)
            self.register_strategy("compacting", CompactingMemoryStrategy)
            self.register_strategy("sliding_window", SlidingWindowMemoryStrategy)
            
        except ImportError as e:
            self.logger.warning(f"Failed to import built-in strategies: {e}")
    
    def _discover_plugins(self) -> None:
        """Discover and load memory plugins via entry points."""
        if not ENTRY_POINTS_AVAILABLE:
            self.logger.debug("Entry points not available, skipping plugin discovery")
            return
        
        try:
            # Discover memory plugins via entry points
            discovered_plugins = entry_points(group="dsat.memory_plugins")
            
            for entry_point in discovered_plugins:
                try:
                    plugin_class = entry_point.load()
                    plugin = plugin_class()
                    
                    # Initialize with empty config - plugins should handle defaults
                    plugin.initialize({})
                    
                    self.register_plugin(plugin)
                    
                except Exception as e:
                    self.logger.error(f"Failed to load plugin {entry_point.name}: {e}")
                    
        except Exception as e:
            self.logger.debug(f"Plugin discovery failed: {e}")


# Global registry instance
_registry: Optional[MemoryStrategyRegistry] = None


def get_registry() -> MemoryStrategyRegistry:
    """Get the global memory strategy registry."""
    global _registry
    if _registry is None:
        _registry = MemoryStrategyRegistry()
    return _registry


def register_strategy(name: str, strategy_class: Type[BaseMemoryStrategy]) -> None:
    """Register a memory strategy globally."""
    get_registry().register_strategy(name, strategy_class)


def register_memory_manager(name: str, manager_class: Type[BaseMemoryManager]) -> None:
    """Register a memory manager globally."""
    get_registry().register_memory_manager(name, manager_class)


def register_plugin(plugin: MemoryPlugin) -> None:
    """Register a memory plugin globally."""
    get_registry().register_plugin(plugin)


def get_strategy(name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseMemoryStrategy]:
    """Get a memory strategy by name."""
    return get_registry().get_strategy(name, config)


def get_memory_manager(name: str, **kwargs) -> Optional[BaseMemoryManager]:
    """Get a memory manager by name."""
    return get_registry().get_memory_manager(name, **kwargs)


def list_available_strategies() -> List[str]:
    """List all available memory strategies."""
    return get_registry().list_strategies()


def list_available_memory_managers() -> List[str]:
    """List all available memory managers."""
    return get_registry().list_memory_managers()