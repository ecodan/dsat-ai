"""
Tests to validate the new architecture works correctly.
"""

import pytest
from pathlib import Path
import tempfile

def test_scryptorum_works_independently():
    """Test that scryptorum works without agents module."""
    from dsat.scryptorum.core.experiment import Experiment
    from dsat.scryptorum.core.runs import RunType
    from dsat.scryptorum.core.config import ConfigManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Should be able to create experiment
        experiment = Experiment(temp_path, "independent_test")
        assert experiment.experiment_name == "independent_test"
        
        # Should be able to create runs
        trial_run = experiment.create_run(RunType.TRIAL)
        assert trial_run.run_type == RunType.TRIAL
        
        # Should be able to use config manager
        config_manager = ConfigManager(experiment.config_dir)
        config_file = config_manager.save_config("test_config", {"setting": "value"})
        assert config_file.exists()
        
        loaded_config = config_manager.load_config("test_config")
        assert loaded_config["setting"] == "value"

def test_agents_module_optional_imports():
    """Test that agents module handles optional scryptorum imports gracefully."""
    try:
        # Should be able to import basic agents functionality
        from dsat.agents import Agent, AgentConfig, PromptManager
        agents_available = True
    except ImportError:
        agents_available = False
        
    if agents_available:
        # Should be able to import enhanced classes if scryptorum is available
        try:
            from dsat.agents import AgentExperiment, AgentRun
            enhanced_available = True
        except ImportError:
            enhanced_available = False
            
        # If scryptorum is available, enhanced classes should be too
        import sys
        if 'scryptorum' in sys.modules:
            assert enhanced_available, "AgentExperiment should be available when scryptorum is imported"

def test_both_patterns_coexist():
    """Test that both base and agent patterns can coexist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Base scryptorum experiment
        from dsat.scryptorum.core.experiment import Experiment
        base_experiment = Experiment(temp_path / "base", "base_test")
        base_run = base_experiment.create_run()
        
        # Should not have agent methods
        assert not hasattr(base_experiment, 'create_agent')
        assert not hasattr(base_run, 'log_agent_created')
        
        # Try agent experiment if available
        try:
            from dsat.agents import AgentExperiment
            agent_experiment = AgentExperiment(temp_path / "agent", "agent_test")
            agent_run = agent_experiment.create_run()
            
            # Should have agent methods
            assert hasattr(agent_experiment, 'create_agent')
            assert hasattr(agent_run, 'log_agent_created')
            
            # Should also have base methods
            assert hasattr(agent_experiment, 'create_run')
            assert hasattr(agent_run, 'log_metric')
            
        except ImportError:
            # Agent experiment not available - that's fine
            pass

def test_no_coupling_violations():
    """Test that there are no unwanted dependencies."""
    
    # Scryptorum should not import agents
    import dsat.scryptorum.core.experiment
    import dsat.scryptorum.core.runs
    import dsat.scryptorum.core.config
    
    # Check that no agents imports exist in scryptorum modules
    scryptorum_modules = [
        scryptorum.core.experiment,
        scryptorum.core.runs, 
        scryptorum.core.config
    ]
    
    for module in scryptorum_modules:
        module_file = Path(module.__file__)
        content = module_file.read_text()
        
        # Should not have direct imports of agents
        assert "from agents" not in content, f"{module.__name__} should not import agents"
        assert "import agents" not in content, f"{module.__name__} should not import agents"

def test_clean_inheritance_pattern():
    """Test that the inheritance pattern is clean."""
    try:
        from dsat.agents import AgentExperiment, AgentRun
        from dsat.scryptorum.core.experiment import Experiment
        from dsat.scryptorum.core.runs import Run
        
        # AgentExperiment should inherit from Experiment
        assert issubclass(AgentExperiment, Experiment)
        
        # AgentRun should inherit from Run
        assert issubclass(AgentRun, Run)
        
        # Should be able to create instances
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            agent_experiment = AgentExperiment(temp_path, "inheritance_test")
            agent_run = agent_experiment.create_run()
            
            # Should be instances of both base and derived classes
            assert isinstance(agent_experiment, Experiment)
            assert isinstance(agent_experiment, AgentExperiment)
            
            assert isinstance(agent_run, Run)
            assert isinstance(agent_run, AgentRun)
            
    except ImportError:
        # Agents not available - skip this test
        pytest.skip("agents module not available")

def test_configuration_isolation():
    """Test that configurations are properly isolated."""
    from dsat.scryptorum.core.config import ConfigManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Base config manager should work with any JSON config
        config_manager = ConfigManager(temp_path / "config")
        
        # Should handle any type of config
        generic_config = {
            "type": "generic",
            "settings": {"value": 42},
            "features": ["a", "b", "c"]
        }
        
        config_file = config_manager.save_config("generic", generic_config)
        loaded_config = config_manager.load_config("generic")
        
        assert loaded_config == generic_config
        
        # Try agent config if available
        try:
            from dsat.agents import AgentExperiment
            
            agent_experiment = AgentExperiment(temp_path, "config_test")
            
            # Agent configs should be stored separately
            agent_experiment.create_agent_config("test_agent", prompt="test:v1")
            
            # Should have separate namespaces
            generic_configs = config_manager.list_configs()
            agent_configs = agent_experiment.list_agent_configs()
            
            assert "generic" in generic_configs
            assert "test_agent" in agent_configs
            
            # Should not interfere with each other
            assert "test_agent" not in generic_configs
            assert "generic" not in agent_configs
            
        except ImportError:
            # Agents not available - that's fine
            pass

def test_backwards_compatibility():
    """Test that existing scryptorum code continues to work."""
    from dsat.scryptorum import experiment, metric, timer, Experiment, RunType
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Old decorator pattern should still work
        @experiment(name="compatibility_test")
        def test_experiment():
            return test_metric()
        
        @metric(name="test_metric", metric_type="test")
        def test_metric():
            return 0.5
        
        # Should execute without issues
        from dsat.scryptorum.core.decorators import set_default_run_type
        set_default_run_type(RunType.TRIAL)
        
        result = test_experiment(project_root=temp_path)
        assert result == 0.5
        
        # Manual experiment creation should still work
        manual_experiment = Experiment(temp_path, "manual_test")
        manual_run = manual_experiment.create_run()
        manual_run.log_metric("manual_metric", 0.7)
        manual_run.finish()
        
        # Should have created the expected structure
        assert (temp_path / "experiments" / "manual_test").exists()
        assert (temp_path / "experiments" / "manual_test" / "runs" / "trial_run").exists()