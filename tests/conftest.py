import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import torch
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for testing."""
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("test content")
    return temp_file


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = MagicMock()
    config.name = "test_config"
    config.version = "1.0.0"
    config.debug = False
    return config


@pytest.fixture
def sample_tensor():
    """Create a sample torch tensor for testing."""
    return torch.randn(4, 4)


@pytest.fixture
def sample_numpy_array():
    """Create a sample numpy array for testing."""
    return np.random.randn(4, 4)


@pytest.fixture
def mock_device():
    """Mock torch device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_pointcloud():
    """Create a sample point cloud for testing."""
    return torch.randn(1000, 3)  # 1000 points with x, y, z coordinates


@pytest.fixture
def mock_env():
    """Mock environment for RL testing."""
    env = MagicMock()
    env.observation_space = MagicMock()
    env.action_space = MagicMock()
    env.reset.return_value = torch.zeros(10)
    env.step.return_value = (torch.zeros(10), 0.0, False, {})
    return env


@pytest.fixture
def mock_policy():
    """Mock policy for testing."""
    policy = MagicMock()
    policy.forward.return_value = torch.randn(1, 5)
    policy.parameters.return_value = [torch.randn(10, 10)]
    return policy


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    dataset.__getitem__.return_value = {
        'observations': torch.randn(10),
        'actions': torch.randn(5),
        'rewards': torch.tensor(1.0)
    }
    return dataset


@pytest.fixture
def mock_dataloader():
    """Mock dataloader for testing."""
    dataloader = MagicMock()
    sample_batch = {
        'observations': torch.randn(32, 10),
        'actions': torch.randn(32, 5),
        'rewards': torch.randn(32)
    }
    dataloader.__iter__.return_value = iter([sample_batch])
    dataloader.__len__.return_value = 10
    return dataloader


@pytest.fixture
def mock_wandb_logger():
    """Mock wandb logger for testing."""
    logger = MagicMock()
    logger.log_metrics = MagicMock()
    logger.log_hyperparams = MagicMock()
    return logger


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        'model': {
            'name': 'test_model',
            'layers': [64, 128, 64],
            'activation': 'relu'
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 10
        },
        'data': {
            'train_path': '/path/to/train',
            'val_path': '/path/to/val',
            'test_path': '/path/to/test'
        }
    }


@pytest.fixture
def mock_lightning_module():
    """Mock PyTorch Lightning module for testing."""
    module = MagicMock()
    module.training_step.return_value = torch.tensor(0.5)
    module.validation_step.return_value = torch.tensor(0.3)
    module.test_step.return_value = torch.tensor(0.2)
    return module


@pytest.fixture
def mock_optimizer():
    """Mock optimizer for testing."""
    optimizer = MagicMock()
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    optimizer.param_groups = [{'lr': 0.001}]
    return optimizer


@pytest.fixture
def sample_trajectory():
    """Sample trajectory data for RL testing."""
    return {
        'observations': torch.randn(100, 10),
        'actions': torch.randn(100, 5),
        'rewards': torch.randn(100),
        'dones': torch.zeros(100, dtype=torch.bool),
        'next_observations': torch.randn(100, 10)
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible testing."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)