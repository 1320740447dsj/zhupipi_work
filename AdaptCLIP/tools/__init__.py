from .logger import get_logger
from .utils import get_transform, normalize, setup_seed


def __getattr__(name):
    if name == 'Evaluator':
        from .effecient_metric import Evaluator
        return Evaluator
    if name == 'visualizer':
        from .visualization import visualizer
        return visualizer
    raise AttributeError(f"module 'tools' has no attribute {name!r}")


__all__ = ['get_logger', 'get_transform', 'normalize', 'setup_seed', 'Evaluator', 'visualizer']
