"""Vision Classification Framework - Main Package"""

__version__ = "1.0.0"
__author__ = "Vision-CLS Contributors"

from .train import Config, Trainer
from .datasets import ImageDataset
from .modules import pretrained_network, get_ml_model
from .inference import Predictor, predict_single

__all__ = [
    'Config',
    'Trainer',
    'ImageDataset',
    'pretrained_network',
    'get_ml_model',
    'Predictor',
    'predict_single',
]
