"""7 Algoritmos principales del sistema RAMSES ANALIZER."""

from .ema_rotation import EMARotationAlgorithm
from .swing_points import SwingPointsAlgorithm
from .break_points import BreakPointsAlgorithm
from .candle_structure import CandleStructureAlgorithm
from .micro_levels import MicroLevelsAlgorithm
from .bollinger_extremes import BollingerExtremesAlgorithm
from .technical_strategy import TechnicalStrategyAlgorithm

__all__ = [
    "EMARotationAlgorithm",
    "SwingPointsAlgorithm",
    "BreakPointsAlgorithm",
    "CandleStructureAlgorithm",
    "MicroLevelsAlgorithm",
    "BollingerExtremesAlgorithm",
    "TechnicalStrategyAlgorithm",
]