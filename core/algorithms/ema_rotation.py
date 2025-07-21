"""EMA3/EMA9 + MACD Rotation Algorithm."""

from typing import Dict, Any
import pandas as pd
import numpy as np

from ..indicators.ema import EMAIndicator
from ..indicators.macd import MACDIndicator


class EMARotationAlgorithm:
    """Algoritmo de rotación EMA3/EMA9 + MACD para momentum timing."""
    
    def __init__(self):
        self.ema_indicator = EMAIndicator()
        self.macd_indicator = MACDIndicator()
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza rotación EMA3/EMA9 con confirmación MACD.
        
        Args:
            data: DataFrame con OHLCV data
            
        Returns:
            Dict con resultado del análisis
        """
        # Calcular EMAs
        ema3 = self.ema_indicator.calculate(data['close'], period=3)
        ema9 = self.ema_indicator.calculate(data['close'], period=9)
        
        # Calcular MACD
        macd_data = self.macd_indicator.calculate(data['close'])
        
        # Condiciones de rotación
        current_ema3 = ema3.iloc[-1]
        current_ema9 = ema9.iloc[-1]
        current_macd = macd_data['macd'].iloc[-1]
        current_signal = macd_data['signal'].iloc[-1]
        
        # Análisis de rotación
        ema_bullish = current_ema3 > current_ema9
        macd_bullish = current_macd > current_signal
        
        # Detección de cruces
        prev_ema3 = ema3.iloc[-2]
        prev_ema9 = ema9.iloc[-2]
        prev_macd = macd_data['macd'].iloc[-2]
        prev_signal = macd_data['signal'].iloc[-2]
        
        ema_cross_bullish = (prev_ema3 <= prev_ema9) and (current_ema3 > current_ema9)
        ema_cross_bearish = (prev_ema3 >= prev_ema9) and (current_ema3 < current_ema9)
        
        macd_cross_bullish = (prev_macd <= prev_signal) and (current_macd > current_signal)
        macd_cross_bearish = (prev_macd >= prev_signal) and (current_macd < current_signal)
        
        # Señal de rotación
        rotation_signal = None
        if ema_bullish and macd_bullish:
            rotation_signal = "BUY"
        elif not ema_bullish and not macd_bullish:
            rotation_signal = "SELL"
        else:
            rotation_signal = "WAIT"
        
        # Fuerza de la señal
        signal_strength = 0
        if ema_cross_bullish or ema_cross_bearish:
            signal_strength += 0.4
        if macd_cross_bullish or macd_cross_bearish:
            signal_strength += 0.4
        if ema_bullish == macd_bullish:
            signal_strength += 0.2
        
        return {
            "algorithm": "EMA_ROTATION",
            "signal": rotation_signal,
            "strength": signal_strength,
            "conditions": {
                "ema3_gt_ema9": ema_bullish,
                "macd_gt_signal": macd_bullish,
                "ema_cross_bullish": ema_cross_bullish,
                "macd_cross_bullish": macd_cross_bullish
            },
            "values": {
                "ema3": current_ema3,
                "ema9": current_ema9,
                "macd": current_macd,
                "signal": current_signal
            }
        }