"""Swing Points Dinámicos Algorithm."""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


class SwingPointsAlgorithm:
    """Algoritmo de Swing Points Dinámicos para niveles adaptativos macro."""
    
    def __init__(self, lookback_period: int = 20, sensitivity: float = 0.02):
        self.lookback_period = lookback_period
        self.sensitivity = sensitivity
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza swing points dinámicos.
        
        Args:
            data: DataFrame con OHLCV data
            
        Returns:
            Dict con resultado del análisis
        """
        # Identificar swing highs y lows
        swing_highs = self._find_swing_highs(data['high'])
        swing_lows = self._find_swing_lows(data['low'])
        
        # Calcular niveles de soporte y resistencia
        support_levels = self._calculate_support_levels(swing_lows)
        resistance_levels = self._calculate_resistance_levels(swing_highs)
        
        # Precio actual
        current_price = data['close'].iloc[-1]
        
        # Análisis de proximidad a niveles
        nearest_support = self._find_nearest_level(current_price, support_levels, "below")
        nearest_resistance = self._find_nearest_level(current_price, resistance_levels, "above")
        
        # Detección de breaks
        support_break = self._detect_support_break(data, support_levels)
        resistance_break = self._detect_resistance_break(data, resistance_levels)
        
        # Señal basada en swing points
        swing_signal = self._generate_swing_signal(
            current_price, nearest_support, nearest_resistance,
            support_break, resistance_break
        )
        
        return {
            "algorithm": "SWING_POINTS",
            "signal": swing_signal["signal"],
            "strength": swing_signal["strength"],
            "levels": {
                "support": support_levels,
                "resistance": resistance_levels,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance
            },
            "breaks": {
                "support_break": support_break,
                "resistance_break": resistance_break
            },
            "current_price": current_price
        }
    
    def _find_swing_highs(self, high_series: pd.Series) -> List[Tuple[int, float]]:
        """Encuentra swing highs en la serie de precios altos."""
        swing_highs = []
        
        for i in range(self.lookback_period, len(high_series) - self.lookback_period):
            current_high = high_series.iloc[i]
            
            # Verificar si es un máximo local
            is_swing_high = True
            for j in range(i - self.lookback_period, i + self.lookback_period + 1):
                if j != i and high_series.iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append((i, current_high))
        
        return swing_highs
    
    def _find_swing_lows(self, low_series: pd.Series) -> List[Tuple[int, float]]:
        """Encuentra swing lows en la serie de precios bajos."""
        swing_lows = []
        
        for i in range(self.lookback_period, len(low_series) - self.lookback_period):
            current_low = low_series.iloc[i]
            
            # Verificar si es un mínimo local
            is_swing_low = True
            for j in range(i - self.lookback_period, i + self.lookback_period + 1):
                if j != i and low_series.iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append((i, current_low))
        
        return swing_lows
    
    def _calculate_support_levels(self, swing_lows: List[Tuple[int, float]]) -> List[float]:
        """Calcula niveles de soporte basados en swing lows."""
        if not swing_lows:
            return []
        
        # Agrupar swing lows similares
        levels = []
        swing_prices = [price for _, price in swing_lows]
        
        for price in swing_prices:
            # Verificar si hay un nivel similar existente
            similar_level = None
            for level in levels:
                if abs(price - level) / level < self.sensitivity:
                    similar_level = level
                    break
            
            if similar_level is None:
                levels.append(price)
        
        return sorted(levels)
    
    def _calculate_resistance_levels(self, swing_highs: List[Tuple[int, float]]) -> List[float]:
        """Calcula niveles de resistencia basados en swing highs."""
        if not swing_highs:
            return []
        
        # Agrupar swing highs similares
        levels = []
        swing_prices = [price for _, price in swing_highs]
        
        for price in swing_prices:
            # Verificar si hay un nivel similar existente
            similar_level = None
            for level in levels:
                if abs(price - level) / level < self.sensitivity:
                    similar_level = level
                    break
            
            if similar_level is None:
                levels.append(price)
        
        return sorted(levels, reverse=True)
    
    def _find_nearest_level(self, price: float, levels: List[float], direction: str) -> float:
        """Encuentra el nivel más cercano en la dirección especificada."""
        if not levels:
            return None
        
        if direction == "below":
            below_levels = [level for level in levels if level < price]
            return max(below_levels) if below_levels else None
        else:  # above
            above_levels = [level for level in levels if level > price]
            return min(above_levels) if above_levels else None
    
    def _detect_support_break(self, data: pd.DataFrame, support_levels: List[float]) -> bool:
        """Detecta si se ha roto un nivel de soporte."""
        if not support_levels:
            return False
        
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        
        for level in support_levels:
            if prev_price >= level and current_price < level:
                return True
        
        return False
    
    def _detect_resistance_break(self, data: pd.DataFrame, resistance_levels: List[float]) -> bool:
        """Detecta si se ha roto un nivel de resistencia."""
        if not resistance_levels:
            return False
        
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        
        for level in resistance_levels:
            if prev_price <= level and current_price > level:
                return True
        
        return False
    
    def _generate_swing_signal(self, current_price: float, nearest_support: float, 
                             nearest_resistance: float, support_break: bool, 
                             resistance_break: bool) -> Dict[str, Any]:
        """Genera señal basada en análisis de swing points."""
        signal = "WAIT"
        strength = 0.0
        
        if resistance_break:
            signal = "BUY"
            strength = 0.8
        elif support_break:
            signal = "SELL"
            strength = 0.8
        elif nearest_support and nearest_resistance:
            # Calcular posición dentro del rango
            range_size = nearest_resistance - nearest_support
            position_in_range = (current_price - nearest_support) / range_size
            
            if position_in_range < 0.2:
                signal = "BUY"
                strength = 0.4
            elif position_in_range > 0.8:
                signal = "SELL"
                strength = 0.4
        
        return {"signal": signal, "strength": strength}