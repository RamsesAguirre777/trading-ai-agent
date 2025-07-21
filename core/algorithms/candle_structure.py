"""Candle Structure Algorithm."""

from typing import Dict, Any
import pandas as pd
import numpy as np


class CandleStructureAlgorithm:
    """Algoritmo de Estructura de Velas para price action timing."""
    
    def __init__(self):
        self.body_threshold = 0.6  # 60% del cuerpo para considerar vela fuerte
        self.wick_threshold = 0.3  # 30% de mecha para considerar rechazo
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza estructura de velas para timing de price action.
        
        Args:
            data: DataFrame con OHLCV data
            
        Returns:
            Dict con resultado del análisis
        """
        if len(data) < 3:
            return self._empty_result()
        
        current_candle = self._analyze_candle(data.iloc[-1])
        prev_candle = self._analyze_candle(data.iloc[-2])
        
        # Patrones de velas
        patterns = self._identify_patterns(data.tail(3))
        
        # Estructura de Higher Highs/Lower Lows
        structure_analysis = self._analyze_structure(data.tail(10))
        
        # Señal basada en estructura
        structure_signal = self._generate_structure_signal(
            current_candle, prev_candle, patterns, structure_analysis
        )
        
        return {
            "algorithm": "CANDLE_STRUCTURE",
            "signal": structure_signal["signal"],
            "strength": structure_signal["strength"],
            "current_candle": current_candle,
            "patterns": patterns,
            "structure": structure_analysis,
            "conditions": {
                "bullish_structure": structure_analysis["trend"] == "bullish",
                "strong_candle": current_candle["strength"] > 0.6,
                "pattern_confirmation": len(patterns["bullish"]) > 0
            }
        }
    
    def _analyze_candle(self, candle: pd.Series) -> Dict[str, Any]:
        """Analiza una vela individual."""
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']
        
        # Calcular componentes de la vela
        total_range = high_price - low_price
        if total_range == 0:
            total_range = 0.0001  # Evitar división por cero
            
        body_size = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        
        # Calcular porcentajes
        body_percent = body_size / total_range if total_range > 0 else 0
        upper_wick_percent = upper_wick / total_range if total_range > 0 else 0
        lower_wick_percent = lower_wick / total_range if total_range > 0 else 0
        
        # Determinar tipo de vela
        is_bullish = close_price > open_price
        is_bearish = close_price < open_price
        is_doji = body_percent < 0.1
        
        # Calcular fuerza de la vela
        strength = body_percent if not is_doji else 0.1
        
        return {
            "is_bullish": is_bullish,
            "is_bearish": is_bearish,
            "is_doji": is_doji,
            "body_percent": body_percent,
            "upper_wick_percent": upper_wick_percent,
            "lower_wick_percent": lower_wick_percent,
            "strength": strength,
            "total_range": total_range,
            "body_size": body_size
        }
    
    def _identify_patterns(self, data: pd.DataFrame) -> Dict[str, list]:
        """Identifica patrones de velas."""
        patterns = {"bullish": [], "bearish": []}
        
        if len(data) < 3:
            return patterns
        
        candles = [self._analyze_candle(data.iloc[i]) for i in range(len(data))]
        
        # Hammer/Doji en soporte
        if self._is_hammer(candles[-1]):
            patterns["bullish"].append("hammer")
        
        # Shooting star en resistencia
        if self._is_shooting_star(candles[-1]):
            patterns["bearish"].append("shooting_star")
        
        # Engulfing patterns
        if len(candles) >= 2:
            if self._is_bullish_engulfing(candles[-2], candles[-1]):
                patterns["bullish"].append("bullish_engulfing")
            if self._is_bearish_engulfing(candles[-2], candles[-1]):
                patterns["bearish"].append("bearish_engulfing")
        
        return patterns
    
    def _is_hammer(self, candle: Dict[str, Any]) -> bool:
        """Detecta patrón hammer."""
        return (
            candle["lower_wick_percent"] > 0.6 and
            candle["upper_wick_percent"] < 0.1 and
            candle["body_percent"] < 0.3
        )
    
    def _is_shooting_star(self, candle: Dict[str, Any]) -> bool:
        """Detecta patrón shooting star."""
        return (
            candle["upper_wick_percent"] > 0.6 and
            candle["lower_wick_percent"] < 0.1 and
            candle["body_percent"] < 0.3
        )
    
    def _is_bullish_engulfing(self, prev_candle: Dict[str, Any], curr_candle: Dict[str, Any]) -> bool:
        """Detecta patrón bullish engulfing."""
        return (
            prev_candle["is_bearish"] and
            curr_candle["is_bullish"] and
            curr_candle["body_size"] > prev_candle["body_size"] * 1.2
        )
    
    def _is_bearish_engulfing(self, prev_candle: Dict[str, Any], curr_candle: Dict[str, Any]) -> bool:
        """Detecta patrón bearish engulfing."""
        return (
            prev_candle["is_bullish"] and
            curr_candle["is_bearish"] and
            curr_candle["body_size"] > prev_candle["body_size"] * 1.2
        )
    
    def _analyze_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza estructura de Higher Highs/Lower Lows."""
        highs = data['high'].values
        lows = data['low'].values
        
        if len(highs) < 3:
            return {"trend": "neutral", "higher_highs": 0, "lower_lows": 0}
        
        higher_highs = 0
        lower_lows = 0
        
        # Contar Higher Highs
        for i in range(2, len(highs)):
            if highs[i] > highs[i-1] > highs[i-2]:
                higher_highs += 1
        
        # Contar Lower Lows
        for i in range(2, len(lows)):
            if lows[i] < lows[i-1] < lows[i-2]:
                lower_lows += 1
        
        # Determinar tendencia
        if higher_highs > lower_lows:
            trend = "bullish"
        elif lower_lows > higher_highs:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "trend": trend,
            "higher_highs": higher_highs,
            "lower_lows": lower_lows,
            "structure_strength": abs(higher_highs - lower_lows) / len(data)
        }
    
    def _generate_structure_signal(self, current_candle: Dict[str, Any], 
                                 prev_candle: Dict[str, Any], patterns: Dict[str, list],
                                 structure: Dict[str, Any]) -> Dict[str, Any]:
        """Genera señal basada en estructura de velas."""
        signal = "WAIT"
        strength = 0.0
        
        # Scoring basado en múltiples factores
        bullish_score = 0
        bearish_score = 0
        
        # Factor estructura
        if structure["trend"] == "bullish":
            bullish_score += 0.3
        elif structure["trend"] == "bearish":
            bearish_score += 0.3
        
        # Factor patrones
        bullish_score += len(patterns["bullish"]) * 0.2
        bearish_score += len(patterns["bearish"]) * 0.2
        
        # Factor fuerza de vela actual
        if current_candle["is_bullish"] and current_candle["strength"] > 0.6:
            bullish_score += 0.3
        elif current_candle["is_bearish"] and current_candle["strength"] > 0.6:
            bearish_score += 0.3
        
        # Determinar señal final
        if bullish_score > bearish_score and bullish_score > 0.5:
            signal = "BUY"
            strength = min(bullish_score, 1.0)
        elif bearish_score > bullish_score and bearish_score > 0.5:
            signal = "SELL"
            strength = min(bearish_score, 1.0)
        
        return {"signal": signal, "strength": strength}
    
    def _empty_result(self) -> Dict[str, Any]:
        """Retorna resultado vacío cuando no hay suficientes datos."""
        return {
            "algorithm": "CANDLE_STRUCTURE",
            "signal": "WAIT",
            "strength": 0.0,
            "current_candle": {},
            "patterns": {"bullish": [], "bearish": []},
            "structure": {"trend": "neutral"},
            "conditions": {
                "bullish_structure": False,
                "strong_candle": False,
                "pattern_confirmation": False
            }
        }