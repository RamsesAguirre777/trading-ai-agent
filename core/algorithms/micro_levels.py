"""Micro Levels Algorithm."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class MicroLevelsAlgorithm:
    """Algoritmo de Niveles Micro de Velas para precisión quirúrgica."""
    
    def __init__(self, precision_threshold: float = 0.001):
        self.precision_threshold = precision_threshold  # 0.1% para niveles micro
        self.lookback_candles = 5
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza niveles micro de velas para precisión quirúrgica.
        
        Args:
            data: DataFrame con OHLCV data
            
        Returns:
            Dict con resultado del análisis
        """
        if len(data) < self.lookback_candles + 1:
            return self._empty_result()
        
        # Analizar niveles micro en velas recientes
        micro_levels = self._identify_micro_levels(data.tail(self.lookback_candles + 1))
        
        # Precio actual y análisis de proximidad
        current_price = data['close'].iloc[-1]
        proximity_analysis = self._analyze_proximity(current_price, micro_levels)
        
        # Detectar interacciones con niveles micro
        level_interactions = self._detect_level_interactions(data.tail(3), micro_levels)
        
        # Generar señal micro
        micro_signal = self._generate_micro_signal(
            proximity_analysis, level_interactions, current_price
        )
        
        return {
            "algorithm": "MICRO_LEVELS",
            "signal": micro_signal["signal"],
            "strength": micro_signal["strength"],
            "micro_levels": micro_levels,
            "proximity": proximity_analysis,
            "interactions": level_interactions,
            "current_price": current_price,
            "conditions": {
                "near_micro_support": proximity_analysis.get("near_support", False),
                "near_micro_resistance": proximity_analysis.get("near_resistance", False),
                "level_bounce": level_interactions.get("bounce", False)
            }
        }
    
    def _identify_micro_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identifica niveles micro en las velas."""
        micro_supports = []
        micro_resistances = []
        
        for _, candle in data.iterrows():
            # Niveles de apertura y cierre como micro niveles
            open_price = candle['open']
            close_price = candle['close']
            high_price = candle['high']
            low_price = candle['low']
            
            # Calcular niveles de retroceso dentro de la vela
            candle_range = high_price - low_price
            
            if candle_range > 0:
                # Niveles de Fibonacci micro (dentro de la vela)
                fib_levels = [
                    low_price + candle_range * 0.236,
                    low_price + candle_range * 0.382,
                    low_price + candle_range * 0.5,
                    low_price + candle_range * 0.618,
                    low_price + candle_range * 0.786
                ]
                
                # Clasificar como soporte o resistencia basado en price action
                if close_price > open_price:  # Vela alcista
                    micro_supports.extend([open_price, low_price])
                    micro_supports.extend([level for level in fib_levels if level <= close_price])
                    micro_resistances.extend([high_price])
                    micro_resistances.extend([level for level in fib_levels if level > close_price])
                else:  # Vela bajista
                    micro_resistances.extend([open_price, high_price])
                    micro_resistances.extend([level for level in fib_levels if level >= close_price])
                    micro_supports.extend([low_price])
                    micro_supports.extend([level for level in fib_levels if level < close_price])
        
        # Limpiar y agrupar niveles similares
        micro_supports = self._cluster_levels(micro_supports)
        micro_resistances = self._cluster_levels(micro_resistances)
        
        return {
            "supports": sorted(micro_supports),
            "resistances": sorted(micro_resistances, reverse=True)
        }
    
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """Agrupa niveles similares para evitar redundancia."""
        if not levels:
            return []
        
        clustered = []
        sorted_levels = sorted(levels)
        
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Si el nivel está muy cerca del cluster actual, agregarlo
            if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) < self.precision_threshold:
                current_cluster.append(level)
            else:
                # Finalizar cluster actual y empezar uno nuevo
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Agregar último cluster
        clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def _analyze_proximity(self, current_price: float, micro_levels: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analiza proximidad del precio actual a niveles micro."""
        supports = micro_levels["supports"]
        resistances = micro_levels["resistances"]
        
        # Encontrar niveles más cercanos
        nearest_support = None
        nearest_resistance = None
        
        # Soporte más cercano por debajo
        below_supports = [s for s in supports if s < current_price]
        if below_supports:
            nearest_support = max(below_supports)
        
        # Resistencia más cercana por encima
        above_resistances = [r for r in resistances if r > current_price]
        if above_resistances:
            nearest_resistance = min(above_resistances)
        
        # Calcular distancias
        support_distance = None
        resistance_distance = None
        
        if nearest_support:
            support_distance = (current_price - nearest_support) / current_price
        
        if nearest_resistance:
            resistance_distance = (nearest_resistance - current_price) / current_price
        
        # Determinar si está cerca de niveles
        near_support = support_distance is not None and support_distance < self.precision_threshold * 2
        near_resistance = resistance_distance is not None and resistance_distance < self.precision_threshold * 2
        
        return {
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "support_distance": support_distance,
            "resistance_distance": resistance_distance,
            "near_support": near_support,
            "near_resistance": near_resistance
        }
    
    def _detect_level_interactions(self, data: pd.DataFrame, micro_levels: Dict[str, List[float]]) -> Dict[str, Any]:
        """Detecta interacciones del precio con niveles micro."""
        if len(data) < 2:
            return {"bounce": False, "break": False, "interaction_type": None}
        
        prev_candle = data.iloc[-2]
        curr_candle = data.iloc[-1]
        
        all_levels = micro_levels["supports"] + micro_levels["resistances"]
        
        bounce_detected = False
        break_detected = False
        interaction_type = None
        
        for level in all_levels:
            # Detectar bounce
            if self._is_bounce(prev_candle, curr_candle, level):
                bounce_detected = True
                interaction_type = "bounce"
            
            # Detectar break
            if self._is_break(prev_candle, curr_candle, level):
                break_detected = True
                interaction_type = "break"
        
        return {
            "bounce": bounce_detected,
            "break": break_detected,
            "interaction_type": interaction_type
        }
    
    def _is_bounce(self, prev_candle: pd.Series, curr_candle: pd.Series, level: float) -> bool:
        """Detecta si hay un bounce en un nivel micro."""
        # El precio tocó el nivel y rebotó
        prev_low = prev_candle['low']
        prev_high = prev_candle['high']
        curr_close = curr_candle['close']
        
        # Bounce desde soporte
        if (prev_low <= level <= prev_high and 
            curr_close > level and 
            abs(prev_low - level) / level < self.precision_threshold):
            return True
        
        # Bounce desde resistencia
        if (prev_low <= level <= prev_high and 
            curr_close < level and 
            abs(prev_high - level) / level < self.precision_threshold):
            return True
        
        return False
    
    def _is_break(self, prev_candle: pd.Series, curr_candle: pd.Series, level: float) -> bool:
        """Detecta si hay un break de un nivel micro."""
        prev_close = prev_candle['close']
        curr_close = curr_candle['close']
        
        # Break alcista
        if prev_close <= level and curr_close > level * (1 + self.precision_threshold):
            return True
        
        # Break bajista
        if prev_close >= level and curr_close < level * (1 - self.precision_threshold):
            return True
        
        return False
    
    def _generate_micro_signal(self, proximity: Dict[str, Any], 
                             interactions: Dict[str, Any], 
                             current_price: float) -> Dict[str, Any]:
        """Genera señal basada en análisis de niveles micro."""
        signal = "WAIT"
        strength = 0.0
        
        # Señal basada en bounces
        if interactions["bounce"]:
            if proximity["near_support"]:
                signal = "BUY"
                strength = 0.7
            elif proximity["near_resistance"]:
                signal = "SELL"
                strength = 0.7
        
        # Señal basada en breaks
        elif interactions["break"]:
            if (proximity["nearest_resistance"] and 
                current_price > proximity["nearest_resistance"]):
                signal = "BUY"
                strength = 0.6
            elif (proximity["nearest_support"] and 
                  current_price < proximity["nearest_support"]):
                signal = "SELL"
                strength = 0.6
        
        # Señal basada en proximidad extrema
        elif proximity["near_support"] and proximity["support_distance"] < self.precision_threshold:
            signal = "BUY"
            strength = 0.4
        elif proximity["near_resistance"] and proximity["resistance_distance"] < self.precision_threshold:
            signal = "SELL"
            strength = 0.4
        
        return {"signal": signal, "strength": strength}
    
    def _empty_result(self) -> Dict[str, Any]:
        """Retorna resultado vacío cuando no hay suficientes datos."""
        return {
            "algorithm": "MICRO_LEVELS",
            "signal": "WAIT",
            "strength": 0.0,
            "micro_levels": {"supports": [], "resistances": []},
            "proximity": {},
            "interactions": {"bounce": False, "break": False},
            "current_price": 0.0,
            "conditions": {
                "near_micro_support": False,
                "near_micro_resistance": False,
                "level_bounce": False
            }
        }