"""Bollinger Extremes Algorithm."""

from typing import Dict, Any
import pandas as pd
import numpy as np

from ..indicators.bollinger_bands import BollingerBandsIndicator


class BollingerExtremesAlgorithm:
    """Algoritmo de Extremos Bollinger Bands para condiciones extremas."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, extreme_threshold: float = 0.95):
        self.period = period
        self.std_dev = std_dev
        self.extreme_threshold = extreme_threshold  # 95% para considerar extremo
        self.bb_indicator = BollingerBandsIndicator(period, std_dev)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza extremos de Bollinger Bands para condiciones extremas.
        
        Args:
            data: DataFrame con OHLCV data
            
        Returns:
            Dict con resultado del análisis
        """
        if len(data) < self.period + 1:
            return self._empty_result()
        
        # Calcular Bollinger Bands
        bb_data = self.bb_indicator.calculate(data['close'])
        
        current_price = data['close'].iloc[-1]
        current_bb_upper = bb_data['bb_upper'].iloc[-1]
        current_bb_lower = bb_data['bb_lower'].iloc[-1]
        current_bb_middle = bb_data['bb_middle'].iloc[-1]
        
        # Analizar posición dentro de las bandas
        position_analysis = self._analyze_position_in_bands(
            current_price, current_bb_upper, current_bb_lower, current_bb_middle
        )
        
        # Detectar extremos
        extremes_analysis = self._detect_extremes(
            data, bb_data, position_analysis
        )
        
        # Analizar squeeze y expansión
        squeeze_analysis = self._analyze_squeeze(bb_data)
        
        # Generar señal de extremos
        extremes_signal = self._generate_extremes_signal(
            position_analysis, extremes_analysis, squeeze_analysis
        )
        
        return {
            "algorithm": "BOLLINGER_EXTREMES",
            "signal": extremes_signal["signal"],
            "strength": extremes_signal["strength"],
            "position": position_analysis,
            "extremes": extremes_analysis,
            "squeeze": squeeze_analysis,
            "bb_values": {
                "upper": current_bb_upper,
                "middle": current_bb_middle,
                "lower": current_bb_lower,
                "current_price": current_price
            },
            "conditions": {
                "extreme_overbought": extremes_analysis["is_extreme_high"],
                "extreme_oversold": extremes_analysis["is_extreme_low"],
                "squeeze_active": squeeze_analysis["is_squeeze"]
            }
        }
    
    def _analyze_position_in_bands(self, price: float, bb_upper: float, 
                                  bb_lower: float, bb_middle: float) -> Dict[str, Any]:
        """Analiza la posición del precio dentro de las Bollinger Bands."""
        band_width = bb_upper - bb_lower
        position_in_band = (price - bb_lower) / band_width if band_width > 0 else 0.5
        
        # Normalizar posición (0 = banda inferior, 1 = banda superior)
        position_in_band = max(0, min(1, position_in_band))
        
        # Determinar zona
        if position_in_band >= 0.8:
            zone = "upper_extreme"
        elif position_in_band >= 0.6:
            zone = "upper"
        elif position_in_band >= 0.4:
            zone = "middle"
        elif position_in_band >= 0.2:
            zone = "lower"
        else:
            zone = "lower_extreme"
        
        # Distancia a las bandas
        distance_to_upper = (bb_upper - price) / price
        distance_to_lower = (price - bb_lower) / price
        distance_to_middle = abs(price - bb_middle) / price
        
        return {
            "position_percent": position_in_band,
            "zone": zone,
            "distance_to_upper": distance_to_upper,
            "distance_to_lower": distance_to_lower,
            "distance_to_middle": distance_to_middle,
            "band_width": band_width,
            "above_middle": price > bb_middle
        }
    
    def _detect_extremes(self, data: pd.DataFrame, bb_data: pd.DataFrame, 
                        position: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta condiciones extremas en las Bollinger Bands."""
        current_price = data['close'].iloc[-1]
        current_open = data['open'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        
        current_bb_upper = bb_data['bb_upper'].iloc[-1]
        current_bb_lower = bb_data['bb_lower'].iloc[-1]
        
        # Detectar toques y penetraciones extremas
        touches_upper = current_high >= current_bb_upper
        touches_lower = current_low <= current_bb_lower
        
        # Detectar penetraciones significativas
        penetrates_upper = current_close > current_bb_upper
        penetrates_lower = current_close < current_bb_lower
        
        # Detectar extremos basados en posición
        is_extreme_high = position["position_percent"] >= self.extreme_threshold
        is_extreme_low = position["position_percent"] <= (1 - self.extreme_threshold)
        
        # Detectar velas de rechazo en extremos
        rejection_upper = (
            touches_upper and 
            current_price < current_bb_upper and
            current_open < current_bb_upper
        )
        
        rejection_lower = (
            touches_lower and 
            current_price > current_bb_lower and
            current_open > current_bb_lower
        )
        
        # Calcular intensidad del extremo
        extreme_intensity = 0.0
        if is_extreme_high:
            extreme_intensity = position["position_percent"]
        elif is_extreme_low:
            extreme_intensity = 1 - position["position_percent"]
        
        return {
            "is_extreme_high": is_extreme_high,
            "is_extreme_low": is_extreme_low,
            "touches_upper": touches_upper,
            "touches_lower": touches_lower,
            "penetrates_upper": penetrates_upper,
            "penetrates_lower": penetrates_lower,
            "rejection_upper": rejection_upper,
            "rejection_lower": rejection_lower,
            "extreme_intensity": extreme_intensity
        }
    
    def _analyze_squeeze(self, bb_data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza condiciones de squeeze en Bollinger Bands."""
        if len(bb_data) < 20:
            return {"is_squeeze": False, "squeeze_intensity": 0.0}
        
        # Calcular ancho de banda actual y histórico
        current_width = bb_data['bb_upper'].iloc[-1] - bb_data['bb_lower'].iloc[-1]
        historical_widths = (bb_data['bb_upper'] - bb_data['bb_lower']).tail(20)
        avg_width = historical_widths.mean()
        min_width = historical_widths.min()
        
        # Detectar squeeze (ancho de banda menor al promedio)
        squeeze_ratio = current_width / avg_width if avg_width > 0 else 1.0
        is_squeeze = squeeze_ratio < 0.8  # 80% del promedio
        
        # Intensidad del squeeze
        squeeze_intensity = max(0, 1 - squeeze_ratio) if is_squeeze else 0.0
        
        # Detectar expansión
        is_expansion = squeeze_ratio > 1.2  # 120% del promedio
        
        return {
            "is_squeeze": is_squeeze,
            "is_expansion": is_expansion,
            "squeeze_ratio": squeeze_ratio,
            "squeeze_intensity": squeeze_intensity,
            "current_width": current_width,
            "avg_width": avg_width
        }
    
    def _generate_extremes_signal(self, position: Dict[str, Any], 
                                extremes: Dict[str, Any], 
                                squeeze: Dict[str, Any]) -> Dict[str, Any]:
        """Genera señal basada en análisis de extremos Bollinger."""
        signal = "WAIT"
        strength = 0.0
        
        # Señales de reversión en extremos
        if extremes["is_extreme_low"] and extremes["rejection_lower"]:
            signal = "BUY"
            strength = 0.8 * extremes["extreme_intensity"]
        elif extremes["is_extreme_high"] and extremes["rejection_upper"]:
            signal = "SELL"
            strength = 0.8 * extremes["extreme_intensity"]
        
        # Señales de breakout después de squeeze
        elif squeeze["is_squeeze"] and squeeze["squeeze_intensity"] > 0.5:
            if extremes["penetrates_upper"]:
                signal = "BUY"
                strength = 0.6 * squeeze["squeeze_intensity"]
            elif extremes["penetrates_lower"]:
                signal = "SELL"
                strength = 0.6 * squeeze["squeeze_intensity"]
        
        # Señales de continuación en tendencia
        elif position["zone"] == "upper" and not extremes["is_extreme_high"]:
            if position["above_middle"]:
                signal = "BUY"
                strength = 0.4
        elif position["zone"] == "lower" and not extremes["is_extreme_low"]:
            if not position["above_middle"]:
                signal = "SELL"
                strength = 0.4
        
        return {"signal": signal, "strength": strength}
    
    def _empty_result(self) -> Dict[str, Any]:
        """Retorna resultado vacío cuando no hay suficientes datos."""
        return {
            "algorithm": "BOLLINGER_EXTREMES",
            "signal": "WAIT",
            "strength": 0.0,
            "position": {"zone": "middle"},
            "extremes": {"is_extreme_high": False, "is_extreme_low": False},
            "squeeze": {"is_squeeze": False},
            "bb_values": {},
            "conditions": {
                "extreme_overbought": False,
                "extreme_oversold": False,
                "squeeze_active": False
            }
        }