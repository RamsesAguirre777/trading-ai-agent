"""Technical Strategy Algorithm - Cerebro de decisión final."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .ema_rotation import EMARotationAlgorithm
from .swing_points import SwingPointsAlgorithm
from .break_points import BreakPointsAlgorithm
from .candle_structure import CandleStructureAlgorithm
from .micro_levels import MicroLevelsAlgorithm
from .bollinger_extremes import BollingerExtremesAlgorithm


class TechnicalStrategyAlgorithm:
    """Estrategia Técnica Completa - Cerebro de decisión final que integra los 7 algoritmos."""
    
    def __init__(self):
        # Inicializar los 6 algoritmos anteriores
        self.ema_rotation = EMARotationAlgorithm()
        self.swing_points = SwingPointsAlgorithm()
        self.break_points = BreakPointsAlgorithm()
        self.candle_structure = CandleStructureAlgorithm()
        self.micro_levels = MicroLevelsAlgorithm()
        self.bollinger_extremes = BollingerExtremesAlgorithm()
        
        # Configuración de pesos para cada algoritmo
        self.algorithm_weights = {
            "ema_rotation": 0.2,
            "swing_points": 0.15,
            "break_points": 0.2,
            "candle_structure": 0.15,
            "micro_levels": 0.15,
            "bollinger_extremes": 0.15
        }
        
        # Umbrales para señales
        self.supreme_threshold = 0.9  # 90% para SUPREME
        self.strong_threshold = 0.7   # 70% para STRONG
        self.buy_sell_threshold = 0.6 # 60% para BUY/SELL
    
    def analyze(self, data: pd.DataFrame, multi_timeframe_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Análisis técnico completo con los 7 algoritmos.
        
        Args:
            data: DataFrame principal con OHLCV data
            multi_timeframe_data: Dict con data de múltiples timeframes
            
        Returns:
            Dict con resultado del análisis completo
        """
        if len(data) < 50:  # Mínimo de datos requerido
            return self._empty_result()
        
        # Ejecutar todos los algoritmos
        algorithm_results = self._run_all_algorithms(data, multi_timeframe_data)
        
        # Calcular score de confluencia
        confluence_score = self._calculate_confluence_score(algorithm_results)
        
        # Detectar condiciones extremas
        extreme_conditions = self._detect_extreme_conditions(algorithm_results)
        
        # Generar señal final
        final_signal = self._generate_final_signal(
            confluence_score, extreme_conditions, algorithm_results
        )
        
        # Contar condiciones cumplidas (para el sistema 7/7)
        conditions_met = self._count_conditions_met(algorithm_results)
        
        return {
            "algorithm": "TECHNICAL_STRATEGY",
            "signal": final_signal["signal"],
            "strength": final_signal["strength"],
            "confidence": final_signal["confidence"],
            "signal_type": final_signal["signal_type"],
            "confluence_score": confluence_score,
            "conditions_met": conditions_met,
            "extreme_conditions": extreme_conditions,
            "algorithm_results": algorithm_results,
            "justification": self._generate_justification(algorithm_results, final_signal),
            "technical_score": f"{conditions_met['total']}/7",
            "supreme_signal": final_signal["signal_type"] == "SUPREME"
        }
    
    def _run_all_algorithms(self, data: pd.DataFrame, 
                           multi_timeframe_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Ejecuta todos los algoritmos y recopila resultados."""
        results = {}
        
        # EMA Rotation + MACD
        try:
            results["ema_rotation"] = self.ema_rotation.analyze(data)
        except Exception as e:
            results["ema_rotation"] = {"signal": "WAIT", "strength": 0.0, "error": str(e)}
        
        # Swing Points Dinámicos
        try:
            results["swing_points"] = self.swing_points.analyze(data)
        except Exception as e:
            results["swing_points"] = {"signal": "WAIT", "strength": 0.0, "error": str(e)}
        
        # Break Points Multi-Timeframe
        try:
            if multi_timeframe_data:
                results["break_points"] = self.break_points.analyze(multi_timeframe_data)
            else:
                # Usar solo el timeframe actual
                results["break_points"] = self.break_points.analyze({"current": data})
        except Exception as e:
            results["break_points"] = {"signal": "WAIT", "strength": 0.0, "error": str(e)}
        
        # Estructura de Velas
        try:
            results["candle_structure"] = self.candle_structure.analyze(data)
        except Exception as e:
            results["candle_structure"] = {"signal": "WAIT", "strength": 0.0, "error": str(e)}
        
        # Niveles Micro
        try:
            results["micro_levels"] = self.micro_levels.analyze(data)
        except Exception as e:
            results["micro_levels"] = {"signal": "WAIT", "strength": 0.0, "error": str(e)}
        
        # Extremos Bollinger
        try:
            results["bollinger_extremes"] = self.bollinger_extremes.analyze(data)
        except Exception as e:
            results["bollinger_extremes"] = {"signal": "WAIT", "strength": 0.0, "error": str(e)}
        
        return results
    
    def _calculate_confluence_score(self, algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula score de confluencia ponderado."""
        total_bullish_score = 0.0
        total_bearish_score = 0.0
        total_weight = 0.0
        
        signal_counts = {"BUY": 0, "SELL": 0, "WAIT": 0}
        
        for algo_name, result in algorithm_results.items():
            if "signal" in result and "strength" in result:
                weight = self.algorithm_weights.get(algo_name, 0.1)
                signal = result["signal"]
                strength = result["strength"]
                
                signal_counts[signal] += 1
                
                if signal == "BUY":
                    total_bullish_score += strength * weight
                elif signal == "SELL":
                    total_bearish_score += strength * weight
                
                total_weight += weight
        
        # Normalizar scores
        if total_weight > 0:
            bullish_score = total_bullish_score / total_weight
            bearish_score = total_bearish_score / total_weight
        else:
            bullish_score = bearish_score = 0.0
        
        # Score final
        final_score = bullish_score - bearish_score  # -1 a +1
        
        return {
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "final_score": final_score,
            "signal_counts": signal_counts,
            "dominant_signal": max(signal_counts, key=signal_counts.get)
        }
    
    def _detect_extreme_conditions(self, algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta condiciones extremas de Bollinger para SUPREME signals."""
        extreme_conditions = {
            "bollinger_extreme_high": False,
            "bollinger_extreme_low": False,
            "extreme_detected": False
        }
        
        bb_result = algorithm_results.get("bollinger_extremes", {})
        if "extremes" in bb_result:
            extremes = bb_result["extremes"]
            extreme_conditions["bollinger_extreme_high"] = extremes.get("is_extreme_high", False)
            extreme_conditions["bollinger_extreme_low"] = extremes.get("is_extreme_low", False)
            extreme_conditions["extreme_detected"] = (
                extreme_conditions["bollinger_extreme_high"] or 
                extreme_conditions["bollinger_extreme_low"]
            )
        
        return extreme_conditions
    
    def _count_conditions_met(self, algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cuenta las condiciones técnicas cumplidas (sistema 7/7)."""
        conditions = {
            "ema_rotation_bullish": False,
            "swing_points_bullish": False,
            "break_points_bullish": False,
            "candle_structure_bullish": False,
            "micro_levels_bullish": False,
            "bollinger_position_good": False,
            "volume_confirmation": False  # Placeholder para confirmación de volumen
        }
        
        # Verificar cada algoritmo
        for algo_name, result in algorithm_results.items():
            signal = result.get("signal", "WAIT")
            strength = result.get("strength", 0.0)
            
            if signal in ["BUY", "SELL"] and strength > 0.5:
                if algo_name == "ema_rotation":
                    conditions["ema_rotation_bullish"] = signal == "BUY"
                elif algo_name == "swing_points":
                    conditions["swing_points_bullish"] = signal == "BUY"
                elif algo_name == "break_points":
                    conditions["break_points_bullish"] = signal == "BUY"
                elif algo_name == "candle_structure":
                    conditions["candle_structure_bullish"] = signal == "BUY"
                elif algo_name == "micro_levels":
                    conditions["micro_levels_bullish"] = signal == "BUY"
                elif algo_name == "bollinger_extremes":
                    conditions["bollinger_position_good"] = True
        
        # Contar total
        total_met = sum(1 for condition in conditions.values() if condition)
        
        return {
            "conditions": conditions,
            "total": total_met,
            "percentage": total_met / 7.0
        }
    
    def _generate_final_signal(self, confluence_score: Dict[str, Any], 
                              extreme_conditions: Dict[str, Any], 
                              algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera la señal final basada en todos los análisis."""
        final_score = confluence_score["final_score"]
        signal_counts = confluence_score["signal_counts"]
        
        # Determinar tipo de señal
        signal_type = "WAIT"
        signal = "WAIT"
        strength = abs(final_score)
        confidence = 0.0
        
        # Lógica SUPREME: Technical completo + Extremo
        conditions_met = self._count_conditions_met(algorithm_results)
        if (conditions_met["total"] >= 6 and 
            extreme_conditions["extreme_detected"] and 
            abs(final_score) > self.supreme_threshold):
            signal_type = "SUPREME"
            signal = "BUY" if final_score > 0 else "SELL"
            confidence = 0.95
        
        # Lógica STRONG: 3+ algoritmos en confluencia
        elif (signal_counts["BUY"] >= 4 or signal_counts["SELL"] >= 4) and abs(final_score) > self.strong_threshold:
            signal_type = "STRONG"
            signal = "BUY" if final_score > 0 else "SELL"
            confidence = 0.8
        
        # Lógica BUY/SELL: Score de confluencia > 0.6
        elif abs(final_score) > self.buy_sell_threshold:
            signal_type = "REGULAR"
            signal = "BUY" if final_score > 0 else "SELL"
            confidence = 0.6
        
        return {
            "signal": signal,
            "signal_type": signal_type,
            "strength": strength,
            "confidence": confidence
        }
    
    def _generate_justification(self, algorithm_results: Dict[str, Any], 
                               final_signal: Dict[str, Any]) -> List[str]:
        """Genera justificación automática para la decisión."""
        justification = []
        
        # Agregar justificación por algoritmo
        for algo_name, result in algorithm_results.items():
            if result.get("signal") != "WAIT" and result.get("strength", 0) > 0.5:
                signal = result["signal"]
                strength = result["strength"]
                justification.append(
                    f"{algo_name.upper()}: {signal} (strength: {strength:.2f})"
                )
        
        # Agregar razón del tipo de señal
        signal_type = final_signal["signal_type"]
        if signal_type == "SUPREME":
            justification.append("SUPREME SIGNAL: All technical conditions + extreme detected")
        elif signal_type == "STRONG":
            justification.append("STRONG SIGNAL: Multiple algorithm confluence")
        elif signal_type == "REGULAR":
            justification.append("REGULAR SIGNAL: Confluence score above threshold")
        
        return justification
    
    def _empty_result(self) -> Dict[str, Any]:
        """Retorna resultado vacío cuando no hay suficientes datos."""
        return {
            "algorithm": "TECHNICAL_STRATEGY",
            "signal": "WAIT",
            "strength": 0.0,
            "confidence": 0.0,
            "signal_type": "INSUFFICIENT_DATA",
            "confluence_score": {"final_score": 0.0},
            "conditions_met": {"total": 0, "percentage": 0.0},
            "extreme_conditions": {"extreme_detected": False},
            "algorithm_results": {},
            "justification": ["Insufficient data for analysis"],
            "technical_score": "0/7",
            "supreme_signal": False
        }