# ANALIZER SYSTEM - Trading Multi-Timeframe Analysis Engine
# Integración: Alpaca API + MCPs + Metodología Ramses

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import talib

@dataclass
class TimeframeWeight:
    """Pesos para cada timeframe según metodología Ramses"""
    timeframe: str
    weight: float
    
class MultiTimeframeAnalyzer:
    """
    ANALIZER SYSTEM - Motor de análisis multi-timeframe
    Basado en metodología completa de Ramses
    """
    
    def __init__(self, alpaca_key: str, alpaca_secret: str, base_url: str):
        self.api = tradeapi.REST(alpaca_key, alpaca_secret, base_url)
        
        # Timeframes según metodología Ramses
        self.timeframes = [
            TimeframeWeight('1Day', 0.40),   # Contexto principal
            TimeframeWeight('1Hour', 0.20),  # Tendencia intermedia
            TimeframeWeight('30Min', 0.10),  # Setup entry/exit
            TimeframeWeight('15Min', 0.08),  # Confirmación
            TimeframeWeight('13Min', 0.06),  # Confirmación
            TimeframeWeight('10Min', 0.05),  # Confirmación
            TimeframeWeight('5Min', 0.04),   # Ejecución
            TimeframeWeight('3Min', 0.03),   # Ejecución
            TimeframeWeight('2Min', 0.02),   # Ejecución
            TimeframeWeight('1Min', 0.02),   # Ejecución precisa
        ]
        
        # EMAs según metodología Ramses
        self.ema_periods = [3, 9, 20, 50, 200]
        
    async def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Obtiene datos para todos los timeframes"""
        data = {}
        
        for tf in self.timeframes:
            try:
                # Convertir timeframe a formato Alpaca
                alpaca_tf = self._convert_timeframe(tf.timeframe)
                
                # Obtener datos históricos
                bars = self.api.get_bars(
                    symbol, 
                    alpaca_tf,
                    start=datetime.now() - timedelta(days=252),  # 1 año de datos
                    end=datetime.now(),
                    adjustment='raw'
                ).df
                
                data[tf.timeframe] = bars
                
            except Exception as e:
                print(f"Error obteniendo datos para {tf.timeframe}: {e}")
                
        return data
    
    def calculate_ema_system(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula sistema EMA completo según metodología Ramses"""
        for period in self.ema_periods:
            df[f'EMA_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # Cruces EMA según metodología
        df['EMA_3_9_cross'] = np.where(df['EMA_3'] > df['EMA_9'], 1, 
                                     np.where(df['EMA_3'] < df['EMA_9'], -1, 0))
        df['EMA_9_20_cross'] = np.where(df['EMA_9'] > df['EMA_20'], 1,
                                      np.where(df['EMA_9'] < df['EMA_20'], -1, 0))
        df['EMA_20_50_cross'] = np.where(df['EMA_20'] > df['EMA_50'], 1,
                                       np.where(df['EMA_20'] < df['EMA_50'], -1, 0))
        df['EMA_50_200_cross'] = np.where(df['EMA_50'] > df['EMA_200'], 1,
                                        np.where(df['EMA_50'] < df['EMA_200'], -1, 0))
        
        # ALGORITMO RAMSES 4: Estructura de Velas
        df = self.calculate_candle_structure(df)
        
        return df
    
    def calculate_candle_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ALGORITMO RAMSES 4: Análisis de Estructura de Velas
        Price Action puro - Timing perfecto de entradas y salidas
        """
        # Inicializar columnas
        df['bullish_entry'] = False
        df['bullish_exit'] = False 
        df['bearish_entry'] = False
        df['bearish_exit'] = False
        
        # Calcular estructura para cada vela (excepto la primera)
        for i in range(1, len(df)):
            # ENTRADA ALCISTA: high > high[1] AND low > low[1]
            df.loc[df.index[i], 'bullish_entry'] = (
                df['high'].iloc[i] > df['high'].iloc[i-1] and
                df['low'].iloc[i] > df['low'].iloc[i-1]
            )
            
            # SALIDA ALCISTA: high < high[1] OR low < low[1]
            df.loc[df.index[i], 'bullish_exit'] = (
                df['high'].iloc[i] < df['high'].iloc[i-1] or
                df['low'].iloc[i] < df['low'].iloc[i-1]
            )
            
            # ENTRADA BAJISTA: high < high[1] AND low < low[1]
            df.loc[df.index[i], 'bearish_entry'] = (
                df['high'].iloc[i] < df['high'].iloc[i-1] and
                df['low'].iloc[i] < df['low'].iloc[i-1]
            )
            
            # SALIDA BAJISTA: high > high[1] OR low > low[1]
            df.loc[df.index[i], 'bearish_exit'] = (
                df['high'].iloc[i] > df['high'].iloc[i-1] or
                df['low'].iloc[i] > df['low'].iloc[i-1]
            )
        
        # Señales estructurales consolidadas
        df['structure_signal'] = np.where(df['bullish_entry'], 1,
                                np.where(df['bearish_entry'], -1, 0))
        
        df['structure_exit'] = np.where(df['bullish_exit'], -1,
                               np.where(df['bearish_exit'], 1, 0))
        
        # CONFLUENCIA ESTRUCTURA + EMAs (GENIUS COMBINATION)
        df['structure_ema_confluence'] = np.where(
            (df['structure_signal'] == 1) & (df['EMA_3_9_cross'] == 1), 2,  # STRONG BUY
            np.where((df['structure_signal'] == -1) & (df['EMA_3_9_cross'] == -1), -2,  # STRONG SELL
                   np.where((df['structure_signal'] == 1), 1,  # WEAK BUY
                          np.where((df['structure_signal'] == -1), -1, 0))))  # WEAK SELL
        
        # Secuencias estructurales (fuerza de la estructura)
        df['bullish_sequence'] = 0
        df['bearish_sequence'] = 0
        
        bullish_count = 0
        bearish_count = 0
        
        for i in range(len(df)):
            if df['bullish_entry'].iloc[i]:
                bullish_count += 1
                bearish_count = 0
            elif df['bullish_exit'].iloc[i]:
                bullish_count = 0
                
            if df['bearish_entry'].iloc[i]:
                bearish_count += 1
                bullish_count = 0
            elif df['bearish_exit'].iloc[i]:
                bearish_count = 0
                
            df.loc[df.index[i], 'bullish_sequence'] = bullish_count
            df.loc[df.index[i], 'bearish_sequence'] = bearish_count
        
        # Structure Strength Score
        df['structure_strength'] = np.where(df['bullish_sequence'] > 0, 
                                          df['bullish_sequence'] / 10,  # Normalize
                                          -df['bearish_sequence'] / 10)
        
        return df
    
    def calculate_macd_system(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula MACD según metodología Ramses"""
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
        
        # Cruces MACD + EMA 3/9 según metodología
        df['MACD_cross'] = np.where(df['MACD'] > df['MACD_signal'], 1,
                                  np.where(df['MACD'] < df['MACD_signal'], -1, 0))
        
        # ALGORITMO RAMSES: Rotación EMA3/EMA9 + MACD Confirmación
        df['MACD_hist_prev'] = df['MACD_hist'].shift(1)
        df['ramses_rotation_signal'] = self.detectar_rotacion_ema_macd(df)
        
        return df
    
    def detectar_rotacion_ema_macd(self, df: pd.DataFrame) -> pd.Series:
        """
        ALGORITMO RAMSES: Detecta rotación EMA3/EMA9 + MACD confirmación
        Lógica original de Ramses integrada en ANALIZER SYSTEM
        """
        signals = []
        
        for i in range(len(df)):
            if i == 0:  # Skip first row (no previous data)
                signals.append(0)
                continue
                
            # Datos según algoritmo original
            data = {
                "ema3": df.iloc[i]['EMA_3'],
                "ema9": df.iloc[i]['EMA_9'], 
                "macd_hist_actual": df.iloc[i]['MACD_hist'],
                "macd_hist_prev": df.iloc[i-1]['MACD_hist'] if i > 0 else 0
            }
            
            # Lógica original de Ramses
            ema3 = data.get("ema3")
            ema9 = data.get("ema9")
            macd_hist_actual = data.get("macd_hist_actual")
            macd_hist_prev = data.get("macd_hist_prev")

            # Condiciones exactas del algoritmo original
            cruz_alcista = (ema3 > ema9 and 
                          macd_hist_prev < 0 and 
                          macd_hist_actual > 0)
            
            cruz_bajista = (ema3 < ema9 and 
                          macd_hist_prev > 0 and 
                          macd_hist_actual < 0)
            
            # Señal final
            if cruz_alcista:
                signals.append(1)   # BUY signal
            elif cruz_bajista:
                signals.append(-1)  # SELL signal
            else:
                signals.append(0)   # NEUTRAL
                
        return pd.Series(signals, index=df.index)
    
    def calculate_parabolic_sar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula Parabolic SAR"""
        df['SAR'] = talib.SAR(df['high'], df['low'])
        df['SAR_signal'] = np.where(df['close'] > df['SAR'], 1, -1)
        
        # ALGORITMO RAMSES 5: Niveles Micro de Velas
        df = self.calculate_micro_candle_levels(df)
        
        return df
    
    def calculate_micro_candle_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ALGORITMO RAMSES 5: Niveles Micro de Velas Individuales
        Análisis granular intra-vela para precisión quirúrgica
        """
        # Niveles matemáticos de cada vela
        df['correccion'] = (df['high'] + df['low']) / 2  # Midpoint perfecto
        df['resistencia_vela'] = df['high'] - (df['high'] - df['low']) * 0.33  # 33% desde high
        df['soporte_vela'] = df['low'] + (df['high'] - df['low']) * 0.33       # 33% desde low
        df['swing_vela'] = df['open']  # Punto de referencia inicial
        
        # Sesgo intra-vela basado en posición respecto a corrección
        df['sesgo_intra_vela'] = np.where(df['close'] > df['correccion'], 1,    # Alcista
                                        np.where(df['close'] < df['correccion'], -1, 0))  # Bajista
        
        # Proximidad a niveles micro (TRADING INTELLIGENCE)
        threshold = 0.002  # 0.2% de tolerancia
        
        df['near_correccion'] = np.where(
            abs(df['close'] - df['correccion']) / df['close'] <= threshold, 1, 0)
        
        df['near_resistencia_vela'] = np.where(
            abs(df['close'] - df['resistencia_vela']) / df['close'] <= threshold, 1, 0)
        
        df['near_soporte_vela'] = np.where(
            abs(df['close'] - df['soporte_vela']) / df['close'] <= threshold, 1, 0)
        
        # Respeto a niveles micro (validación de calidad)
        df['respeta_soporte'] = np.where(
            (df['low'] <= df['soporte_vela'] * 1.005) & 
            (df['close'] > df['soporte_vela']), 1, 0)
        
        df['respeta_resistencia'] = np.where(
            (df['high'] >= df['resistencia_vela'] * 0.995) & 
            (df['close'] < df['resistencia_vela']), 1, 0)
        
        # Fuerza de la vela basada en niveles
        df['vela_strength'] = (
            df['respeta_soporte'] * 0.3 +
            df['respeta_resistencia'] * 0.3 +
            df['near_correccion'] * 0.4
        )
        
        # CONFLUENCIA CON ALGORITMOS ANTERIORES (GENIUS INTEGRATION)
        
        # Estructura + Niveles micro = Timing perfecto
        df['estructura_micro_confluence'] = 0
        if 'structure_signal' in df.columns:
            df['estructura_micro_confluence'] = np.where(
                (df['structure_signal'] == 1) & (df['near_soporte_vela'] == 1), 2,    # PERFECT BUY
                np.where((df['structure_signal'] == -1) & (df['near_resistencia_vela'] == 1), -2,  # PERFECT SELL
                       np.where((df['structure_signal'] == 1) & (df['near_correccion'] == 1), 1,    # GOOD BUY
                              np.where((df['structure_signal'] == -1) & (df['near_correccion'] == 1), -1, 0))))  # GOOD SELL
        
        # Swing Points + Niveles micro = Confluencia suprema
        df['swing_micro_confluence'] = 0
        if 'near_support' in df.columns and 'near_resistance' in df.columns:
            df['swing_micro_confluence'] = np.where(
                (df['near_support'] == 1) & (df['near_soporte_vela'] == 1), 2,        # DOUBLE SUPPORT
                np.where((df['near_resistance'] == 1) & (df['near_resistencia_vela'] == 1), -2,    # DOUBLE RESISTANCE
                       np.where((df['near_support'] == 1) & (df['near_correccion'] == 1), 1,        # SUPPORT + MIDPOINT
                              np.where((df['near_resistance'] == 1) & (df['near_correccion'] == 1), -1, 0))))  # RESISTANCE + MIDPOINT
        
        # Niveles micro como stop loss dinámico
        df['stop_loss_micro'] = np.where(df['sesgo_intra_vela'] == 1, 
                                       df['soporte_vela'],     # Stop bajo soporte si alcista
                                       df['resistencia_vela'])  # Stop sobre resistencia si bajista
        
        # Take profit micro
        df['take_profit_micro'] = np.where(df['sesgo_intra_vela'] == 1,
                                         df['resistencia_vela'],  # TP en resistencia si alcista
                                         df['soporte_vela'])      # TP en soporte si bajista
        
        return df
    
    def calculate_bollinger_exponential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula Bollinger Bands Exponenciales (BBB, BBT)"""
        # Bollinger Bands tradicionales primero
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
        
        # Exponential Bollinger según metodología Ramses
        ema_base = talib.EMA(df['close'], timeperiod=20)
        std_dev = df['close'].rolling(20).std()
        
        df['BBT'] = ema_base + (2 * std_dev)  # Bollinger Band Top
        df['BBB'] = ema_base - (2 * std_dev)  # Bollinger Band Bottom
        
        # ALGORITMO RAMSES 2: Swing Points Dinámicos
        df = self.calculate_swing_points_dynamic(df)
        
        # ALGORITMO RAMSES 6: Extremos Bollinger Bands
        df = self.calculate_bollinger_extremes(df)
        
        # Señales BBB/BBT
        df['BB_signal'] = np.where(df['close'] > df['BBT'], -1,  # Sobrecompra
                                 np.where(df['close'] < df['BBB'], 1, 0))  # Sobreventa
        
        return df
    
    def calculate_bollinger_extremes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ALGORITMO RAMSES 6: Extremos Bollinger Bands
        Detección de condiciones extremas usando apertura de vela
        """
        # Extremos básicos según algoritmo original
        df['extremo_buy'] = df['open'] < df['BB_lower']     # Apertura bajo banda inferior
        df['extremo_sell'] = df['open'] > df['BB_upper']    # Apertura sobre banda superior
        
        # Fuerza del extremo (mejora del algoritmo original)
        df['extremo_strength_buy'] = np.where(df['extremo_buy'], 
                                            (df['BB_lower'] - df['open']) / df['open'] * 100, 0)
        
        df['extremo_strength_sell'] = np.where(df['extremo_sell'],
                                             (df['open'] - df['BB_upper']) / df['open'] * 100, 0)
        
        # Extremos con exponential Bollinger (BBB/BBT)
        df['extremo_exp_buy'] = df['open'] < df['BBB']      # Apertura bajo BBB
        df['extremo_exp_sell'] = df['open'] > df['BBT']     # Apertura sobre BBT
        
        # Extremos DOBLES (tradicional + exponential)
        df['extremo_double_buy'] = df['extremo_buy'] & df['extremo_exp_buy']
        df['extremo_double_sell'] = df['extremo_sell'] & df['extremo_exp_sell']
        
        # Señal consolidada de extremos
        df['extremo_signal'] = np.where(df['extremo_double_buy'], 2,      # STRONG BUY
                                      np.where(df['extremo_buy'], 1,       # BUY
                                             np.where(df['extremo_double_sell'], -2,  # STRONG SELL
                                                    np.where(df['extremo_sell'], -1, 0))))  # SELL
        
        # CONFLUENCIA CON ALGORITMOS ANTERIORES (GENIUS INTEGRATION)
        
        # Extremos + Break Point proximity
        df['extremo_bp_confluence'] = 0
        if 'near_correccion' in df.columns:  # From Algorithm 5
            df['extremo_bp_confluence'] = np.where(
                (df['extremo_signal'] > 0) & (df['near_correccion'] == 1), 
                df['extremo_signal'] + 1,  # Extra boost if near correction
                df['extremo_signal']
            )
        
        # Extremos + Estructura de velas
        df['extremo_estructura_confluence'] = 0
        if 'structure_signal' in df.columns:  # From Algorithm 4
            df['extremo_estructura_confluence'] = np.where(
                (df['extremo_signal'] > 0) & (df['structure_signal'] > 0), 
                3,  # SUPREME BUY: Extremo + Structure bullish
                np.where((df['extremo_signal'] < 0) & (df['structure_signal'] < 0),
                       -3,  # SUPREME SELL: Extremo + Structure bearish
                       df['extremo_signal']))
        
        # Extremos + Swing Points
        df['extremo_swing_confluence'] = 0
        if 'near_support' in df.columns and 'near_resistance' in df.columns:
            df['extremo_swing_confluence'] = np.where(
                (df['extremo_signal'] > 0) & (df['near_support'] == 1),
                4,  # ULTIMATE BUY: Extremo + Support level
                np.where((df['extremo_signal'] < 0) & (df['near_resistance'] == 1),
                       -4,  # ULTIMATE SELL: Extremo + Resistance level
                       df['extremo_signal']))
        
        # Recovery signals (reversión desde extremos)
        df['recovery_from_oversold'] = (
            (df['extremo_buy'].shift(1) == True) &  # Previous candle was oversold
            (df['close'] > df['BB_lower']) &         # Current close above lower band
            (df['close'] > df['open'])               # Current candle is green
        )
        
        df['recovery_from_overbought'] = (
            (df['extremo_sell'].shift(1) == True) &  # Previous candle was overbought
            (df['close'] < df['BB_upper']) &          # Current close below upper band
            (df['close'] < df['open'])                # Current candle is red
        )
        
        # Risk management basado en extremos
        df['stop_loss_extremo'] = np.where(
            df['extremo_signal'] > 0,
            df['BB_lower'] * 0.999,  # Stop slightly below lower band for longs
            df['BB_upper'] * 1.001   # Stop slightly above upper band for shorts
        )
        
        df['take_profit_extremo'] = np.where(
            df['extremo_signal'] > 0,
            df['BB_upper'],  # TP at upper band for longs
            df['BB_lower']   # TP at lower band for shorts
        )
        
        return df
    
    def calculate_swing_points_dynamic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ALGORITMO RAMSES 2: Swing Points Dinámicos
        Niveles de soporte/resistencia que se ajustan automáticamente a volatilidad
        """
        # Bollinger Band Width para volatilidad
        basis = df['close'].rolling(window=20).mean()
        dev = df['close'].rolling(window=20).std() * 2
        df['band_width'] = (basis + dev) - (basis - dev)
        
        # Swing Points dinámicos basados en EMA20 + volatilidad
        df['swingHigh33'] = df['EMA_20'] + df['band_width'] * 0.33
        df['swingHigh66'] = df['EMA_20'] + df['band_width'] * 0.66
        df['swingLow33'] = df['EMA_20'] - df['band_width'] * 0.33
        df['swingLow66'] = df['EMA_20'] - df['band_width'] * 0.66
        
        # Cruces EMA dinámicos (complementando algoritmo 1)
        df['crossUp_3_9'] = ((df['EMA_3'] > df['EMA_9']) & 
                            (df['EMA_3'].shift(1) <= df['EMA_9'].shift(1)))
        df['crossDown_3_9'] = ((df['EMA_3'] < df['EMA_9']) & 
                              (df['EMA_3'].shift(1) >= df['EMA_9'].shift(1)))
        
        # Cruces adicionales según metodología Ramses
        df['crossUp_9_20'] = ((df['EMA_9'] > df['EMA_20']) & 
                             (df['EMA_9'].shift(1) <= df['EMA_20'].shift(1)))
        df['crossDown_9_20'] = ((df['EMA_9'] < df['EMA_20']) & 
                               (df['EMA_9'].shift(1) >= df['EMA_20'].shift(1)))
        
        df['crossUp_20_50'] = ((df['EMA_20'] > df['EMA_50']) & 
                              (df['EMA_20'].shift(1) <= df['EMA_50'].shift(1)))
        df['crossDown_20_50'] = ((df['EMA_20'] < df['EMA_50']) & 
                                (df['EMA_20'].shift(1) >= df['EMA_50'].shift(1)))
        
        df['crossUp_50_200'] = ((df['EMA_50'] > df['EMA_200']) & 
                               (df['EMA_50'].shift(1) <= df['EMA_200'].shift(1)))
        df['crossDown_50_200'] = ((df['EMA_50'] < df['EMA_200']) & 
                                 (df['EMA_50'].shift(1) >= df['EMA_200'].shift(1)))
        
        # Señales de proximidad a swing points (TRADING INTELLIGENCE)
        df['near_resistance'] = np.where(
            (df['close'] >= df['swingHigh33'] * 0.98) & 
            (df['close'] <= df['swingHigh33'] * 1.02), 1, 0)
        
        df['near_support'] = np.where(
            (df['close'] <= df['swingLow33'] * 1.02) & 
            (df['close'] >= df['swingLow33'] * 0.98), 1, 0)
        
        # Confluencia swing points + cruces EMA (GENIUS COMBINATION)
        df['swing_ema_bullish'] = np.where(
            (df['crossUp_3_9'] == 1) & (df['near_support'] == 1), 1, 0)
        
        df['swing_ema_bearish'] = np.where(
            (df['crossDown_3_9'] == 1) & (df['near_resistance'] == 1), 1, 0)
        
        return df
    
    def calculate_confluence_score(self, multi_tf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calcula score de confluencia multi-timeframe según metodología Ramses
        Peso principal: Vela 1D (40%) + BREAK POINT como filtro supremo
        """
        confluence_scores = []
        
        # ALGORITMO RAMSES 3: Calcular Break Point dinámico
        break_point = self.calculate_break_point_dynamic(multi_tf_data)
        
        for i, tf in enumerate(self.timeframes):
            if tf.timeframe not in multi_tf_data:
                continue
                
            df = multi_tf_data[tf.timeframe]
            if df.empty:
                continue
            
            # Score por timeframe
            latest_row = df.iloc[-1]
            current_price = latest_row['close']
            
            # Señales individuales
            ema_score = (
                latest_row['EMA_3_9_cross'] * 0.3 +
                latest_row['EMA_9_20_cross'] * 0.25 +
                latest_row['EMA_20_50_cross'] * 0.25 +
                latest_row['EMA_50_200_cross'] * 0.2
            )
            
            # Algoritmo 1: Rotación EMA + MACD
            ramses_rotation_score = latest_row.get('ramses_rotation_signal', 0)
            
            # Algoritmo 2: Swing Points confluence
            swing_bullish = latest_row.get('swing_ema_bullish', 0)
            swing_bearish = latest_row.get('swing_ema_bearish', 0)
            swing_score = swing_bullish - swing_bearish
            
            # Algoritmo 4: Price Action Structure
            structure_confluence = latest_row.get('structure_ema_confluence', 0)
            structure_strength = latest_row.get('structure_strength', 0)
            
            # Algoritmo 5: Niveles Micro de Velas
            estructura_micro = latest_row.get('estructura_micro_confluence', 0)
            swing_micro = latest_row.get('swing_micro_confluence', 0)
            vela_strength = latest_row.get('vela_strength', 0)
            
            # Algoritmo 6: Extremos Bollinger Bands
            extremo_signal = latest_row.get('extremo_signal', 0)
            extremo_estructura = latest_row.get('extremo_estructura_confluence', 0)
            extremo_swing = latest_row.get('extremo_swing_confluence', 0)
            recovery_oversold = latest_row.get('recovery_from_oversold', False)
            recovery_overbought = latest_row.get('recovery_from_overbought', False)
            
            macd_score = latest_row.get('MACD_cross', 0)
            sar_score = latest_row['SAR_signal']
            bb_score = latest_row['BB_signal']
            
            # Score consolidado del timeframe con TODOS los algoritmos Ramses
            tf_score = (
                ema_score * 0.12 + 
                ramses_rotation_score * 0.18 +     # Algoritmo 1: EMA/MACD
                swing_score * 0.10 +               # Algoritmo 2: Swing Points
                structure_confluence * 0.15 +      # Algoritmo 4: Structure
                structure_strength * 0.06 +        # Algoritmo 4: Strength
                estructura_micro * 0.12 +          # Algoritmo 5: Estructura + Micro
                swing_micro * 0.08 +               # Algoritmo 5: Swing + Micro
                vela_strength * 0.04 +             # Algoritmo 5: Vela Quality
                extremo_signal * 0.08 +            # Algoritmo 6: Extremos
                extremo_estructura * 0.05 +        # Algoritmo 6: Extremo + Estructura
                extremo_swing * 0.03 +             # Algoritmo 6: Extremo + Swing
                macd_score * 0.02 + 
                sar_score * 0.005 + 
                bb_score * 0.005
            )
            
            # BONUS por recovery signals (oportunidades especiales)
            recovery_bonus = 0
            if recovery_oversold:
                recovery_bonus += 0.3  # Bonus por recovery from oversold
            if recovery_overbought:
                recovery_bonus -= 0.3  # Negative bonus (bearish) for recovery from overbought
                
            tf_score += recovery_bonus
            
            # ALGORITMO 3: Break Point Filter (CRÍTICO)
            bp_bias = 1 if current_price > break_point else -1
            bp_distance = abs(current_price - break_point) / break_point
            bp_proximity_bonus = max(0, 1 - (bp_distance * 100))  # Bonus si está cerca del BP
            
            # Score final con Break Point validation
            if (tf_score > 0 and bp_bias > 0) or (tf_score < 0 and bp_bias < 0):
                bp_validated_score = tf_score * (1 + bp_proximity_bonus * 0.5)
            else:
                bp_validated_score = tf_score * 0.3  # Penalizar si va contra BP
            
            # Aplicar peso del timeframe según metodología
            weighted_score = bp_validated_score * tf.weight
            
            confluence_scores.append({
                'timeframe': tf.timeframe,
                'score': tf_score,
                'bp_validated_score': bp_validated_score,
                'weighted_score': weighted_score,
                'weight': tf.weight,
                'break_point': break_point,
                'current_price': current_price,
                'bp_bias': bp_bias,
                'bp_distance_pct': bp_distance * 100,
                'timestamp': latest_row.name
            })
        
        return pd.DataFrame(confluence_scores)
    
    def calculate_break_point_dynamic(self, multi_tf_data: Dict[str, pd.DataFrame]) -> float:
        """
        ALGORITMO RAMSES 3: Break Point Dinámico Multi-Timeframe
        Núcleo técnico del sistema - Pivote central para todas las decisiones
        """
        ema3_list = []
        ema9_list = []
        weights = []
        
        # Recopilar EMAs de todos los timeframes con pesos
        for tf in self.timeframes:
            if tf.timeframe not in multi_tf_data:
                continue
                
            df = multi_tf_data[tf.timeframe]
            if df.empty:
                continue
            
            latest_row = df.iloc[-1]
            
            # Agregar EMAs con peso del timeframe
            ema3_list.append(latest_row['EMA_3'] * tf.weight)
            ema9_list.append(latest_row['EMA_9'] * tf.weight)
            weights.append(tf.weight)
        
        if not ema3_list or not ema9_list:
            return 0.0
        
        # Cálculo Break Point WEIGHTED según metodología Ramses
        total_weight = sum(weights)
        ema3_weighted_avg = sum(ema3_list) / total_weight
        ema9_weighted_avg = sum(ema9_list) / total_weight
        
        break_point = (ema3_weighted_avg + ema9_weighted_avg) / 2
        
        return round(break_point, 2)
    
    async def analyze_symbol(self, symbol: str) -> Dict:
        """Análisis completo de un símbolo según metodología ANALIZER"""
        print(f"🔍 Analizando {symbol} con metodología ANALIZER...")
        
        # 1. Obtener datos multi-timeframe
        multi_tf_data = await self.get_multi_timeframe_data(symbol)
        
        # 2. Calcular indicadores en cada timeframe
        for tf, df in multi_tf_data.items():
            df = self.calculate_ema_system(df)
            df = self.calculate_macd_system(df)
            df = self.calculate_parabolic_sar(df)
            df = self.calculate_bollinger_exponential(df)
            multi_tf_data[tf] = df
        
        # 3. Calcular confluencia multi-timeframe
        confluence_df = self.calculate_confluence_score(multi_tf_data)
        
        # 4. Score final según metodología Ramses
        final_score = confluence_df['weighted_score'].sum()
        
        # 5. Decisión de trading
        if final_score > 0.6:
            decision = "STRONG BUY"
        elif final_score > 0.3:
            decision = "BUY"
        elif final_score < -0.6:
            decision = "STRONG SELL"
        elif final_score < -0.3:
            decision = "SELL"
        else:
            decision = "NEUTRAL"
        
        return {
            'symbol': symbol,
            'final_score': final_score,
            'decision': decision,
            'confluence_breakdown': confluence_df.to_dict('records'),
            'timestamp': datetime.now(),
            'methodology': 'ANALIZER Multi-Timeframe Confluence'
        }
    
    def _convert_timeframe(self, tf: str) -> str:
        """Convierte timeframes a formato Alpaca"""
        conversion = {
            '1Day': '1Day',
            '1Hour': '1Hour', 
            '30Min': '30Min',
            '15Min': '15Min',
            '13Min': '15Min',  # Alpaca no tiene 13Min, usar 15Min
            '10Min': '15Min',  # Alpaca no tiene 10Min, usar 15Min
            '5Min': '5Min',
            '3Min': '5Min',    # Alpaca no tiene 3Min, usar 5Min
            '2Min': '1Min',    # Alpaca no tiene 2Min, usar 1Min
            '1Min': '1Min'
        }
        return conversion.get(tf, '1Hour')

# EJEMPLO DE USO
async def run_analizer_system():
    """Ejecuta el sistema ANALIZER completo"""
    
    # Configuración Alpaca (PAPER TRADING)
    analyzer = MultiTimeframeAnalyzer(
        alpaca_key="TU_ALPACA_KEY",
        alpaca_secret="TU_ALPACA_SECRET", 
        base_url="https://paper-api.alpaca.markets"  # Paper trading
    )
    
    # Símbolos a analizar
    symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
    
    print("🚀 INICIANDO ANALIZER SYSTEM - Metodología Multi-Timeframe")
    print("=" * 60)
    
    for symbol in symbols:
        try:
            analysis = await analyzer.analyze_symbol(symbol)
            
            print(f"\n📊 ANÁLISIS {symbol}:")
            print(f"Score Final: {analysis['final_score']:.3f}")
            print(f"Decisión: {analysis['decision']}")
            print(f"Timestamp: {analysis['timestamp']}")
            
            # Mostrar breakdown por timeframe
            print("\n🔍 Breakdown por Timeframe:")
            for tf_data in analysis['confluence_breakdown']:
                print(f"  {tf_data['timeframe']}: {tf_data['score']:.3f} "
                      f"(peso: {tf_data['weight']:.2f}) = {tf_data['weighted_score']:.3f}")
                
        except Exception as e:
            print(f"❌ Error analizando {symbol}: {e}")
    
    print("\n✅ ANÁLISIS COMPLETO FINALIZADO")

if __name__ == "__main__":
    # Ejecutar sistema ANALIZER
    asyncio.run(run_analizer_system())