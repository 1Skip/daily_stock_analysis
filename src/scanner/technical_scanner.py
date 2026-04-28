"""技术面扫描器 - 基于技术指标筛选潜力股票"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """扫描结果"""
    symbol: str
    name: str
    price: float
    score: float
    signals: Dict[str, Any]
    reason: str


class TechnicalScanner:
    """技术面扫描器

    支持多种筛选策略：
    - 突破筛选：放量突破、形态突破
    - 趋势筛选：均线多头排列、MACD金叉
    - 超跌反弹：RSI超卖、缩量企稳
    - 资金流向：主力净流入、大单占比
    """

    def __init__(self, data_provider=None):
        self.data_provider = data_provider
        self.logger = logging.getLogger(self.__class__.__name__)

    async def scan_a_shares(
        self,
        strategy: str = "breakout",
        top_n: int = 20,
        min_volume: float = 1000000,  # 最小成交额（万元）
        min_market_cap: float = 50,   # 最小市值（亿元）
    ) -> List[ScanResult]:
        """扫描A股潜力股票

        Args:
            strategy: 筛选策略 (breakout/trend/oversold/flow)
            top_n: 返回前N个结果
            min_volume: 最小日成交额（万元）
            min_market_cap: 最小市值（亿元）

        Returns:
            List[ScanResult]: 扫描结果列表
        """
        try:
            # 获取A股全市场数据
            symbols = await self._get_a_share_symbols()
            self.logger.info(f"开始扫描 {len(symbols)} 只A股，策略: {strategy}")

            # 批量获取数据并筛选
            results = []
            batch_size = 50

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                batch_results = await self._analyze_batch(batch, strategy, min_volume, min_market_cap)
                results.extend(batch_results)
                await asyncio.sleep(0.5)  # 避免请求过快

            # 按分数排序，取前N
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_n]

        except Exception as e:
            self.logger.error(f"扫描A股失败: {e}")
            return []

    async def scan_watchlist(
        self,
        symbols: List[str],
        strategy: str = "breakout"
    ) -> List[ScanResult]:
        """扫描自选股票池 - 使用实时行情批量获取"""
        self.logger.info(f"开始扫描 {len(symbols)} 只自选股票")

        try:
            results = await self._scan_with_realtime_data(symbols)
        except Exception as e:
            self.logger.warning(f"实时数据扫描失败: {e}，回退到K线扫描")
            results = []
            for symbol in symbols:
                try:
                    result = await self._analyze_symbol(symbol, strategy)
                    if result and result.score >= 50:
                        results.append(result)
                except Exception:
                    continue

        self.logger.info(f"扫描完成: {len(results)}/{len(symbols)} 通过")
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    async def _scan_with_realtime_data(self, symbols: List[str]) -> List[ScanResult]:
        """使用批量实时行情数据扫描（更可靠）"""
        import akshare as ak

        symbol_set = set(symbols)
        results = []

        try:
            self.logger.info("获取A股实时行情...")
            df = ak.stock_zh_a_spot_em()
            self.logger.info(f"获取到 {len(df)} 只股票实时数据")

            # 过滤只保留目标股票
            df = df[df['代码'].isin(symbol_set)]
            self.logger.info(f"匹配到 {len(df)} 只目标股票")

            for _, row in df.iterrows():
                try:
                    code = row['代码']
                    name = row['名称']
                    price = float(row['最新价'])
                    change_pct = float(row['涨跌幅'])
                    volume_ratio = float(row.get('量比', 1))
                    turnover = float(row.get('换手率', 0))
                    volume = float(row.get('成交量', 0))
                    amount = float(row.get('成交额', 0))

                    if pd.isna(price) or price <= 0:
                        continue

                    # 过滤ST/退市
                    if 'ST' in str(name) or '退' in str(name):
                        continue

                    # 只保留主板（过滤创业板300/301、科创板688/689、北交所4/8）
                    if not self._is_main_board(code):
                        continue

                    # 基于实时数据评分
                    score = 60  # 基础分
                    signals = {}
                    reasons = []

                    # 涨幅适中（不跌且不太大）
                    if 0.5 < change_pct < 5:
                        score += 10
                        reasons.append(f"温和上涨{change_pct:+.2f}%")
                    elif 1 < change_pct:
                        score += 5
                        reasons.append(f"上涨{change_pct:+.2f}%")
                    elif change_pct < -5:
                        score -= 10
                        reasons.append(f"跌幅较大{change_pct:.2f}%")
                    elif change_pct < -7:
                        score -= 15  # 跌幅过大，可能有雷

                    # 量比（>1表示放量）
                    if volume_ratio > 1.5:
                        score += 10
                        signals['volume_surge'] = True
                        reasons.append(f"量比{volume_ratio:.1f}倍")
                    elif volume_ratio > 1.0:
                        score += 5

                    # 换手率适中
                    if 2 < turnover < 15:
                        score += 10
                        reasons.append(f"换手率{turnover:.1f}%")
                    elif turnover > 15:
                        score -= 5  # 换手率过高可能有风险

                    # 振幅（>3%说明股性活跃）
                    amplitude = float(row.get('振幅', 0))
                    if 3 < amplitude < 10:
                        score += 5
                        signals['active'] = True
                        reasons.append(f"振幅{amplitude:.1f}%")

                    # PE估值：<0亏损扣分，0-30合理加分
                    pe = float(row.get('市盈率-动态', 0)) if pd.notna(row.get('市盈率-动态', 0)) else 0
                    if 0 < pe < 20:
                        score += 8
                        reasons.append(f"PE{pe:.1f}低估")
                    elif 0 < pe < 50:
                        score += 3
                    elif pe < 0:
                        score -= 5  # 亏损股
                        signals['loss'] = True

                    # 总市值过滤（<50亿扣分）
                    market_cap = float(row.get('总市值', 0)) if pd.notna(row.get('总市值', 0)) else 0
                    if market_cap > 5e10:  # >500亿，大盘股稳定
                        score += 5
                        reasons.append(f"市值{market_cap/1e8:.0f}亿")
                    elif market_cap < 5e9:  # <50亿，小盘股风险
                        score -= 5

                    # 代码加名称用于识别
                    reason_str = '; '.join(reasons) if reasons else f"当前价¥{price:.2f}"

                    result = ScanResult(
                        symbol=code,
                        name=name,
                        price=price,
                        score=score,
                        signals=signals,
                        reason=reason_str
                    )

                    if score >= 50:
                        results.append(result)
                        self.logger.info(f"✓ {code} {name} 评分:{score:.0f} ¥{price:.2f}")
                    else:
                        self.logger.info(f"✗ {code} {name} 评分:{score:.0f}")

                except Exception as e:
                    self.logger.warning(f"处理 {row.get('代码', '?')} 失败: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"获取实时行情失败: {e}")

        return results

    async def _get_a_share_symbols(self) -> List[str]:
        """获取A股全市场代码列表"""
        try:
            # 使用akshare获取A股列表
            import akshare as ak
            stocks = ak.stock_zh_a_spot_em()
            # 过滤ST、退市、北交所
            stocks = stocks[
                ~stocks['名称'].str.contains('ST|退', na=False) &
                ~stocks['代码'].str.startswith('8', na=False) &
                ~stocks['代码'].str.startswith('4', na=False)
            ]
            return stocks['代码'].tolist()[:500]  # 限制数量，先取前500只
        except Exception as e:
            self.logger.error(f"获取A股列表失败: {e}")
            return []

    async def _analyze_batch(
        self,
        symbols: List[str],
        strategy: str,
        min_volume: float,
        min_market_cap: float
    ) -> List[ScanResult]:
        """批量分析股票"""
        results = []

        for symbol in symbols:
            try:
                result = await self._analyze_symbol(symbol, strategy)
                if result and result.score > 50:
                    # 过滤成交量和市值
                    if self._pass_filters(result, min_volume, min_market_cap):
                        results.append(result)
            except Exception as e:
                continue

        return results

    async def _analyze_symbol(self, symbol: str, strategy: str) -> Optional[ScanResult]:
        """分析单只股票"""
        try:
            # 获取K线数据
            df = await self._get_kline_data(symbol, days=60)
            if df is None:
                self.logger.info(f"扫描 {symbol}: K线数据获取失败")
                return None
            if len(df) < 30:
                self.logger.info(f"扫描 {symbol}: K线数据不足 ({len(df)}天)")
                return None

            # 计算技术指标
            df = self._calculate_indicators(df)

            # 根据策略评分
            if strategy == "breakout":
                return self._score_breakout(symbol, df)
            elif strategy == "trend":
                return self._score_trend(symbol, df)
            elif strategy == "oversold":
                return self._score_oversold(symbol, df)
            elif strategy == "flow":
                return self._score_flow(symbol, df)
            else:
                return self._score_comprehensive(symbol, df)

        except Exception as e:
            self.logger.warning(f"分析 {symbol} 出错: {e}")
            return None

    async def _get_kline_data(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取K线数据"""
        try:
            import akshare as ak

            # 判断市场
            if symbol.startswith('6'):
                code = f"{symbol}.SH"
            else:
                code = f"{symbol}.SZ"

            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
            if df is not None and len(df) > 0:
                df = df.tail(days)
                try:
                    df.columns = ['date', 'open', 'close', 'high', 'low', 'volume',
                                 'amount', 'amplitude', 'pct_change', 'change', 'turnover']
                except Exception:
                    pass  # 如果列名不匹配，使用原始列名
                self.logger.debug(f"获取 {symbol} K线数据: {len(df)} 天")
                return df
            self.logger.debug(f"获取 {symbol} K线数据: 为空")
            return None
        except Exception as e:
            self.logger.warning(f"获取 {symbol} K线数据失败: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['boll_mid'] = df['close'].rolling(20).mean()
        df['boll_std'] = df['close'].rolling(20).std()
        df['boll_up'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_down'] = df['boll_mid'] - 2 * df['boll_std']

        # 成交量指标
        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()

        return df

    def _score_breakout(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """突破策略评分"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        score = 50
        signals = {}
        reasons = []

        # 1. 价格突破均线
        if latest['close'] > latest['ma20'] and prev['close'] <= prev['ma20']:
            score += 15
            signals['ma_breakout'] = True
            reasons.append("突破20日均线")

        # 2. 放量
        if latest['volume'] > latest['vol_ma5'] * 1.5:
            score += 10
            signals['volume_surge'] = True
            reasons.append(f"放量{latest['volume']/latest['vol_ma5']:.1f}倍")

        # 3. MACD金叉
        if latest['macd_hist'] > 0 and prev['macd_hist'] <= 0:
            score += 15
            signals['macd_golden'] = True
            reasons.append("MACD金叉")

        # 4. 均线多头排列
        if latest['ma5'] > latest['ma10'] > latest['ma20']:
            score += 10
            signals['ma_bullish'] = True
            reasons.append("均线多头排列")

        # 5. 突破近期高点
        recent_high = df['high'].tail(20).max()
        if latest['close'] > recent_high * 0.98 and latest['close'] < prev['close'] * 1.05:
            score += 10
            signals['near_high'] = True
            reasons.append("接近20日高点")

        if score <= 50 or len(reasons) < 2:
            return None

        return ScanResult(
            symbol=symbol,
            name=self._get_stock_name(symbol),
            price=float(latest['close']),
            score=min(score, 100),
            signals=signals,
            reason="、".join(reasons)
        )

    def _score_trend(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """趋势策略评分"""
        latest = df.iloc[-1]
        score = 50
        signals = {}
        reasons = []

        # 均线多头排列
        if latest['ma5'] > latest['ma10'] > latest['ma20'] > latest['ma60']:
            score += 20
            signals['strong_trend'] = True
            reasons.append("均线完全多头排列")

        # 价格在均线上方
        if latest['close'] > latest['ma5']:
            score += 10
            signals['above_ma5'] = True
            reasons.append("价格在5日均线上方")

        # MACD在零轴上方
        if latest['macd'] > 0:
            score += 10
            signals['macd_positive'] = True
            reasons.append("MACD零轴上方")

        # 趋势强度（斜率）
        ma20_slope = (latest['ma20'] - df.iloc[-5]['ma20']) / 5
        if ma20_slope > 0:
            score += min(ma20_slope * 100, 10)
            signals['trend_up'] = True

        if score < 50:
            return None

        return ScanResult(
            symbol=symbol,
            name=self._get_stock_name(symbol),
            price=float(latest['close']),
            score=min(score, 100),
            signals=signals,
            reason="、".join(reasons)
        )

    def _score_oversold(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """超跌反弹策略评分"""
        latest = df.iloc[-1]
        score = 50
        signals = {}
        reasons = []

        # RSI超卖
        if latest['rsi'] < 30:
            score += 20
            signals['rsi_oversold'] = True
            reasons.append(f"RSI超卖({latest['rsi']:.1f})")

        # 价格跌破下轨
        if latest['close'] < latest['boll_down']:
            score += 15
            signals['below_boll'] = True
            reasons.append("跌破布林带下轨")

        # 缩量
        if latest['volume'] < latest['vol_ma20'] * 0.7:
            score += 10
            signals['low_volume'] = True
            reasons.append("缩量整理")

        # 出现止跌信号（锤子线等简化版）
        if latest['close'] > latest['open'] and latest['low'] < prev['close'] * 0.98:
            score += 10
            signals['bounce_signal'] = True
            reasons.append("出现反弹信号")

        if score < 50:
            return None

        return ScanResult(
            symbol=symbol,
            name=self._get_stock_name(symbol),
            price=float(latest['close']),
            score=min(score, 100),
            signals=signals,
            reason="、".join(reasons)
        )

    def _score_flow(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """资金流向策略（简化版，需要外部数据）"""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        score = 50
        signals = {}
        reasons = []

        # 放量上涨
        if latest['pct_change'] > 2 and latest['volume'] > latest['vol_ma20'] * 1.3:
            score += 20
            signals['volume_price_up'] = True
            reasons.append("放量上涨")

        # 连阳
        last_3_days = df.tail(3)
        if all(last_3_days['close'] > last_3_days['open']):
            score += 15
            signals['three_red'] = True
            reasons.append("三连阳")

        if score < 50:
            return None

        return ScanResult(
            symbol=symbol,
            name=self._get_stock_name(symbol),
            price=float(latest['close']),
            score=min(score, 100),
            signals=signals,
            reason="、".join(reasons)
        )

    def _score_comprehensive(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """综合评分"""
        latest = df.iloc[-1]
        score = 50
        signals = {}
        reasons = []

        # 综合多种信号
        if latest['close'] > latest['ma20']:
            score += 10
            reasons.append("站上20日线")

        if latest['macd'] > latest['macd_signal']:
            score += 10
            reasons.append("MACD金叉")

        if latest['rsi'] > 50 and latest['rsi'] < 70:
            score += 10
            reasons.append("RSI健康")

        if latest['volume'] > latest['vol_ma5']:
            score += 10
            reasons.append("放量")

        if score < 50:
            return None

        return ScanResult(
            symbol=symbol,
            name=self._get_stock_name(symbol),
            price=float(latest['close']),
            score=min(score, 100),
            signals=signals,
            reason="、".join(reasons)
        )

    def _pass_filters(self, result: ScanResult, min_volume: float, min_market_cap: float) -> bool:
        """过滤条件检查"""
        # 这里简化处理，实际需要根据市值和成交额数据
        return True

    @staticmethod
    def _is_main_board(code: str) -> bool:
        """判断是否主板股票（过滤创业板、科创板、北交所）"""
        if not code or len(code) < 3:
            return False
        prefix = code[:3]
        # 创业板: 300-301, 科创板: 688-689, 北交所: 4xx, 8xx
        if prefix in ('300', '301', '688', '689'):
            return False
        if prefix[0] in ('4', '8'):
            return False
        return True

    def _get_stock_name(self, symbol: str) -> str:
        """获取股票名称"""
        try:
            import akshare as ak
            stocks = ak.stock_zh_a_spot_em()
            name = stocks[stocks['代码'] == symbol]['名称'].values
            return name[0] if len(name) > 0 else symbol
        except:
            return symbol
