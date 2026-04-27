"""市场扫描器 - 追踪热点板块和强势股"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HotStock:
    """热门股票"""
    symbol: str
    name: str
    sector: str
    change_pct: float
    volume_ratio: float
    reason: str


@dataclass
class HotSector:
    """热门板块"""
    name: str
    change_pct: float
    leading_stocks: List[str]
    avg_turnover: float


class MarketScanner:
    """市场扫描器

    功能：
    - 追踪涨幅榜、资金流向
    - 识别热门板块和题材
    - 筛选强势股
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    async def get_hot_sectors(self) -> List[HotSector]:
        """获取热门板块"""
        try:
            import akshare as ak

            # 获取行业板块涨幅
            sectors = ak.stock_sector_spot()

            # 按涨幅排序
            sectors = sectors.sort_values('涨跌幅', ascending=False).head(10)

            results = []
            for _, row in sectors.iterrows():
                sector = HotSector(
                    name=row['板块名称'],
                    change_pct=float(row['涨跌幅']),
                    leading_stocks=[],  # 需要单独获取
                    avg_turnover=float(row.get('成交额', 0))
                )
                results.append(sector)

            return results

        except Exception as e:
            self.logger.error(f"获取热门板块失败: {e}")
            return []

    async def get_hot_stocks(self, limit: int = 20) -> List[HotStock]:
        """获取热门股票（涨幅榜+资金流入）"""
        try:
            import akshare as ak

            # 获取涨幅榜
            gainers = ak.stock_zh_a_spot_em()
            gainers = gainers.sort_values('涨跌幅', ascending=False).head(limit)

            results = []
            for _, row in gainers.iterrows():
                stock = HotStock(
                    symbol=row['代码'],
                    name=row['名称'],
                    sector=row.get('所属行业', '未知'),
                    change_pct=float(row['涨跌幅']),
                    volume_ratio=float(row.get('量比', 1.0)),
                    reason=f"涨幅{row['涨跌幅']:.1f}%"
                )
                results.append(stock)

            return results

        except Exception as e:
            self.logger.error(f"获取热门股票失败: {e}")
            return []

    async def get_limit_up_stocks(self) -> List[Dict[str, Any]]:
        """获取涨停股票列表"""
        try:
            import akshare as ak

            # 获取涨停股
            limit_up = ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d"))

            results = []
            for _, row in limit_up.iterrows():
                results.append({
                    'symbol': row['代码'],
                    'name': row['名称'],
                    'first_time': row.get('首次封板时间', ''),
                    'last_time': row.get('最后封板时间', ''),
                    'open_count': int(row.get('开板数', 0)),
                    'volume': float(row.get('成交额', 0)),
                    'reason': row.get('涨停原因', '未知')
                })

            return results

        except Exception as e:
            self.logger.error(f"获取涨停股票失败: {e}")
            return []

    async def get_money_flow(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取资金流入股票"""
        try:
            import akshare as ak

            # 获取个股资金流向
            flow = ak.stock_individual_fund_flow_rank()
            flow = flow.head(limit)

            results = []
            for _, row in flow.iterrows():
                results.append({
                    'symbol': row['代码'],
                    'name': row['名称'],
                    'main_inflow': float(row.get('主力净流入', 0)),
                    'main_pct': float(row.get('主力净流入占比', 0)),
                    'change_pct': float(row.get('涨跌幅', 0))
                })

            return results

        except Exception as e:
            self.logger.error(f"获取资金流向失败: {e}")
            return []

    async def get_sector_rotation(self) -> Dict[str, Any]:
        """获取板块轮动信息"""
        try:
            sectors = await self.get_hot_sectors()

            # 分析领涨板块
            leading_sectors = [s for s in sectors if s.change_pct > 2]

            return {
                'leading_sectors': leading_sectors,
                'hot_theme': sectors[0].name if sectors else None,
                'market_style': '成长' if any('科技' in s.name for s in sectors[:3]) else '价值'
            }

        except Exception as e:
            self.logger.error(f"获取板块轮动失败: {e}")
            return {}

    async def scan_concept_leaders(self, concept: str) -> List[HotStock]:
        """扫描某一概念股的龙头股

        Args:
            concept: 概念名称，如"人工智能"

        Returns:
            该概念的龙头股列表
        """
        try:
            import akshare as ak

            # 获取概念股列表
            stocks = ak.stock_board_concept_cons_em(symbol=concept)

            results = []
            for _, row in stocks.iterrows():
                results.append(HotStock(
                    symbol=row['代码'],
                    name=row['名称'],
                    sector=concept,
                    change_pct=float(row.get('涨跌幅', 0)),
                    volume_ratio=float(row.get('量比', 1.0)),
                    reason=f"{concept}概念股"
                ))

            # 按涨幅排序
            results.sort(key=lambda x: x.change_pct, reverse=True)
            return results[:10]

        except Exception as e:
            self.logger.error(f"扫描{concept}概念股失败: {e}")
            return []
