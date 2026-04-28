"""股票推荐引擎 - LLM驱动的智能推荐系统"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import sys
from pathlib import Path
# 将 src 目录加入 Python 路径，确保 scanner 模块可导入
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
from scanner.technical_scanner import TechnicalScanner, ScanResult
from scanner.market_scanner import MarketScanner, HotStock

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """推荐类型"""
    BREAKOUT = "breakout"          # 突破推荐
    TREND_FOLLOW = "trend"         # 趋势跟随
    OVERSOLD_BOUNCE = "oversold"   # 超跌反弹
    HOT_SECTOR = "hot_sector"      # 热点板块
    COMPREHENSIVE = "comprehensive" # 综合推荐


@dataclass
class RecommendationResult:
    """推荐结果"""
    symbol: str
    name: str
    current_price: float
    recommendation_type: RecommendationType
    score: float  # 技术评分
    ai_score: float  # AI评分
    confidence: str  # 高/中/低
    reasoning: str  # 推荐理由
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)
    holding_period: str = "短线(1-2周)"
    analysis_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))


class StockRecommender:
    """股票推荐引擎

    工作流程：
    1. 扫描器筛选候选股票
    2. LLM深度分析每只候选股
    3. 综合打分排序
    4. 生成推荐报告
    """

    def __init__(self, llm_client=None):
        self.scanner = TechnicalScanner()
        self.market_scanner = MarketScanner()
        self.llm_client = llm_client
        self.logger = logging.getLogger(self.__class__.__name__)

    async def recommend(
        self,
        recommendation_type: RecommendationType = RecommendationType.COMPREHENSIVE,
        top_n: int = 5,
        scan_limit: int = 50,
        watchlist: Optional[List[str]] = None
    ) -> List[RecommendationResult]:
        """生成股票推荐

        Args:
            recommendation_type: 推荐类型
            top_n: 推荐数量
            scan_limit: 扫描股票数量上限
            watchlist: 自选股票池（可选）

        Returns:
            List[RecommendationResult]: 推荐列表
        """
        self.logger.info(f"开始生成推荐，类型: {recommendation_type.value}, 数量: {top_n}")

        # 1. 获取候选股票
        candidates = await self._get_candidates(
            recommendation_type,
            scan_limit,
            watchlist
        )

        if not candidates:
            self.logger.warning("没有找到候选股票")
            return []

        self.logger.info(f"筛选出 {len(candidates)} 只候选股票")

        # 2. LLM深度分析（如果有LLM客户端）
        if self.llm_client:
            analyzed_results = await self._analyze_with_llm(candidates, recommendation_type)
        else:
            # 无LLM时，基于技术评分生成结果
            analyzed_results = self._generate_basic_results(candidates, recommendation_type)

        # 3. 排序并取前N
        analyzed_results.sort(key=lambda x: (x.ai_score + x.score) / 2, reverse=True)

        return analyzed_results[:top_n]

    async def recommend_from_hot(self, top_n: int = 5) -> List[RecommendationResult]:
        """基于市场热点推荐"""
        try:
            # 获取热门板块
            hot_sectors = await self.market_scanner.get_hot_sectors()
            hot_stocks = await self.market_scanner.get_hot_stocks(limit=30)

            self.logger.info(f"发现 {len(hot_sectors)} 个热门板块，{len(hot_stocks)} 只热门股")

            # 转换为候选
            candidates = []
            for stock in hot_stocks[:20]:
                candidates.append(ScanResult(
                    symbol=stock.symbol,
                    name=stock.name,
                    price=0,  # 需要重新获取
                    score=50 + stock.change_pct * 2,  # 基于涨幅评分
                    signals={'hot_sector': True, 'sector': stock.sector},
                    reason=stock.reason
                ))

            # 重新获取价格并分析
            valid_candidates = []
            for c in candidates:
                price = await self._get_current_price(c.symbol)
                if price:
                    c.price = price
                    valid_candidates.append(c)

            if self.llm_client:
                return await self._analyze_with_llm(valid_candidates, RecommendationType.HOT_SECTOR)[:top_n]
            else:
                return self._generate_basic_results(valid_candidates, RecommendationType.HOT_SECTOR)[:top_n]

        except Exception as e:
            self.logger.error(f"热点推荐失败: {e}")
            return []

    async def _get_candidates(
        self,
        rec_type: RecommendationType,
        limit: int,
        watchlist: Optional[List[str]]
    ) -> List[ScanResult]:
        """获取候选股票"""

        # 映射推荐类型到扫描策略
        strategy_map = {
            RecommendationType.BREAKOUT: "breakout",
            RecommendationType.TREND_FOLLOW: "trend",
            RecommendationType.OVERSOLD_BOUNCE: "oversold",
            RecommendationType.COMPREHENSIVE: "comprehensive"
        }

        strategy = strategy_map.get(rec_type, "comprehensive")

        if watchlist:
            # 从自选池扫描
            return await self.scanner.scan_watchlist(watchlist, strategy)
        else:
            # 全市场扫描（限制数量）
            return await self.scanner.scan_a_shares(
                strategy=strategy,
                top_n=limit
            )

    async def _analyze_with_llm(
        self,
        candidates: List[ScanResult],
        rec_type: RecommendationType
    ) -> List[RecommendationResult]:
        """使用LLM深度分析候选股票"""
        results = []

        for candidate in candidates:
            try:
                result = await self._analyze_single_stock(candidate, rec_type)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"分析 {candidate.symbol} 失败: {e}")
                continue

        return results

    async def _analyze_single_stock(
        self,
        candidate: ScanResult,
        rec_type: RecommendationType
    ) -> Optional[RecommendationResult]:
        """分析单只股票"""

        # 构建分析提示
        prompt = self._build_analysis_prompt(candidate, rec_type)

        try:
            # 调用LLM
            response = await self._call_llm(prompt)

            # 解析响应
            analysis = self._parse_llm_response(response)

            # 计算置信度
            confidence = self._calculate_confidence(
                candidate.score,
                analysis.get('ai_score', 50)
            )

            return RecommendationResult(
                symbol=candidate.symbol,
                name=candidate.name,
                current_price=candidate.price,
                recommendation_type=rec_type,
                score=candidate.score,
                ai_score=analysis.get('ai_score', 50),
                confidence=confidence,
                reasoning=analysis.get('reasoning') or candidate.reason,
                target_price=analysis.get('target_price'),
                stop_loss=analysis.get('stop_loss'),
                support_levels=analysis.get('support_levels', []),
                resistance_levels=analysis.get('resistance_levels', []),
                risk_factors=analysis.get('risk_factors', []),
                catalysts=analysis.get('catalysts', []),
                holding_period=analysis.get('holding_period', '短线')
            )

        except Exception as e:
            self.logger.error(f"LLM分析 {candidate.symbol} 失败: {e}")
            return self._generate_fallback_result(candidate, rec_type)

    def _build_analysis_prompt(self, candidate: ScanResult, rec_type: RecommendationType) -> str:
        """构建LLM分析提示"""
        type_desc = {
            RecommendationType.BREAKOUT: "突破策略",
            RecommendationType.TREND_FOLLOW: "趋势跟随",
            RecommendationType.OVERSOLD_BOUNCE: "超跌反弹",
            RecommendationType.HOT_SECTOR: "热点板块",
            RecommendationType.COMPREHENSIVE: "综合分析"
        }

        prompt = f"""你是一位专业的股票分析师，请对以下股票进行深入分析：

股票信息：
- 代码: {candidate.symbol}
- 名称: {candidate.name}
- 当前价格: {candidate.price:.2f}
- 技术评分: {candidate.score}/100
- 技术信号: {json.dumps(candidate.signals, ensure_ascii=False)}
- 推荐理由: {candidate.reason}
- 分析类型: {type_desc.get(rec_type, '综合分析')}

请提供以下分析（以JSON格式返回）：
{{
    "ai_score": 70,  // AI评分(0-100)
    "reasoning": "详细的推荐理由，包括技术面和市场环境分析",
    "target_price": 12.50,  // 目标价
    "stop_loss": 10.20,  // 止损价
    "support_levels": [10.50, 10.00],  // 支撑位
    "resistance_levels": [12.00, 13.00],  // 压力位
    "risk_factors": ["风险1", "风险2"],  // 风险提示
    "catalysts": ["催化剂1", "催化剂2"],  // 潜在催化剂
    "holding_period": "短线/中线/长线"  // 建议持仓周期
}}

注意：
1. 基于技术分析信号给出客观评分
2. 目标价和止损价要合理，基于当前价格计算
3. 风险提示要全面
4. 只返回JSON格式，不要有其他文字
"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        if hasattr(self.llm_client, 'generate'):
            return await self.llm_client.generate(prompt)
        elif hasattr(self.llm_client, 'chat'):
            response = await self.llm_client.chat.completions.create(
                model="gemini-pro",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        else:
            # 模拟返回
            return json.dumps({
                "ai_score": 65,
                "reasoning": "技术面显示积极信号，建议关注",
                "target_price": 0,
                "stop_loss": 0,
                "support_levels": [],
                "resistance_levels": [],
                "risk_factors": ["市场波动风险"],
                "catalysts": ["业绩预期"],
                "holding_period": "短线"
            })

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 提取JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            return {}
        except Exception as e:
            self.logger.warning(f"解析LLM响应失败: {e}")
            return {}

    def _calculate_confidence(self, tech_score: float, ai_score: float) -> str:
        """计算置信度"""
        avg_score = (tech_score + ai_score) / 2
        if avg_score >= 80:
            return "高"
        elif avg_score >= 65:
            return "中"
        else:
            return "低"

    def _generate_basic_results(
        self,
        candidates: List[ScanResult],
        rec_type: RecommendationType
    ) -> List[RecommendationResult]:
        """无LLM时生成基础推荐结果"""
        results = []

        for c in candidates:
            # 基于技术评分计算目标价和止损价
            target = c.price * 1.08 if c.price > 0 else None
            stop_loss = c.price * 0.95 if c.price > 0 else None

            confidence = self._calculate_confidence(c.score, c.score)

            results.append(RecommendationResult(
                symbol=c.symbol,
                name=c.name,
                current_price=c.price,
                recommendation_type=rec_type,
                score=c.score,
                ai_score=c.score,
                confidence=confidence,
                reasoning=c.reason,
                target_price=target,
                stop_loss=stop_loss,
                support_levels=[c.price * 0.97, c.price * 0.95] if c.price > 0 else [],
                resistance_levels=[c.price * 1.05, c.price * 1.08] if c.price > 0 else [],
                risk_factors=["市场系统性风险", "个股基本面变化"],
                catalysts=["技术面突破", "市场情绪回暖"],
                holding_period="短线(1-2周)"
            ))

        return results

    def _generate_fallback_result(
        self,
        candidate: ScanResult,
        rec_type: RecommendationType
    ) -> RecommendationResult:
        """生成回退结果（LLM失败时）"""
        return RecommendationResult(
            symbol=candidate.symbol,
            name=candidate.name,
            current_price=candidate.price,
            recommendation_type=rec_type,
            score=candidate.score,
            ai_score=50,
            confidence="中",
            reasoning=candidate.reason,
            target_price=candidate.price * 1.08 if candidate.price > 0 else None,
            stop_loss=candidate.price * 0.95 if candidate.price > 0 else None,
            support_levels=[],
            resistance_levels=[],
            risk_factors=["分析数据不完整"],
            catalysts=[],
            holding_period="短线"
        )

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            price = df[df['代码'] == symbol]['最新价'].values
            return float(price[0]) if len(price) > 0 else None
        except:
            return None

    async def generate_report(self, recommendations: List[RecommendationResult]) -> str:
        """生成推荐报告"""
        if not recommendations:
            return "暂无推荐股票"

        report_lines = [
            "📈 股票推荐报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"共推荐 {len(recommendations)} 只股票",
            ""
        ]

        for i, rec in enumerate(recommendations, 1):
            emoji = "🟢" if rec.confidence == "高" else "🟡" if rec.confidence == "中" else "⚪"

            report_lines.extend([
                f"{emoji} #{i} {rec.name}({rec.symbol})",
                f"   类型: {rec.recommendation_type.value} | 置信度: {rec.confidence}",
                f"   当前价: ¥{rec.current_price:.2f}",
            ])

            if rec.target_price:
                upside = (rec.target_price / rec.current_price - 1) * 100
                report_lines.append(f"   目标价: ¥{rec.target_price:.2f} (+{upside:.1f}%)")

            if rec.stop_loss:
                report_lines.append(f"   止损价: ¥{rec.stop_loss:.2f}")

            report_lines.extend([
                f"   推荐理由: {rec.reasoning}",
                f"   建议持仓: {rec.holding_period}",
                ""
            ])

        return "\n".join(report_lines)
