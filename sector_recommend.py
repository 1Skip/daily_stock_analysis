# -*- coding: utf-8 -*-
"""
===============================
板块概念股票推荐 - 独立入口
===============================

基于板块概念 + 技术面扫描 + LLM深度分析的选股系统

使用方式:
    python sector_recommend.py                      # 从所有板块推荐
    python sector_recommend.py --sector apple       # 苹果概念推荐
    python sector_recommend.py --sector power       # 电力板块推荐
    python sector_recommend.py --top 3              # 每板块推荐3只
    python sector_recommend.py --notify             # 发送通知
"""

import os
import sys
import argparse
import asyncio
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from src.config import setup_env
setup_env()

from src.recommender.stock_recommender import StockRecommender, RecommendationType
from src.logging_config import setup_logging
from src.notification import NotificationService

logger = logging.getLogger(__name__)


def load_sectors_config() -> Dict:
    """加载板块配置"""
    config_path = Path(__file__).parent / "sectors_config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载板块配置失败: {e}")
        return {"sectors": {}}


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='板块概念股票推荐系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python sector_recommend.py                    # 所有板块各推荐5只
  python sector_recommend.py --sector apple     # 仅苹果概念
  python sector_recommend.py --sector tesla     # 仅特斯拉概念
  python sector_recommend.py --sector power     # 仅电力板块
  python sector_recommend.py --sector computing # 仅算力租赁
  python sector_recommend.py --top 3            # 每板块推荐3只
  python sector_recommend.py --notify           # 发送推送通知
        '''
    )

    parser.add_argument(
        '--sector',
        type=str,
        choices=['apple', 'tesla', 'power', 'computing', 'all'],
        default='all',
        help='板块选择 (默认: all)'
    )

    parser.add_argument(
        '--top',
        type=int,
        default=5,
        help='每板块推荐数量 (默认: 5)'
    )

    parser.add_argument(
        '--notify',
        action='store_true',
        help='发送推送通知'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='保存报告到文件'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式'
    )

    return parser.parse_args()


def get_sector_stocks(sector_key: str, config: Dict) -> tuple:
    """获取板块名称和股票列表"""
    sector_map = {
        'apple': 'apple_concept',
        'tesla': 'tesla_concept',
        'power': 'power',
        'computing': 'computing_rental'
    }

    if sector_key == 'all':
        return None, []

    sector_id = sector_map.get(sector_key)
    if not sector_id or sector_id not in config['sectors']:
        return None, []

    sector = config['sectors'][sector_id]
    return sector['name'], sector['stocks']


async def recommend_from_sector(
    sector_name: str,
    stocks: List[str],
    top_n: int,
    recommender: StockRecommender
) -> List[Dict]:
    """从指定板块中推荐股票"""
    logger.info(f"开始分析板块: {sector_name} ({len(stocks)}只股票)")

    recommendations = await recommender.recommend(
        recommendation_type=RecommendationType.COMPREHENSIVE,
        top_n=top_n,
        scan_limit=len(stocks),
        watchlist=stocks
    )

    return recommendations


async def run_sector_recommendation(args: argparse.Namespace) -> None:
    """运行板块推荐"""

    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    logger.info("=" * 50)
    logger.info("板块概念股票推荐系统")
    logger.info("=" * 50)

    # 加载板块配置
    config = load_sectors_config()
    if not config['sectors']:
        logger.error("板块配置加载失败")
        return

    # 确定要分析的板块
    sectors_to_analyze = []
    if args.sector == 'all':
        sector_list = [
            ('apple_concept', '苹果概念'),
            ('tesla_concept', '特斯拉概念'),
            ('power', '电力'),
            ('computing_rental', '算力租赁')
        ]
        for sector_id, sector_name in sector_list:
            if sector_id in config['sectors']:
                stocks = config['sectors'][sector_id]['stocks']
                sectors_to_analyze.append((sector_name, stocks))
    else:
        sector_name, stocks = get_sector_stocks(args.sector, config)
        if sector_name:
            sectors_to_analyze.append((sector_name, stocks))

    if not sectors_to_analyze:
        logger.error("没有找到可分析的板块")
        return

    # 初始化推荐引擎
    recommender = StockRecommender(llm_client=init_llm_client())

    # 存储所有推荐结果
    all_recommendations = {}

    # 逐个板块分析
    for sector_name, stocks in sectors_to_analyze:
        print(f"\n{'='*60}")
        print(f"📊 正在分析: {sector_name}")
        print(f"{'='*60}")

        try:
            recommendations = await recommend_from_sector(
                sector_name, stocks, args.top, recommender
            )

            if recommendations:
                all_recommendations[sector_name] = recommendations
                print(f"✅ {sector_name}: 推荐 {len(recommendations)} 只股票")
                for rec in recommendations:
                    print(f"   • {rec.name}({rec.symbol}) - 评分:{rec.score:.1f}")
            else:
                print(f"⚠️ {sector_name}: 未找到符合条件的股票")

        except Exception as e:
            logger.error(f"分析 {sector_name} 失败: {e}")

    # 生成完整报告
    if all_recommendations:
        report = generate_sector_report(all_recommendations)
        print("\n" + report)

        # 保存报告
        if args.save:
            await save_report(report)

        # 发送通知
        if args.notify:
            await send_sector_notification(all_recommendations, report)
    else:
        print("\n⚠️ 所有板块均未找到推荐股票")


def init_llm_client():
    """初始化LLM客户端"""
    try:
        from litellm import completion

        api_key = (os.getenv("GEMINI_API_KEY") or
                  os.getenv("OPENAI_API_KEY") or
                  os.getenv("ANTHROPIC_API_KEY"))

        if not api_key:
            logger.warning("未配置LLM API Key，将使用基础技术分析")
            return None

        class LiteLLMClient:
            def __init__(self):
                self.model = self._select_model()

            def _select_model(self):
                if os.getenv("GEMINI_API_KEY"):
                    return "gemini/gemini-1.5-flash"
                elif os.getenv("OPENAI_API_KEY"):
                    return "gpt-4o-mini"
                elif os.getenv("ANTHROPIC_API_KEY"):
                    return "claude-3-haiku-20240307"
                return "gemini/gemini-1.5-flash"

            async def generate(self, prompt: str) -> str:
                try:
                    response = completion(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"LLM调用失败: {e}")
                    return "{}"

        return LiteLLMClient()

    except ImportError:
        logger.warning("未安装LiteLLM")
        return None


def generate_sector_report(all_recommendations: Dict) -> str:
    """生成板块推荐报告"""
    lines = []
    lines.append("\n" + "="*60)
    lines.append("📈 板块概念股票推荐报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("="*60)

    total_stocks = sum(len(recs) for recs in all_recommendations.values())
    lines.append(f"\n共分析 {len(all_recommendations)} 个板块，推荐 {total_stocks} 只股票\n")

    for sector_name, recommendations in all_recommendations.items():
        lines.append(f"\n{'─'*60}")
        lines.append(f"🔸 {sector_name} (推荐{len(recommendations)}只)")
        lines.append(f"{'─'*60}")

        for i, rec in enumerate(recommendations, 1):
            emoji = "🟢" if rec.confidence == "高" else "🟡" if rec.confidence == "中" else "⚪"
            lines.append(f"\n{emoji} #{i} {rec.name}({rec.symbol})")
            lines.append(f"   当前价: ¥{rec.current_price:.2f}")
            if rec.target_price:
                change = ((rec.target_price - rec.current_price) / rec.current_price) * 100
                lines.append(f"   目标价: ¥{rec.target_price:.2f} ({change:+.1f}%)")
            if rec.stop_loss:
                lines.append(f"   止损价: ¥{rec.stop_loss:.2f}")
            lines.append(f"   技术评分: {rec.score:.1f}")
            lines.append(f"   置信度: {rec.confidence}")
            lines.append(f"   推荐理由: {rec.reasoning[:50]}...")

    lines.append("\n" + "="*60)
    lines.append("⚠️ 风险提示: 以上推荐仅供参考，不构成投资建议")
    lines.append("="*60)

    return "\n".join(lines)


async def save_report(report: str):
    """保存报告到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sector_recommendations_{timestamp}.txt"
    filepath = Path(filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"报告已保存: {filepath.absolute()}")


async def send_sector_notification(all_recommendations: Dict, report: str):
    """发送板块推荐通知"""
    try:
        # 生成简洁的通知内容
        summary_lines = [f"📈 板块概念推荐 ({datetime.now().strftime('%m-%d')})\n"]

        for sector_name, recommendations in all_recommendations.items():
            summary_lines.append(f"\n🔸 {sector_name}")
            for rec in recommendations[:3]:  # 每板块最多显示3只
                emoji = "🟢" if rec.confidence == "高" else "🟡"
                price_str = f"¥{rec.current_price:.2f}" if rec.current_price > 0 else ""
                # 推荐理由取前80字，避免推送太长
                reason = rec.reasoning or ""
                if reason:
                    reason = reason[:80] + "…" if len(reason) > 80 else reason
                line = f"   {emoji} {rec.name}({rec.symbol}) {price_str}"
                if reason:
                    line += f"\n      💡 {reason}"
                summary_lines.append(line)

        summary = "\n".join(summary_lines)

        # 使用 NotificationService 发送
        service = NotificationService()
        if service.is_available():
            success = service.send(summary)
            if success:
                logger.info("通知已发送")
            else:
                logger.warning("通知发送失败")
        else:
            logger.warning("未配置通知渠道，无法发送通知")

    except Exception as e:
        logger.error(f"发送通知失败: {e}")


def main():
    """主入口"""
    args = parse_arguments()

    try:
        asyncio.run(run_sector_recommendation(args))
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
