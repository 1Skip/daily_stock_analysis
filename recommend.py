# -*- coding: utf-8 -*-
"""
===============================
股票推荐系统 - 独立入口
===============================

基于技术面扫描 + LLM深度分析的自动选股系统

使用方式:
    python recommend.py                      # 综合推荐
    python recommend.py --type breakout      # 突破推荐
    python recommend.py --type trend         # 趋势推荐
    python recommend.py --type oversold      # 超跌反弹
    python recommend.py --type hot           # 热点板块
    python recommend.py --top 10             # 推荐10只
    python recommend.py --watchlist stocks.txt  # 从自选扫描
    python recommend.py --notify             # 发送通知
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# 设置环境
from dotenv import load_dotenv
load_dotenv()

from src.config import setup_env
setup_env()

from src.recommender.stock_recommender import StockRecommender, RecommendationType
from src.logging_config import setup_logging
from src.notification import send_notifications

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='股票智能推荐系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python recommend.py                    # 综合推荐5只股票
  python recommend.py --type breakout    # 突破形态推荐
  python recommend.py --type trend       # 趋势跟随推荐
  python recommend.py --type oversold    # 超跌反弹推荐
  python recommend.py --type hot         # 热点板块推荐
  python recommend.py --top 10           # 推荐10只
  python recommend.py --scan 100         # 扫描100只候选
  python recommend.py --notify           # 发送推送通知
  python recommend.py --save             # 保存到文件
        '''
    )

    parser.add_argument(
        '--type',
        type=str,
        default='comprehensive',
        choices=['breakout', 'trend', 'oversold', 'hot', 'comprehensive'],
        help='推荐类型 (默认: comprehensive)'
    )

    parser.add_argument(
        '--top',
        type=int,
        default=5,
        help='推荐数量 (默认: 5)'
    )

    parser.add_argument(
        '--scan',
        type=int,
        default=50,
        help='扫描候选股数量 (默认: 50)'
    )

    parser.add_argument(
        '--watchlist',
        type=str,
        help='自选股票文件路径，每行一个代码'
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
        '--output',
        type=str,
        default='recommendations',
        help='输出文件名前缀 (默认: recommendations)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式'
    )

    parser.add_argument(
        '--llm',
        action='store_true',
        default=True,
        help='使用LLM深度分析 (默认启用)'
    )

    return parser.parse_args()


def load_watchlist(filepath: str) -> List[str]:
    """加载自选股票列表"""
    stocks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                code = line.strip()
                if code and not code.startswith('#'):
                    stocks.append(code)
        logger.info(f"从 {filepath} 加载了 {len(stocks)} 只自选股票")
    except Exception as e:
        logger.error(f"加载自选列表失败: {e}")
    return stocks


def init_llm_client():
    """初始化LLM客户端"""
    try:
        # 尝试导入LiteLLM
        from litellm import completion
        import os

        # 检查是否有API key
        api_key = (os.getenv("GEMINI_API_KEY") or
                  os.getenv("OPENAI_API_KEY") or
                  os.getenv("ANTHROPIC_API_KEY"))

        if not api_key:
            logger.warning("未配置LLM API Key，将使用基础技术分析")
            return None

        # 创建一个简单的LLM客户端包装器
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
        logger.warning("未安装LiteLLM，将使用基础技术分析")
        return None


async def run_recommendation(args: argparse.Namespace) -> None:
    """运行推荐流程"""

    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    logger.info("=" * 50)
    logger.info("股票智能推荐系统启动")
    logger.info("=" * 50)

    # 映射推荐类型
    type_mapping = {
        'breakout': RecommendationType.BREAKOUT,
        'trend': RecommendationType.TREND_FOLLOW,
        'oversold': RecommendationType.OVERSOLD_BOUNCE,
        'hot': RecommendationType.HOT_SECTOR,
        'comprehensive': RecommendationType.COMPREHENSIVE
    }

    rec_type = type_mapping.get(args.type, RecommendationType.COMPREHENSIVE)

    # 加载自选列表
    watchlist = None
    if args.watchlist:
        watchlist = load_watchlist(args.watchlist)
        if not watchlist:
            logger.error("自选列表为空，退出")
            return

    # 初始化LLM客户端
    llm_client = init_llm_client() if args.llm else None

    # 创建推荐引擎
    recommender = StockRecommender(llm_client=llm_client)

    logger.info(f"推荐类型: {args.type}")
    logger.info(f"推荐数量: {args.top}")
    logger.info(f"扫描数量: {args.scan}")

    try:
        # 执行推荐
        if rec_type == RecommendationType.HOT_SECTOR:
            # 热点推荐使用不同的入口
            recommendations = await recommender.recommend_from_hot(top_n=args.top)
        else:
            recommendations = await recommender.recommend(
                recommendation_type=rec_type,
                top_n=args.top,
                scan_limit=args.scan,
                watchlist=watchlist
            )

        if not recommendations:
            logger.warning("未找到符合条件的推荐股票")
            return

        # 生成报告
        report = await recommender.generate_report(recommendations)

        # 输出到控制台
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)

        # 保存到文件
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{args.output}_{timestamp}.txt"
            filepath = Path(filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info(f"报告已保存到: {filepath.absolute()}")

        # 发送通知
        if args.notify:
            await send_recommendation_notification(recommendations, report)

    except Exception as e:
        logger.error(f"推荐过程出错: {e}", exc_info=True)
        raise


async def send_recommendation_notification(recommendations, report: str) -> None:
    """发送推荐通知"""
    try:
        # 简化的通知内容
        summary = f"📈 今日股票推荐 ({len(recommendations)}只)\n\n"

        for i, rec in enumerate(recommendations[:3], 1):
            emoji = "🟢" if rec.confidence == "高" else "🟡"
            summary += f"{emoji} {rec.name}({rec.symbol})\n"
            summary += f"   目标: ¥{rec.target_price:.2f} 止损: ¥{rec.stop_loss:.2f}\n"
            summary += f"   {rec.reasoning[:30]}...\n\n"

        # 调用通知系统
        await send_notifications(summary, "recommendation")
        logger.info("通知已发送")

    except Exception as e:
        logger.error(f"发送通知失败: {e}")


def main():
    """主入口"""
    args = parse_arguments()

    try:
        asyncio.run(run_recommendation(args))
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
