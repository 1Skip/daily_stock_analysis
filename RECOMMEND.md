# 股票推荐功能使用指南

本项目新增智能股票推荐功能，基于技术面扫描 + LLM 深度分析，自动筛选潜力股票。

## 功能特点

- **多策略推荐**：突破、趋势、超跌反弹、热点板块、综合推荐
- **LLM 深度分析**：自动生成目标价、止损位、风险提示
- **多渠道推送**：支持微信、飞书、Telegram、邮件等
- **零成本自动化**：GitHub Actions 定时运行

---

## 快速开始

### 方式一：本地运行

```bash
# 1. 确保依赖已安装
pip install -r requirements.txt

# 2. 配置 API Key（用于 LLM 分析）
export GEMINI_API_KEY="your_api_key"

# 3. 运行推荐
python recommend.py                    # 综合推荐5只
python recommend.py --type breakout    # 突破形态推荐
python recommend.py --type trend       # 趋势推荐
python recommend.py --top 10           # 推荐10只
```

### 方式二：GitHub Actions 自动化

1. 启用 Actions：在你的 Fork 仓库点击 "Actions" → "I understand..."
2. 配置 Secrets：Settings → Secrets → 添加 `GEMINI_API_KEY`
3. 手动运行：Actions → "股票推荐" → "Run workflow"
4. 自动运行：每天北京时间 9:00 自动推送推荐

---

## 推荐类型说明

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `breakout` | 突破推荐 | 放量突破、形态突破、MACD金叉 |
| `trend` | 趋势跟随 | 均线多头排列、强势上涨趋势 |
| `oversold` | 超跌反弹 | RSI超卖、跌破布林带下轨 |
| `hot` | 热点板块 | 涨停股、资金流入、板块领涨 |
| `comprehensive` | 综合推荐 | 综合多种信号，平衡型 |

---

## 命令行参数

```
python recommend.py [选项]

选项：
  --type {breakout,trend,oversold,hot,comprehensive}
                        推荐类型 (默认: comprehensive)
  --top N               推荐数量 (默认: 5)
  --scan N              扫描候选股数量 (默认: 50)
  --watchlist FILE      自选股票文件路径
  --notify              发送推送通知
  --save                保存报告到文件
  --output PREFIX       输出文件名前缀
  --debug               调试模式
```

---

## 自选股票推荐

创建自选股票文件 `my_stocks.txt`：

```
600519
300750
000001
# 注释行会被忽略
```

运行推荐：
```bash
python recommend.py --watchlist my_stocks.txt --type trend
```

---

## 输出示例

```
📈 股票推荐报告
生成时间: 2026-01-27 09:00
共推荐 5 只股票

🟢 #1 贵州茅台(600519)
   类型: comprehensive | 置信度: 高
   当前价: ¥1688.00
   目标价: ¥1823.04 (+8.0%)
   止损价: ¥1603.60
   推荐理由: 突破20日均线、放量、MACD金叉，技术面显示强势突破信号
   建议持仓: 短线(1-2周)

🟡 #2 宁德时代(300750)
   类型: comprehensive | 置信度: 中
   当前价: ¥198.50
   目标价: ¥214.38 (+8.0%)
   止损价: ¥188.58
   推荐理由: 均线多头排列、站上5日线，趋势向好
   建议持仓: 短线(1-2周)
```

---

## 配置 Secrets（GitHub Actions）

必需配置：
- `GEMINI_API_KEY`：Google Gemini API 密钥（免费额度足够）

可选通知渠道（至少配置一个才能收到推送）：
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`
- `FEISHU_WEBHOOK_URL`
- `WECHAT_CORP_ID` 系列
- `DINGTALK_WEBHOOK`
- `EMAIL_SMTP_SERVER` 系列

---

## 如何获取 API Key

### Gemini（推荐）
1. 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 登录 Google 账号
3. 点击 "Create API Key"
4. 复制密钥到 GitHub Secrets

---

## 技术实现

```
recommend.py              # 主入口
src/scanner/              # 扫描器模块
  technical_scanner.py    # 技术面筛选
  market_scanner.py       # 市场热点扫描
src/recommender/          # 推荐引擎模块
  stock_recommender.py    # LLM推荐逻辑
```

---

## 注意事项

1. **风险提示**：推荐结果仅供参考，不构成投资建议
2. **数据延迟**：A股数据有 15 分钟延迟
3. **扫描范围**：默认扫描前500只活跃股票，全市场扫描较耗时
4. **LLM 限制**：免费 API 有速率限制，大批量扫描请控制并发

---

## 自定义开发

修改扫描策略：
- 编辑 `src/scanner/technical_scanner.py` 中的评分逻辑
- 添加新的技术指标或过滤条件

修改 LLM 提示词：
- 编辑 `src/recommender/stock_recommender.py` 中的 `_build_analysis_prompt`

添加新的推荐类型：
- 在 `RecommendationType` 枚举中添加新类型
- 在 `TechnicalScanner` 中实现对应的评分函数
