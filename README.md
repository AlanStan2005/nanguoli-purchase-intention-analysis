# 地标农产品南果梨购买意愿与营销策略分析

本仓库整理南果梨购买意愿与营销策略分析代码。项目先通过 Python 爬取电商平台
评论并进行 SnowNLP 情感分析、TF-IDF 词频统计和词云输入表构建；再结合东北三省
43 个行政区三阶段 PPS 抽样问卷，完成数据清洗、信效度检验和消费者画像构建；
最后使用 K-means 与 Logistic 回归识别购买意愿差异，并形成质量追溯、品牌标准化
和渠道营销策略建议。

## 代码结构

- `src/review_pipeline.py`：评论清洗、情感分析、TF-IDF 高频词。
- `src/survey_sampling.py`：PPS 抽样、问卷字段整理、Cronbach alpha。
- `src/questionnaire_quality.py`：KMO、近似 Bartlett 检验、量表得分构建。
- `src/modeling.py`：K-means 消费者分群与 Logistic 回归。
- `src/strategy_rules.py`：由模型结果映射到营销策略建议。
- `report/nanguoli_market_research_report.pdf`：市调项目公开报告。
- `docs/source_design/`：问卷设计、语义差别量表和 K-means 分析说明。
- `docs/SAMPLING_AND_MODELING.md`：抽样、问卷、文本分析和建模流程说明。
