# 求职岗位推荐 Agent

面向应届生、转行/跳槽者和异地求职者的可解释岗位推荐原型。系统把自由文本/表单画像解析、技能抽取、K-Means 用户分群、TF-IDF 内容召回、ItemCF/SVD 协同召回、逻辑回归排序、硬约束过滤和证据化解释组织成一个可追踪的 Agent 工作流。

> 本项目为《机器学习》课程期末项目。仓库包含代码、模型、演示截图和运行记录；课程报告按课程要求单独提交，不在本仓库中。

## 项目信息

- 课程：机器学习
- 学院/专业：电子电气工程学院 / 计算机科学与技术
- 作者：邓博文
- 课程教师：邱禧荷
- 完成日期：2026-11-25
- 课程报告：按课程要求单独提交，不在本仓库中

## 1. 解决什么问题

求职者在招聘平台上通常要面对数百条岗位逐条核对城市、薪资、学历、经验和技能：

- 关键词搜索要求用户先知道岗位名称；
- 热度/广告排序会忽略个人硬约束；
- 协同过滤对新用户冷启动不友好，且推荐理由不透明；
- 通用大模型容易编造岗位、薪资或技能，不适合直接作为高影响决策。

本系统把任务拆成“硬约束过滤 + 可解释排序 + 人工确认”：

1. 解析画像，缺失字段主动追问，非法输入拦截；
2. 用 K-Means 给出可解释的求职者分群；
3. 按城市、薪资、学历、经验做硬过滤；
4. 内容、协同、分群多路召回；
5. 逻辑回归融合内容/技能/约束/热度特征；
6. 输出 Top-10，并给出技能重合、约束满足、分群归属、模型贡献等证据；
7. 低置信度、无匹配、异常输入和可选 LLM 解释都有降级或人工复核；
8. 不自动投递，导出前需要用户确认。

## 2. 主要结果

| 方法 | P@5 | R@10 | NDCG@10 | MAP@10 | Coverage@10 | 约束违规率 |
|---|---:|---:|---:|---:|---:|---:|
| 热度 | 0.005 | 0.019 | 0.0118 | 0.005 | 0.007 | 0.851 |
| 内容 | 0.016 | 0.026 | 0.0217 | 0.008 | 0.699 | 0.932 |
| ItemCF | 0.092 | 0.136 | 0.1212 | 0.059 | 0.547 | 0.373 |
| SVD | 0.078 | 0.119 | 0.1066 | 0.054 | 0.208 | 0.335 |
| 分群先验 | 0.009 | 0.015 | 0.0121 | 0.005 | 0.034 | 0.931 |
| 规则混合 | 0.123 | 0.166 | 0.1544 | 0.080 | 0.500 | 0.657 |
| **本文方法** | **0.283** | **0.411** | **0.3728** | **0.236** | **0.557** | **0.000** |

- 95% bootstrap CI（NDCG@10）：`[0.3496, 0.3989]`。
- 共同合法候选集上：内容 0.3544、规则混合 0.3481、本文方法 0.3728。
- 冷启动用户（40 人，无训练交互）：本文方法 NDCG@10=0.210，内容是 0.031，规则混合 0.034，热度 0.005。
- Agent 30 个场景：有效场景完成率 100%，缺失信息追问补充成功率 100%，非法输入拦截率 100%，无匹配识别率 100%，中位延迟约 56 ms，约束证据覆盖率 100%，技能证据覆盖率 92.1%。

关键消融：去掉内容后 NDCG 降到 0.3411；加入 CF 后降到 0.3402；加入 MMR 后降到 0.3692；去掉硬约束后降到 0.0241 且违规率 0.939；分群特征影响在噪声范围内（0.3739 vs 0.3728）。因此默认配置保留内容、技能、硬约束和轻量学习排序，CF/MMR 仅作为对照。

## 3. 快速开始

```powershell
pip install -r requirements.txt

python scripts/generate_data.py --jobs 3000 --users 400 --seed 42
python scripts/train_eval.py --seed 42

python -m job_agent.cli --demo
python -m job_agent.cli --profile examples/profile_data.json
python -m job_agent.cli --text "我是应届本科生，在杭州找数据分析工作，期望12k，会SQL、Python、Excel和Pandas。"

python -m job_agent.web --port 8000
# 浏览器打开 http://127.0.0.1:8000/?demo=1

python -m unittest discover -s tests -t . -v
python scripts/make_figures.py
```

## 4. 数据

- 12 个职业族 × 250 条岗位 = 3000 条岗位；
- 400 个合成用户；
- 13229 条交互，密度约 1.1%；
- 按用户 70%/10%/20% 划分训练/验证/测试；
- 全部为合成数据，不含真实姓名、电话等个人信息；
- 技能分类和别名由作者整理，参考 O*NET/ESCO 的公开分类思路，未直接复制其数据文件；
- 合成交互由规则化效用 + softmax 采样生成，含探索噪声，因此离线指标不能等同于真实用户满意度。

## 5. 方法与模块

- 文本：jieba 分词 + 自定义技能词典 + TF-IDF 词级 1–2-gram + 余弦相似度。
- 分群：K-Means（k=8），轮廓系数 0.184，稳定性 ARI 0.822；用于画像解释、冷启动先验和分群监控。消融显示分群对排序 NDCG 贡献有限，本项目在实验分析中如实说明。
- 召回：内容、ItemCF、SVD、分群先验。
- 排序：13 个特征（内容/分群/技能/城市/薪资/经验/学历/远程/热度/方向），逻辑回归 C=0.1，按验证集选择；最终分数 0.7×内容+0.3×排序概率。
- 硬约束：城市、薪资上限、学历、经验；无匹配时按四级放宽并提示。
- 解释：模板只引用画像和岗位中都存在的技能、真实约束和分群信息；可选 LLM 输出会做技能集合校验，失败回退模板。
- Agent：显式 trace、session、反馈日志；不保存 PII；不自动投递。

## 6. 复用组件

| 组件 | 版本 | 许可证 | 来源 | 用途 |
|---|---|---|---|---|
| Python | 3.12.7 | PSF | python.org | 运行时 |
| NumPy | 2.1.2 | BSD-3-Clause | numpy.org | 数值计算 |
| pandas | 2.2.3 | BSD-3-Clause | pandas.pydata.org | 数据处理 |
| SciPy | 1.18.1 | BSD-3-Clause | scipy.org | 稀疏矩阵 |
| scikit-learn | 1.9.1 | BSD-3-Clause | scikit-learn.org | TF-IDF/KMeans/GMM/SVD/LR/指标 |
| jieba | 0.42.1 | MIT | github.com/fxsjy/jieba | 中文分词 |
| matplotlib | 3.9.2 | PSF-based | matplotlib.org | 图表 |
| requests | 2.34.2 | Apache-2.0 | requests.readthedocs.io | 可选 LLM 调用（默认关闭） |

所有业务代码、数据生成器、Agent 工作流和图表均为自行编写；未修改开源库源码；未提交任何密钥。详细指标、演示记录和消融结果见 `artifacts/metrics.json`、`artifacts/agent_eval.json` 和 `artifacts/demo_transcript.txt`。

## 7. 目录结构

```text
job-reco-agent/
  data/                 技能词典与合成数据
  job_agent/            数据生成、文本、特征、分群、推荐、Agent、CLI、Web
  scripts/              数据生成、训练评估、图表
  tests/                23 个单元测试
  artifacts/            models/、figures/、metrics.json、demo_transcript.txt
  examples/             示例画像
  web/                  Web 前端
  README.md
```

## 8. 局限与安全边界

- 合成数据、无真实用户研究、无在线 A/B 测试；
- 中文为主，技能同义词和跨语言表达仍会漏匹配；
- 薪资/岗位信息未与招聘方核验；
- 分群可能强化刻板印象，热度特征可能带来流行度偏置；
- 排序器是线性模型，KMeans 假设近似球形簇；
- 可选 LLM 有幻觉风险，已做事实校验和回退，但未做大规模评测；
- 系统只做推荐，不自动投递；导出需要用户确认；反馈日志不含 PII，但仍应做访问控制。

## 9. 仓库内容与报告

本仓库包含可运行代码、预训练模型、演示截图、完整运行记录和指标文件。课程报告按课程要求单独提交，不在本仓库中。

## 10. 主要参考文献

1. Ricci F, Rokach L, Shapira B. Recommender Systems Handbook (2nd ed.). Springer, 2015. DOI: 10.1007/978-1-4899-7637-6.
2. Koren Y, Bell R, Volinsky C. Matrix Factorization Techniques for Recommender Systems. Computer, 2009, 42(8): 30–37. DOI: 10.1109/MC.2009.263.
3. Sarwar B, Karypis G, Konstan J, Riedl J. Item-based Collaborative Filtering Recommendation Algorithms. WWW 2001: 285–295. DOI: 10.1145/371920.372071.
4. Manning C D, Raghavan P, Schütze H. Introduction to Information Retrieval. Cambridge University Press, 2008.
5. Rousseeuw P J. Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. Journal of Computational and Applied Mathematics, 1987, 20: 53–65. DOI: 10.1016/0377-0427(87)90125-7.
6. Järvelin K, Kekäläinen J. Cumulated Gain-based Evaluation of IR Techniques. ACM Transactions on Information Systems, 2002, 20(4): 422–446. DOI: 10.1145/582415.582418.
7. Pedregosa F, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011, 12: 2825–2830.
8. Hu Y, Koren Y, Volinsky C. Collaborative Filtering for Implicit Feedback Datasets. ICDM 2008: 263–272. DOI: 10.1109/ICDM.2008.22.
9. Yao S, et al. ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023. arXiv:2210.03629.
10. Lewis P, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020. arXiv:2005.11401.
11. O*NET Resource Center. O*NET Database 29.0. https://www.onetcenter.org/database.html. Public Domain.
12. jieba Chinese word segmentation, version 0.42.1. https://github.com/fxsjy/jieba (MIT License).
13. scikit-learn documentation. https://scikit-learn.org/stable/ (BSD-3-Clause).

