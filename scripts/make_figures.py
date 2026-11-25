import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from job_agent.clustering import UserClusterer
from job_agent.data_gen import load_processed
from job_agent.features import FeatureBuilder

FIG_DIR = ROOT / "artifacts" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


def load_metrics():
    with open(ROOT / "artifacts" / "metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


def box(ax, x, y, w, h, text, color, edge="#ffffff", fontsize=10):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#17212b", wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>", dashed=False, color="#5b6b82"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        linewidth=1.4, color=color, linestyle="--" if dashed else "-",
        shrinkA=2, shrinkB=2,
    ))


def architecture():
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.text(0.2, 6.95, "求职岗位推荐 Agent 工作流", fontsize=18, fontweight="bold", color="#17212b")
    ax.text(0.2, 6.62, "离线可复现 · 用户分群 · 混合推荐 · 证据化解释 · 人工确认", fontsize=10, color="#637083")
    blue = "#eaf0ff"
    purple = "#f0eaff"
    green = "#e7f8f2"
    orange = "#fff3e6"
    gray = "#f1f4f8"
    box(ax, 0.4, 4.9, 2.2, 1.05, "① 用户画像\n城市/薪资/经验/技能", blue)
    box(ax, 3.0, 4.9, 2.2, 1.05, "② 解析与校验\n缺失追问/异常拦截", blue)
    box(ax, 5.6, 4.9, 2.2, 1.05, "③ 技能抽取与分群\nK-Means (k=8)", purple)
    box(ax, 8.2, 4.9, 2.9, 1.05, "④ 硬约束过滤\n城市/薪资/学历/经验", green)
    box(ax, 8.2, 3.2, 2.9, 1.05, "⑤ 多路召回\n内容 TF-IDF / 协同 / 分群", purple)
    box(ax, 5.6, 3.2, 2.2, 1.05, "⑥ 学习排序\n逻辑回归 + 特征融合", purple)
    box(ax, 3.0, 3.2, 2.2, 1.05, "⑦ 解释与证据\n技能重合/约束/分群", green)
    box(ax, 0.4, 3.2, 2.2, 1.05, "⑧ 置信度与复核\n低分/低区分度提示", orange)
    box(ax, 0.4, 1.5, 2.2, 1.05, "推荐结果 / 反馈\n不自动投递", orange, fontsize=9)
    box(ax, 3.0, 1.5, 5.1, 1.05, "数据与工具：岗位库 3000 条  |  技能词典  |  反馈日志", gray, fontsize=9)
    box(ax, 8.2, 1.5, 2.9, 1.05, "评估：NDCG@10 / Recall@10\n覆盖 / 违规 / 消融", gray, fontsize=9)
    arrow(ax, 2.6, 5.42, 3.0, 5.42)
    arrow(ax, 5.2, 5.42, 5.6, 5.42)
    arrow(ax, 7.8, 5.42, 8.2, 5.42)
    arrow(ax, 9.65, 4.9, 9.65, 4.25)
    arrow(ax, 8.2, 3.72, 7.8, 3.72)
    arrow(ax, 5.6, 3.72, 5.2, 3.72)
    arrow(ax, 3.0, 3.72, 2.6, 3.72)
    arrow(ax, 1.5, 3.2, 1.5, 2.55)
    arrow(ax, 1.5, 1.5, 1.5, 1.15, dashed=False, color="#8a95a6")
    arrow(ax, 2.6, 2.02, 3.0, 2.02, dashed=True, color="#8a95a6")
    arrow(ax, 8.1, 2.02, 8.2, 2.02, dashed=True, color="#8a95a6")
    arrow(ax, 4.5, 2.55, 5.6, 3.2, dashed=True, color="#8a95a6")
    arrow(ax, 9.65, 1.5, 1.5, 3.2, dashed=True, color="#c0c8d4", style="-|>")
    ax.text(0.3, 0.75, "实线：一次请求的执行流；虚线：数据/反馈回流。", fontsize=9, color="#8a95a6")
    ax.text(0.3, 0.45, "所有步骤写入 trace，包含候选数、放宽级别、排序分数、置信度与耗时。", fontsize=9, color="#8a95a6")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_architecture.png", dpi=170)
    plt.close(fig)


def cluster_figures(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    records = metrics["clustering"]["k_selection"]
    ks = [r["k"] for r in records]
    sil = [r["silhouette"] for r in records]
    db = [r["davies_bouldin"] for r in records]
    chosen = metrics["clustering"]["k"]
    ax = axes[0]
    ax.plot(ks, sil, marker="o", color="#315efb", linewidth=2)
    ax.axvline(chosen, color="#22b8a7", linestyle="--", linewidth=1.5, label=f"选中 k={chosen}")
    ax.set_xlabel("聚类数 k")
    ax.set_ylabel("轮廓系数（越大越好）")
    ax.set_title("K-Means 的 k 选择", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax = axes[1]
    ax.plot(ks, db, marker="s", color="#b45309", linewidth=2)
    ax.axvline(chosen, color="#22b8a7", linestyle="--", linewidth=1.5)
    ax.set_xlabel("聚类数 k")
    ax.set_ylabel("Davies-Bouldin（越小越好）")
    ax.set_title("簇内紧凑度与簇间分离度", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cluster_selection.png", dpi=170)
    plt.close(fig)

    jobs, users, _ = load_processed()
    builder = FeatureBuilder(jobs)
    clusterer = UserClusterer.load(ROOT / "artifacts" / "models" / "clusterer.joblib", builder)
    X, _ = builder.build_user_cluster_matrix(users)
    labels = clusterer.kmeans.predict(X)
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    cmap = plt.get_cmap("tab10")
    for c in range(clusterer.k):
        mask = labels == c
        ax.scatter(points[mask, 0], points[mask, 1], s=34, alpha=0.78, color=cmap(c % 10), label=clusterer.profile(c)["name"])
        center = points[mask].mean(axis=0)
        ax.text(center[0], center[1], str(c), fontsize=10, fontweight="bold", color="#17212b",
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=cmap(c % 10), lw=1.1))
    ax.set_xlabel(f"PC1（解释方差 {pca.explained_variance_ratio_[0]:.1%}）")
    ax.set_ylabel(f"PC2（解释方差 {pca.explained_variance_ratio_[1]:.1%}）")
    ax.set_title("用户分群的二维 PCA 可视化", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best", frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cluster_scatter.png", dpi=170)
    plt.close(fig)


def main_metrics(metrics):
    main = metrics["main"]
    order = ["Popularity", "Content", "ItemCF", "SVD", "ClusterPrior", "HybridRule", "Proposed"]
    labels = ["热度", "内容", "ItemCF", "SVD", "分群先验", "规则混合", "本文方法"]
    ndcg = [main[m]["ndcg@10"] for m in order]
    recall = [main[m]["recall@10"] for m in order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    colors = ["#98a2b3"] * 6 + ["#315efb"]
    for ax, values, title in [
        (axes[0], ndcg, "NDCG@10（越高越好）"),
        (axes[1], recall, "Recall@10（越高越好）"),
    ]:
        bars = ax.bar(x, values, color=colors, width=0.66)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_main_metrics.png", dpi=170)
    plt.close(fig)


def ablation(metrics):
    order = ["Full", "NoCluster", "NoContent", "NoSkill", "WithCF", "WithMMR", "NoHardFilter"]
    labels = ["完整", "去分群", "去内容", "去技能", "加协同", "加MMR", "去硬约束"]
    data = metrics["ablation"]
    ndcg = [data[m]["ndcg@10"] for m in order]
    cov = [data[m]["coverage@10"] for m in order]
    colors = ["#315efb", "#98a2b3", "#98a2b3", "#98a2b3", "#b45309", "#b45309", "#b42318"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    for ax, values, title in [
        (axes[0], ndcg, "NDCG@10"),
        (axes[1], cov, "Coverage@10"),
    ]:
        bars = ax.bar(x, values, color=colors, width=0.66)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(max(values) * 1.18, 0.1))
    fig.suptitle("消融实验：各组件对排序质量与覆盖的影响", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ablation.png", dpi=170)
    plt.close(fig)


def cold_start(metrics):
    data = metrics["cold_start"]
    order = ["Popularity", "Content", "ClusterPrior", "HybridRule", "Proposed"]
    labels = ["热度", "内容", "分群先验", "规则混合", "本文方法"]
    values = [data[m]["ndcg@10"] for m in order]
    colors = ["#98a2b3"] * 4 + ["#315efb"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("NDCG@10")
    ax.set_title("冷启动用户（无历史行为）的排序效果", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_coldstart.png", dpi=170)
    plt.close(fig)


def tokenizer(metrics):
    data = metrics["tokenizer_ablation"]
    groups = ["内容单独\n（合法候选）", "完整模型\n（合法候选）", "完整模型\n（端到端）"]
    word = [data["word"]["content_fair_ndcg@10"], data["word"]["fair_ndcg@10"], data["word"]["end_ndcg@10"]]
    char = [data["char"]["content_fair_ndcg@10"], data["char"]["fair_ndcg@10"], data["char"]["end_ndcg@10"]]
    x = np.arange(len(groups))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars1 = ax.bar(x - width / 2, word, width, label="jieba 词粒度", color="#315efb")
    bars2 = ax.bar(x + width / 2, char, width, label="字符 2–4-gram", color="#22b8a7")
    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("NDCG@10")
    ax.set_title("中文表示对照：jieba 词粒度 vs 字符 n-gram", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(word + char) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tokenizer.png", dpi=170)
    plt.close(fig)


def main():
    metrics = load_metrics()
    architecture()
    cluster_figures(metrics)
    main_metrics(metrics)
    ablation(metrics)
    cold_start(metrics)
    tokenizer(metrics)
    for path in sorted(FIG_DIR.glob("fig_*.png")):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
