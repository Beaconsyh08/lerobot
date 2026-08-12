#!/usr/bin/env python3
"""Package stage-return comparison data as a canonical portable HTML report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_artifact(analysis: dict, generated_at: str) -> dict:
    summary = analysis["summary"]
    worst_heatmap = [
        {
            **row,
            "joint_index": int(row["joint"].removeprefix("action[").removesuffix("]")),
        }
        for row in analysis["worst_heatmap"]
    ]
    source = {
        "id": "stage_alignment_comparison",
        "label": "0602 LeRobot direct stage-return comparison",
        "path": "analysis.json",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_json_auto('analysis.json', maximum_object_size=10485760)",
            "description": ("Episode-aligned comparison of original and direct-to-target action Parquet."),
            "executed_at": generated_at,
            "tables_used": [
                "lerobot.0602_dvt2_whitetable",
                "lerobot.0602_dvt2_whitetable_stage_return_aligned",
                "lerobot.0602_dvt2_whitetable_stage_return_direct",
            ],
            "filters": [
                "All 949 episodes",
                "Arm action dimensions 0-6 and 8-14",
                "Pick/place stage 3 to 4; give stage 4 to 5",
            ],
            "metric_definitions": [
                "Episode RMS change: root mean square of aligned minus original arm action across every frame and 14 arm joints.",
                "Target L2 change: Euclidean distance between the original final-stage boundary action and its cohort median target.",
                "Tail max step: maximum absolute adjacent-frame arm-action difference from return-stage start through episode end.",
            ],
        },
    }
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "0602 固定回位直达版：修改前后对比",
        "description": "原始轨迹与从 return-stage start 直接趋向固定目标的轨迹对比。",
        "generatedAt": generated_at,
        "filters": [
            {
                "id": "representative_group",
                "label": "代表轨迹组",
                "dataset": "representative_curves",
                "field": "group",
                "defaultValue": "pick:left",
                "includeAll": False,
                "targets": [{"dataset": "representative_curves", "field": "group"}],
            }
        ],
        "cards": [
            {
                "id": "episodes",
                "description": "完成对齐并通过逐集验证的 episode 数。",
                "dataset": "summary",
                "sourceId": source["id"],
                "metrics": [{"label": "已对齐 episodes", "field": "aligned_episodes", "format": "number"}],
            },
            {
                "id": "changed_share",
                "description": "至少一个 arm action 维度发生变化的帧占全部帧的比例。",
                "dataset": "summary",
                "sourceId": source["id"],
                "metrics": [{"label": "修改帧占比", "field": "changed_frame_fraction", "format": "percent"}],
            },
            {
                "id": "median_rms",
                "description": "949 个 episode 的 arm-action RMS 改动中位数。",
                "dataset": "summary",
                "sourceId": source["id"],
                "metrics": [
                    {
                        "label": "每集 RMS 改动中位数",
                        "field": "episode_rms_change_rad_median",
                        "format": "number",
                        "unit": "rad",
                    }
                ],
            },
            {
                "id": "max_step",
                "description": "刷后返回尾段的最大相邻帧关节步长。",
                "dataset": "summary",
                "sourceId": source["id"],
                "metrics": [
                    {
                        "label": "尾段最大单帧步长",
                        "field": "aligned_tail_max_step_rad_max",
                        "format": "number",
                        "unit": "rad/frame",
                    }
                ],
            },
            {
                "id": "max_change",
                "description": "全数据集中单个关节值的最大绝对修改量，用于识别异常大修正。",
                "dataset": "summary",
                "sourceId": source["id"],
                "metrics": [
                    {
                        "label": "最大单关节修改",
                        "field": "max_abs_change_rad",
                        "format": "number",
                        "unit": "rad",
                    }
                ],
            },
        ],
        "charts": [
            {
                "id": "rms_distribution",
                "title": "Episode RMS 改动分布",
                "subtitle": "949 episodes；按任务类型与执行手分组，单位 rad。",
                "type": "boxPlot",
                "intent": "distribution",
                "question": "四个数据组的典型改动量和离散程度分别多大？",
                "rationale": "箱线图同时保留各组中位数、离散程度和异常值信息。",
                "dataset": "episode_metrics",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "group", "type": "nominal", "label": "任务 × 执行手"},
                    "y": {
                        "field": "rms_change_rad",
                        "type": "quantitative",
                        "label": "Episode RMS 改动",
                        "format": "number",
                    },
                },
                "yAxisTitle": "RMS change (rad)",
                "valueFormat": "number",
                "palette": {"kind": "sequential", "name": "blue"},
                "layout": "full",
            },
            {
                "id": "change_relationship",
                "title": "目标距离与实际 RMS 改动",
                "subtitle": "每个点是一个 episode；颜色区分任务类型与执行手。",
                "type": "scatter",
                "intent": "relationship",
                "question": "边界位置离固定目标越远时，整集实际修改量是否同步增大？",
                "rationale": "episode 粒度散点能显示主关系、分组差异和离群点。",
                "dataset": "episode_metrics",
                "sourceId": source["id"],
                "encodings": {
                    "x": {
                        "field": "target_l2_change_rad",
                        "type": "quantitative",
                        "label": "边界到目标 L2 距离 (rad)",
                    },
                    "y": {
                        "field": "rms_change_rad",
                        "type": "quantitative",
                        "label": "Episode RMS 改动 (rad)",
                    },
                    "color": {"field": "group", "type": "nominal", "label": "任务 × 执行手"},
                    "tooltip": [
                        {"field": "episode_index", "type": "quantitative", "label": "Episode"},
                        {
                            "field": "max_abs_change_rad",
                            "type": "quantitative",
                            "label": "最大修改 (rad)",
                        },
                    ],
                },
                "palette": {"kind": "categorical", "name": "blue-orange"},
                "legend": {"position": "bottom", "interactive": True},
                "layout": "full",
            },
            {
                "id": "representative_curve",
                "title": "代表 episode 的单关节原始与直达轨迹",
                "subtitle": "同一 episode 对比旧版与新版；进度 0/1 分别是 return/final stage start。",
                "type": "line",
                "intent": "trend",
                "question": "固定目标是如何从 return-stage start 平滑注入到最终 stage 的？",
                "rationale": "同一关节的原始、直达和固定目标叠加最直接地显示是否发生折返。",
                "dataset": "representative_curves",
                "sourceId": source["id"],
                "encodings": {
                    "x": {
                        "field": "return_progress",
                        "type": "quantitative",
                        "label": "Return-stage 进度",
                    },
                    "y": {
                        "field": "action_rad",
                        "type": "quantitative",
                        "label": "关节 action (rad)",
                    },
                    "color": {"field": "series", "type": "nominal", "label": "轨迹"},
                    "lineStyle": {"field": "line_style", "type": "nominal"},
                    "tooltip": [
                        {"field": "episode_index", "type": "quantitative", "label": "Episode"},
                        {"field": "joint", "type": "nominal", "label": "Joint"},
                        {
                            "field": "relative_time_s",
                            "type": "quantitative",
                            "label": "距 final start (s)",
                        },
                        {"field": "stage", "type": "quantitative", "label": "Stage"},
                    ],
                },
                "referenceLines": [
                    {
                        "axis": "x",
                        "value": 0,
                        "label": "Return stage start",
                        "color": "neutral",
                        "lineStyle": "dotted",
                    },
                    {
                        "axis": "x",
                        "value": 1,
                        "label": "Final stage start",
                        "color": "neutral",
                        "lineStyle": "dotted",
                    },
                ],
                "palette": {"kind": "categorical", "name": "blue-orange"},
                "legend": {"position": "bottom", "interactive": True, "sort": "spec"},
                "layout": "full",
            },
            {
                "id": "worst_heatmap",
                "title": f"Episode {summary['worst_episode']} 的 14 关节修改热力图",
                "subtitle": "最大改动样例；时间 0 是最终 stage start，颜色为 aligned − original。",
                "type": "heatmap",
                "intent": "relationship",
                "question": "最大改动 episode 的修正集中在哪些关节和哪些尾段帧？",
                "rationale": "关节 × 时间热力图可以同时检查修正幅度、方向和时序连续性。",
                "dataset": "worst_heatmap",
                "sourceId": source["id"],
                "encodings": {
                    "x": {
                        "field": "relative_time_s",
                        "type": "quantitative",
                        "label": "相对最终 stage start 时间 (s)",
                    },
                    "y": {
                        "field": "joint_index",
                        "type": "quantitative",
                        "label": "Arm action index",
                    },
                    "color": {
                        "field": "delta_rad",
                        "type": "quantitative",
                        "label": "Aligned − original (rad)",
                    },
                    "tooltip": [
                        {"field": "joint", "type": "nominal", "label": "Joint"},
                        {"field": "stage", "type": "quantitative", "label": "Stage"},
                        {
                            "field": "abs_delta_rad",
                            "type": "quantitative",
                            "label": "绝对改动 (rad)",
                        },
                    ],
                },
                "palette": {"kind": "diverging", "name": "blue-orange", "midpoint": 0},
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "representative_table",
                "title": "与上一版相同的代表 episode",
                "subtitle": "固定沿用上一份报告的四个 episode 和关节，避免样例变化影响判断。",
                "dataset": "representatives",
                "sourceId": source["id"],
                "defaultSort": {"field": "group", "direction": "asc"},
                "columns": [
                    {"field": "group", "label": "Group", "type": "text"},
                    {"field": "episode_index", "label": "Episode", "type": "number"},
                    {"field": "joint", "label": "Displayed joint", "type": "text"},
                    {"field": "target_rad", "label": "Target", "format": "number", "unit": "rad"},
                    {
                        "field": "rms_change_rad",
                        "label": "RMS change",
                        "format": "number",
                        "unit": "rad",
                    },
                    {
                        "field": "max_abs_change_rad",
                        "label": "Max change",
                        "format": "number",
                        "unit": "rad",
                    },
                ],
            },
            {
                "id": "top_episodes",
                "title": "最大修改 episodes",
                "subtitle": "按单关节最大绝对修改量降序；用于优先人工复核。",
                "dataset": "top_episodes",
                "sourceId": source["id"],
                "defaultSort": {"field": "max_abs_change_rad", "direction": "desc"},
                "density": "dense",
                "columns": [
                    {"field": "episode_index", "label": "Episode", "type": "number"},
                    {"field": "group", "label": "Group", "type": "text"},
                    {
                        "field": "max_abs_change_rad",
                        "label": "Max change",
                        "format": "number",
                        "unit": "rad",
                    },
                    {
                        "field": "rms_change_rad",
                        "label": "RMS change",
                        "format": "number",
                        "unit": "rad",
                    },
                    {
                        "field": "target_l2_change_rad",
                        "label": "Target L2",
                        "format": "number",
                        "unit": "rad",
                    },
                    {
                        "field": "aligned_tail_max_step_rad",
                        "label": "Tail max step",
                        "format": "number",
                        "unit": "rad/frame",
                    },
                ],
            },
        ],
        "sources": [{"id": source["id"], "label": source["label"], "path": source["path"]}],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# 0602 固定回位直达版：修改前后对比",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": source["id"],
                "body": (
                    "## 结论先看\n\n"
                    f"- **改动集中在尾段。** {summary['changed_frames']:,} / {summary['source_frames']:,} 帧发生 arm-action 修改，占 **{summary['changed_frame_fraction']:.2%}**；return-stage start 之前保持不变。\n"
                    f"- **大多数修改较小。** Episode RMS 改动中位数为 **{summary['episode_rms_change_rad_median']:.4f} rad**，90 分位为 **{summary['episode_rms_change_rad_p90']:.4f} rad**。\n"
                    f"- **不再沿原轨迹折返。** {summary['aligned_episodes']} 集均从 return-stage start 姿态直接趋向固定目标；逐关节不越过起点—目标区间且方向不反转。\n"
                    f"- **没有边界跳变失控。** 直达轨迹尾段最大相邻帧步长为 **{summary['aligned_tail_max_step_rad_max']:.3f} rad/frame**，{summary['aligned_episodes']} 集均通过逐集验证。\n"
                    f"- **需要重点看离群样例。** Episode **{summary['worst_episode']}** 的单关节最大修改为 **{summary['max_abs_change_rad']:.4f} rad**。"
                ),
            },
            {
                "id": "metrics",
                "type": "metric-strip",
                "cardIds": ["episodes", "changed_share", "median_rms", "max_step", "max_change"],
            },
            {
                "id": "distribution_note",
                "type": "markdown",
                "sourceId": source["id"],
                "body": (
                    "## 修改量在四组中的分布\n\n"
                    "箱线图用于判断修改是否普遍很大，还是主要由少数离群 episode 拉高。"
                    "每个点的统计口径均覆盖该 episode 的全部帧和 14 个 arm action 维度。"
                ),
            },
            {"id": "distribution", "type": "chart", "chartId": "rms_distribution", "layout": "full"},
            {
                "id": "relationship_note",
                "type": "markdown",
                "sourceId": source["id"],
                "body": (
                    "## 离目标越远，整集改动通常越大\n\n"
                    "横轴是原 stage 边界到固定目标的 14 维 L2 距离，纵轴是整集 RMS 改动。"
                    "右上角 episode 是最值得优先人工检查的样例。"
                ),
            },
            {"id": "relationship", "type": "chart", "chartId": "change_relationship", "layout": "full"},
            {
                "id": "representative_note",
                "type": "markdown",
                "sourceId": source["id"],
                "body": (
                    "## 代表曲线从 return-stage start 直接趋向固定目标\n\n"
                    "这里固定使用上一版报告完全相同的四个 episode 和关节。`Previous aligned` 会继续沿原 stage 3 "
                    "轨迹后再拉回目标，而 `Direct aligned` 从 return-stage start 直接插值到目标。"
                    "顶部筛选器可在 pick/place × 左/右手之间切换；横轴进度 0 是 return-stage start，"
                    "进度 1 是 final-stage start。进度 0 之前的轨迹按要求保持不变。"
                ),
            },
            {"id": "representative_table_block", "type": "table", "tableId": "representative_table"},
            {
                "id": "representative_chart",
                "type": "chart",
                "chartId": "representative_curve",
                "layout": "full",
            },
            {
                "id": "worst_note",
                "type": "markdown",
                "sourceId": source["id"],
                "body": (
                    f"## 最大改动样例集中在 Episode {summary['worst_episode']} 的返回尾段\n\n"
                    "热力图展示所有 14 个 arm action 的 signed delta。颜色连续变化说明修正是渐进注入；"
                    "若出现孤立色块或跨帧突变，则需要回到该 episode 进一步检查。"
                ),
            },
            {"id": "worst_chart", "type": "chart", "chartId": "worst_heatmap", "layout": "full"},
            {
                "id": "top_note",
                "type": "markdown",
                "body": (
                    "## 优先人工复核列表\n\n"
                    "下表按最大单关节修改排序。建议先查看这些 episode 的相机画面，确认固定目标在真实场景中没有碰撞风险。"
                ),
            },
            {"id": "top_table_block", "type": "table", "tableId": "top_episodes", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "body": (
                    "## 范围与指标定义\n\n"
                    "- 数据范围：0602 white-table 数据集全部 949 episodes、157,638 frames。\n"
                    "- Pick/Place 修改区间：stage3 start 之后；stage4 start 起固定在目标。\n"
                    "- Give 规则：stage4 start 之后；stage5 start 起固定在目标。本数据集没有 give 样本。\n"
                    "- 只比较并修改 arm action；state、图像、夹爪 action 和 stage 标签不变。"
                ),
            },
            {
                "id": "method",
                "type": "markdown",
                "body": (
                    "## 目标和轨迹如何生成\n\n"
                    "目标按任务类型 × 执行手分组，取当前最终-stage 边界 14 维 arm action 的逐维中位数。"
                    "每条轨迹从 return-stage start 的真实关节姿态出发，直接在起点与固定目标之间插值，"
                    "不再叠加原 stage 3 的中间轨迹；在最终 stage start 精确到达目标。"
                    "插值权重单调且不越界；若三次平滑曲线会超过 0.1 rad/frame，则自动退向线性权重。"
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 局限与稳健性边界\n\n"
                    "这是 action relabeling，不会同步改变相机图像或观测 state，因此不能替代真实机器人回放验证。"
                    "中位目标能抗少数异常值，但仍可能把多模态的合法返回姿态压成单一目标。"
                    "报告证明数值连续性和字段一致性，不证明新目标在所有物体位置都无碰撞。"
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 建议下一步\n\n"
                    "1. 先人工查看 Top 15 episode，尤其是 Episode 242。\n"
                    "2. 在仿真或低速真机上分别回放 pick-left、pick-right、place-left、place-right 的目标。\n"
                    "3. 通过后再用同配置训练原始集与刷后集，比较 stage transition 和动作成功率。"
                ),
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": (
                    "## 后续需要确认的问题\n\n"
                    "- Pick 与 Place 是否应继续使用不同目标，还是最终应共享同一左右手回位姿态？\n"
                    "- Give 数据到位后，是否沿用独立的 give-left / give-right 中位目标？\n"
                    "- 是否需要为最大修改量设置 episode 跳过阈值，而不是对所有样例强制对齐？"
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary": [summary],
                "group_summary": analysis["group_summary"],
                "episode_metrics": analysis["episode_metrics"],
                "representatives": analysis["representatives"],
                "representative_curves": analysis["representative_curves"],
                "worst_heatmap": worst_heatmap,
                "top_episodes": analysis["top_episodes"],
            },
        },
        "sources": [source],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifact = build_artifact(analysis, generated_at)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
