import json
from pathlib import Path
from typing import List

import numpy as np

from .agent import JobAgent
from .taxonomy import PROJECT_ROOT


SAMPLE_COMPLETE = {
    "city": "杭州",
    "target_cities": ["杭州", "上海"],
    "expected_salary": 12,
    "experience_years": 0,
    "education": "本科",
    "remote_pref": False,
    "skills": ["SQL", "Python", "Excel", "Pandas", "统计分析", "数据可视化"],
    "preferred_families": ["data"],
    "bio": "应届本科，统计学专业，做过课程数据分析和一段报表实习，希望找数据分析方向的工作。",
}

SAMPLE_INCOMPLETE = "我想找数据分析方向的工作，期望月薪 15k 左右，最好在杭州。"

SAMPLE_INVALID = {
    "city": "杭州",
    "expected_salary": -5,
    "experience_years": 0,
    "education": "本科",
    "skills": ["SQL", "Python"],
    "preferred_families": ["data"],
}

SAMPLE_NO_MATCH = {
    "city": "重庆",
    "target_cities": ["重庆"],
    "expected_salary": 100,
    "experience_years": 0,
    "education": "大专",
    "skills": ["会计核算", "财务报表", "Excel"],
    "preferred_families": ["finance"],
}

SAMPLE_REMOTE = {
    "city": "北京",
    "target_cities": ["北京", "上海", "杭州"],
    "expected_salary": 22,
    "experience_years": 3,
    "education": "硕士",
    "remote_pref": True,
    "skills": ["Python", "机器学习", "PyTorch", "NLP", "模型评估"],
    "preferred_families": ["algorithm"],
}


def _format_response(response: dict, max_recs: int = 3) -> str:
    lines = [f"状态：{response['status']}"]
    if response.get("message"):
        lines.append(f"说明：{response['message']}")
    if response.get("cluster"):
        c = response["cluster"]
        lines.append(f"分群：{c['name']}（{c['size']} 人，常见技能：{'、'.join(c['top_skills'][:3])}）")
    if response.get("questions"):
        lines.append("需要补充：")
        for q in response["questions"]:
            lines.append(f"  - {q}")
    if response.get("warnings"):
        lines.append("提示：")
        for w in response["warnings"]:
            lines.append(f"  - {w}")
    for rec in response.get("recommendations", [])[:max_recs]:
        lines.append("")
        lines.append(
            f"[{rec['rank']}] {rec['title']} @ {rec['company']} | {rec['city']} | {rec['salary']} | "
            f"匹配度 {rec['match_level']}（score={rec['score']:.3f}）"
        )
        lines.append(f"    技能：{'、'.join(rec['skills'][:6])}")
        lines.append(f"    解释：{rec['explanation']}")
    if response.get("trace"):
        lines.append("")
        lines.append("执行轨迹：")
        for step in response["trace"]:
            lines.append(f"  - {step['step']}: {step['status']} ({step['elapsed_ms']} ms)")
    return "\n".join(lines)


def run_demo(agent: JobAgent, out_dir: Path | str = PROJECT_ROOT / "artifacts") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    transcript: List[str] = []
    transcript.append("=" * 78)
    transcript.append("求职岗位推荐 Agent 完整运行记录")
    transcript.append("=" * 78)

    transcript.append("\n【案例 1】信息不完整，Agent 主动追问")
    transcript.append(f"用户输入：{SAMPLE_INCOMPLETE}")
    r1 = agent.run(SAMPLE_INCOMPLETE, session_id="demo_incomplete")
    results["incomplete"] = r1
    transcript.append(_format_response(r1))
    transcript.append("\n用户补充：城市=杭州，经验=0 年（应届），学历=本科，技能=SQL/Python/Excel/Pandas/统计分析/数据可视化")
    completed = dict(SAMPLE_COMPLETE)
    completed["bio"] = SAMPLE_INCOMPLETE + " " + completed["bio"]
    r1b = agent.run(completed, session_id="demo_incomplete")
    results["incomplete_followup"] = r1b
    transcript.append(_format_response(r1b, max_recs=3))

    transcript.append("\n【案例 2】信息完整的新用户（冷启动）")
    r2 = agent.run(SAMPLE_COMPLETE, session_id="demo_coldstart")
    results["complete"] = r2
    transcript.append(_format_response(r2, max_recs=5))

    transcript.append("\n【案例 3】异常输入：期望薪资为负数")
    r3 = agent.run(SAMPLE_INVALID, session_id="demo_invalid")
    results["invalid"] = r3
    transcript.append(_format_response(r3))

    transcript.append("\n【案例 4】高约束导致无匹配")
    r4 = agent.run(SAMPLE_NO_MATCH, session_id="demo_nomatch")
    results["no_match"] = r4
    transcript.append(_format_response(r4))

    transcript.append("\n【案例 5】远程偏好与较高期望薪资")
    r5 = agent.run(SAMPLE_REMOTE, session_id="demo_remote")
    results["remote"] = r5
    transcript.append(_format_response(r5, max_recs=3))

    transcript.append("\n【案例 6】用户反馈记录（不包含姓名/电话等隐私字段）")
    if r2.get("recommendations"):
        fb = agent.record_feedback("demo_coldstart", r2["recommendations"][0]["job_id"], "like", "示例反馈")
        results["feedback"] = fb
        transcript.append(json.dumps(fb, ensure_ascii=False))
        transcript.append(f"反馈汇总：{json.dumps(agent.feedback_store.summary(), ensure_ascii=False)}")

    text = "\n".join(transcript)
    (out_dir / "demo_transcript.txt").write_text(text, encoding="utf-8")
    with open(out_dir / "demo.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(text)
    return results


def _base_profile(user: dict) -> dict:
    return {
        "city": user["city"],
        "target_cities": user["target_cities"],
        "expected_salary": user["expected_salary"],
        "experience_years": user["experience_years"],
        "education": user["education"],
        "remote_pref": user["remote_pref"],
        "skills": user["skills"],
        "preferred_families": user["preferred_families"],
        "bio": user["bio"],
    }


def evaluate_agent(agent: JobAgent, out_dir: Path | str = PROJECT_ROOT / "artifacts") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    sample_users = agent.users.sample(n=10, random_state=42).to_dict("records")
    scenarios = []
    for user in sample_users:
        scenarios.append({"name": "complete", "payload": _base_profile(user)})
    missing_cases = [
        {"name": "missing_city", "payload": {k: v for k, v in SAMPLE_COMPLETE.items() if k not in {"city", "target_cities"}}},
        {"name": "missing_salary", "payload": {k: v for k, v in SAMPLE_COMPLETE.items() if k != "expected_salary"}},
        {"name": "missing_experience", "payload": {k: v for k, v in SAMPLE_COMPLETE.items() if k != "experience_years"}},
        {"name": "missing_education", "payload": {k: v for k, v in SAMPLE_COMPLETE.items() if k != "education"}},
        {"name": "missing_skills", "payload": {k: v for k, v in SAMPLE_COMPLETE.items() if k not in {"skills", "preferred_families", "bio"}}},
    ]
    scenarios.extend(missing_cases)
    scenarios.extend([
        {"name": "invalid_salary_negative", "payload": dict(SAMPLE_INVALID)},
        {"name": "invalid_salary_high", "payload": {**SAMPLE_COMPLETE, "expected_salary": 200}},
        {"name": "invalid_experience", "payload": {**SAMPLE_COMPLETE, "experience_years": -1}},
        {"name": "no_match_finance", "payload": dict(SAMPLE_NO_MATCH)},
        {"name": "no_match_salary", "payload": {**SAMPLE_NO_MATCH, "expected_salary": 90}},
        {"name": "no_match_education", "payload": {**SAMPLE_NO_MATCH, "education": "大专", "experience_years": 0}},
        {"name": "cold_data", "payload": dict(SAMPLE_COMPLETE)},
        {"name": "cold_algorithm", "payload": {**SAMPLE_REMOTE, "remote_pref": False}},
        {"name": "cold_operations", "payload": {
            "city": "成都", "target_cities": ["成都", "重庆"], "expected_salary": 9,
            "experience_years": 1, "education": "本科", "remote_pref": False,
            "skills": ["内容运营", "用户运营", "文案", "数据复盘", "活动策划"],
            "preferred_families": ["operations"],
        }},
        {"name": "remote_pref", "payload": dict(SAMPLE_REMOTE)},
        {"name": "many_cities", "payload": {**SAMPLE_COMPLETE, "target_cities": ["杭州", "上海", "北京", "深圳", "广州", "成都"]}},
        {"name": "high_salary", "payload": {**SAMPLE_REMOTE, "expected_salary": 45}},
        {"name": "text_data", "payload": "我是应届本科生，在杭州找数据分析工作，期望12k，会SQL、Python、Excel、Pandas和统计分析。"},
        {"name": "text_backend", "payload": "我有3年后端经验，本科，在北京找Go后端工作，期望25k，熟悉Go、Docker、MySQL、Redis和微服务。"},
        {"name": "text_design", "payload": "我是大专学历，1年设计经验，想在上海找UI设计工作，期望10k，熟悉Figma、Sketch、交互设计和用户体验。"},
    ])
    rows = []
    for scenario in scenarios:
        first = agent.run(scenario["payload"], session_id=f"eval_{scenario['name']}")
        turns = 1
        final = first
        if first["status"] == "need_more_info":
            turns = 2
            if scenario["name"].startswith("missing"):
                merged = dict(SAMPLE_COMPLETE)
                merged.update({k: v for k, v in scenario["payload"].items() if v is not None})
                final = agent.run(merged, session_id=f"eval_{scenario['name']}_retry")
            else:
                final = first
        recs = final.get("recommendations", [])
        evidence_count = sum(1 for r in recs if r.get("evidence", {}).get("matched_skills"))
        constraint_count = sum(1 for r in recs if r.get("evidence", {}).get("constraints"))
        category = "valid"
        if scenario["name"].startswith("missing_"):
            category = "missing"
        elif scenario["name"].startswith("invalid_"):
            category = "invalid"
        elif scenario["name"].startswith("no_match_"):
            category = "no_match"
        rows.append({
            "scenario": scenario["name"],
            "category": category,
            "first_status": first["status"],
            "final_status": final["status"],
            "turns": turns,
            "n_recs": len(recs),
            "confidence": final.get("confidence"),
            "questions": first.get("questions", []),
            "warnings": final.get("warnings", []),
            "latency_ms": final.get("latency_ms", 0),
            "skill_evidence": evidence_count / len(recs) if recs else 0.0,
            "constraint_evidence": constraint_count / len(recs) if recs else 0.0,
        })
    frame = np.array([1 if r["final_status"] == "ok" and r["n_recs"] > 0 else 0 for r in rows])
    valid = [r for r in rows if r["category"] == "valid"]
    missing = [r for r in rows if r["category"] == "missing"]
    invalid = [r for r in rows if r["category"] == "invalid"]
    no_match = [r for r in rows if r["category"] == "no_match"]
    result = {
        "n_scenarios": len(rows),
        "overall_completion_rate": float(frame.mean()),
        "valid_completion_rate": float(np.mean([1 if r["final_status"] == "ok" and r["n_recs"] > 0 else 0 for r in valid])) if valid else 1.0,
        "missing_scenarios": len(missing),
        "missing_resolved_rate": float(np.mean([1 if r["final_status"] == "ok" and r["n_recs"] > 0 else 0 for r in missing])) if missing else 1.0,
        "invalid_scenarios": len(invalid),
        "invalid_handled_rate": float(np.mean([1 if r["first_status"] == "need_more_info" else 0 for r in invalid])) if invalid else 1.0,
        "no_match_scenarios": len(no_match),
        "no_match_correct_rate": float(np.mean([1 if r["final_status"] == "no_match" else 0 for r in no_match])) if no_match else 1.0,
        "avg_turns": float(np.mean([r["turns"] for r in rows])),
        "avg_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        "low_confidence_rate": float(np.mean([1 if r["confidence"] == "low" else 0 for r in rows if r["final_status"] == "ok"])) if any(r["final_status"] == "ok" for r in rows) else 0.0,
        "skill_evidence_coverage": float(np.mean([r["skill_evidence"] for r in rows if r["n_recs"] > 0])) if any(r["n_recs"] > 0 for r in rows) else 0.0,
        "constraint_evidence_coverage": float(np.mean([r["constraint_evidence"] for r in rows if r["n_recs"] > 0])) if any(r["n_recs"] > 0 for r in rows) else 0.0,
        "rows": rows,
    }
    with open(out_dir / "agent_eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
