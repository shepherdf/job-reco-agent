import argparse
import json
from pathlib import Path

from .agent import JobAgent
from .demo import evaluate_agent, run_demo
from .taxonomy import PROJECT_ROOT


def format_response(response: dict, max_recs: int = 10) -> str:
    lines = [f"状态: {response['status']}"]
    if response.get("message"):
        lines.append(response["message"])
    if response.get("cluster"):
        c = response["cluster"]
        lines.append(f"分群: {c['name']} | 规模: {c['size']} | 常见技能: {'、'.join(c['top_skills'][:5])}")
    if response.get("questions"):
        lines.append("需要补充:")
        for q in response["questions"]:
            lines.append(f"  - {q}")
    if response.get("warnings"):
        lines.append("提示:")
        for w in response["warnings"]:
            lines.append(f"  - {w}")
    for rec in response.get("recommendations", [])[:max_recs]:
        lines.append("")
        lines.append(
            f"[{rec['rank']}] {rec['title']} @ {rec['company']} | {rec['city']} | {rec['salary']} | "
            f"{rec['family']} | 匹配度 {rec['match_level']} ({rec['score']:.3f})"
        )
        lines.append(f"    技能: {'、'.join(rec['skills'][:8])}")
        lines.append(f"    解释: {rec['explanation']}")
        if rec.get("evidence", {}).get("matched_skills"):
            lines.append(f"    证据: 匹配技能 {'、'.join(rec['evidence']['matched_skills'][:5])}")
    if response.get("trace"):
        lines.append("")
        lines.append("执行轨迹:")
        for step in response["trace"]:
            lines.append(f"  - {step['step']}: {step['status']} ({step['elapsed_ms']} ms)")
    lines.append("")
    lines.append(f"耗时: {response.get('latency_ms', 0):.2f} ms")
    return "\n".join(lines)


def _ask(prompt: str, default: str | None = None) -> str:
    try:
        value = input(prompt).strip()
    except EOFError:
        return default or ""
    return value or (default or "")


def interactive(agent: JobAgent, top_k: int) -> dict:
    print("求职岗位推荐 Agent（输入“退出”结束）")
    print("示例：我是应届本科生，在杭州找数据分析工作，期望12k，会SQL、Python、Excel、Pandas和统计分析。")
    text = _ask("请描述你的求职需求: ")
    if text in {"退出", "quit", "exit"}:
        return {"status": "exit"}
    response = agent.run(text, top_k=top_k)
    turns = 0
    while response["status"] == "need_more_info" and turns < 3:
        print("\n" + format_response(response))
        payload = dict(response.get("profile", {}))
        questions = response.get("questions", [])
        for q in questions:
            if "哪个城市" in q:
                payload["city"] = _ask("城市: ", payload.get("city"))
            elif "期望月薪" in q:
                payload["expected_salary"] = _ask("期望月薪(k): ", str(payload.get("expected_salary") or ""))
            elif "几年" in q:
                payload["experience_years"] = _ask("工作经验(年): ", str(payload.get("experience_years") or ""))
            elif "学历" in q:
                payload["education"] = _ask("学历: ", payload.get("education") or "本科")
            elif "核心技能" in q or "岗位方向" in q:
                payload["skills"] = _ask("核心技能(用顿号分隔): ", "、".join(payload.get("skills", [])))
        response = agent.run(payload, top_k=top_k)
        turns += 1
    print("\n" + format_response(response, max_recs=top_k))
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="求职岗位推荐 Agent")
    parser.add_argument("--demo", action="store_true", help="运行完整演示案例")
    parser.add_argument("--eval-agent", action="store_true", help="运行 Agent 场景评测")
    parser.add_argument("--profile", type=str, help="JSON 画像文件路径")
    parser.add_argument("--text", type=str, help="自由文本求职需求")
    parser.add_argument("--interactive", action="store_true", help="交互式多轮咨询")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--export", type=str, help="导出推荐清单 CSV")
    parser.add_argument("--yes", action="store_true", help="导出时跳过确认")
    parser.add_argument("--use-llm", action="store_true", help="启用可选 LLM 解释（需要环境变量密钥）")
    args = parser.parse_args()

    if not any([args.demo, args.eval_agent, args.profile, args.text, args.interactive]):
        parser.print_help()
        return
    agent = JobAgent.from_artifacts(PROJECT_ROOT, use_llm=args.use_llm)

    if args.demo:
        run_demo(agent, PROJECT_ROOT / "artifacts")
        return
    if args.eval_agent:
        result = evaluate_agent(agent, PROJECT_ROOT / "artifacts")
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
        return
    if args.interactive:
        response = interactive(agent, args.top_k)
    elif args.profile:
        with open(args.profile, "r", encoding="utf-8") as f:
            payload = json.load(f)
        response = agent.run(payload, top_k=args.top_k)
    else:
        response = agent.run(args.text, top_k=args.top_k)

    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(format_response(response, max_recs=args.top_k))

    if args.export and response.get("status") == "ok":
        confirm = args.yes
        if not confirm:
            confirm = _ask(f"确认导出 {len(response['recommendations'])} 条推荐到 {args.export}? (y/N): ").lower() in {"y", "yes", "是"}
        if confirm:
            path = agent.export_csv(response, args.export)
            print(f"已导出: {path}")
        else:
            print("未导出。Agent 不会自动投递，需用户确认。")


if __name__ == "__main__":
    main()
