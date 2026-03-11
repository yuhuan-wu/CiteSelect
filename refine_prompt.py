from typing import List, Dict, Any
import json


def build_refine_prompt(target_meta: Dict[str, Any], batch_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build messages for LLM refinement with a compact output schema.
    The model only returns the judgment fields; all metadata are restored locally.
    """
    sys_prompt = (
        "你是学术评审助手。任务：基于每条引用的 isInfluential、intents、contexts（引用上下文）和引文论文元数据，"
        "判定该引用对目标论文的重要性。严格输出 NDJSON（每行一个 JSON 对象），不得使用代码块```、数组或任何额外文本。"
        "不要复述输入里的标题、作者、年份、上下文。"
        "\n每行只输出 5 个字段："
        '\n- i: 输入编号'
        '\n- s: importance_score，1-5整数'
        '\n- c: category，取值仅限 背景|方法|结果|扩展|比较|其他'
        '\n- r: citation_role，取值仅限 方法借鉴|性能比较|代表性工作|数据/资源引用|背景铺垫|扩展应用|弱相关'
        '\n- w: 极短理由，10-20个汉字，禁止复述整句上下文'
        "\n另外：输入还包含 authorsWithCitations（形如“Name (count)”的作者列表）与 maxAuthorCitation（该条引用作者中的最高被引数）。"
        "作者影响力是加权因子之一：高被引作者通常意味着该工作更具权威性或影响力。可在同等证据下适度提高重要性评分，但不得取代上下文/意图证据。"
        "\n评分准则："
        "\n- 5分：该引用构成本文方法的关键前人工作，被直接采用/扩展/复现/作为核心基线；或上下文为长句详细介绍对方方法/系统/架构（例如“propose ... system ... consisting of ...”）。"
        "\n  关键词示例：propose, proposed, introduce, present, method, approach, system, architecture, pipeline, module, "
        "consisting of, joint classification and segmentation, bottleneck, GAMs, build on, based on, extend, extension, outperform。"
        "\n- 4分：强相关方法被深入比较/作为主要基线；多处 contexts 或 intents 显示方法层面依赖或对比。"
        "\n  关键词示例：baseline, compare against, improvement over, outperform, state-of-the-art (当作为对比基线使用)。"
        "\n- 3分：背景层面的关键进展，奠定本文方向，例如 “DL-based methods have achieved remarkable progress ...”、"
        "“state-of-the-art progress/remarkable improvement/significant improvement”等。"
        "\n- 2分：一般背景或综述性提及、与本文边缘相关。"
        "\n- 1分：顺带提及或与本文无关。"
        "\n作者影响力加权建议：若 maxAuthorCitation ≥ 5000，可作为强证据将评分上调 0.5~1（不超过5分）；若 1000 ≤ maxAuthorCitation < 5000，可小幅上调 0.5。"
        "但若上下文仅为泛化背景或与本文主题关联弱，则不要因高被引而过度拔高。"
        "\ncitation_role 判定："
        "\n- 方法借鉴：明确写到 based on / adopt / build on / inspired by / use ... method。"
        "\n- 性能比较：明确作为 baseline / compared with / outperform / versus / against。"
        "\n- 代表性工作：被列为某方向的代表方法、经典工作、里程碑进展。"
        "\n- 数据/资源引用：主要引用数据集、benchmark、标注资源、代码库。"
        "\n- 背景铺垫：一般背景、综述式进展描述。"
        "\n- 扩展应用：将目标论文方法迁移到新的任务或场景。"
        "\n- 弱相关：无上下文或仅顺带提及。"
        "\n要求："
        "\n1) 每条 INPUT 必须输出一行 NDJSON；"
        "\n2) w 只能写极短判断依据，例如“仅作泛背景提及”“明确作为比较基线”“引用数据集与标注”；"
        "\n3) 不要输出输入中已有的 title/authors/year/contexts/intents/isInfluential；"
        "\n4) 如果证据不足，也必须输出该条，给出保守分（例如1-2分）。"
    )
    user_lines: List[str] = []
    user_lines.append(
        f'TARGET: title="{target_meta.get("title","")}", '
        f'year={target_meta.get("year")}, venue="{target_meta.get("venue","")}"'
    )
    user_lines.append("INPUT:")
    for item in batch_items:
        ctxs = item.get("edge", {}).get("contexts") or []
        ctx_snips: List[str] = []
        for s in ctxs[:3]:
            s = (s or "").strip()
            if len(s) > 300:
                s = s[:300] + "…"
            ctx_snips.append(s)
        intents = item.get("edge", {}).get("intents") or []
        user_lines.append(json.dumps({
            "idx": item.get("refineIndex"),
            "title": item["citingPaper"].get("title"),
            "authorsWithCitations": item.get("authorsWithCitations"),
            "maxAuthorCitation": item.get("maxAuthorCitation"),
            "year": item["citingPaper"].get("year"),
            "venue": item["citingPaper"].get("venue"),
            "citationCount": item["citingPaper"].get("citationCount"),
            "isInfluential": item.get("edge", {}).get("isInfluential"),
            "intents": intents,
            "contexts": ctx_snips
        }, ensure_ascii=False))
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "\n".join(user_lines)}
    ]
    return messages
