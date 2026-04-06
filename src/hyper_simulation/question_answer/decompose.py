import json
import re

from langchain_ollama import ChatOllama

from hyper_simulation.hypergraph.hypergraph import Hypergraph
from hyper_simulation.llm.chat_completion import get_invoke, get_generate
from hyper_simulation.utils.log import getLogger

logger = getLogger(__name__)


def _extract_json_text(raw_text: str) -> str:
    """Extract JSON text from plain text or markdown fenced blocks."""
    text = (raw_text or "").strip()
    if not text:
        return ""

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj:end_obj + 1].strip()

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        return text[start_arr:end_arr + 1].strip()

    return text


def _build_vertex_context(query: Hypergraph) -> tuple[str, set[int]]:
    vertices = sorted(query.vertices, key=lambda v: v.id)
    valid_ids = {v.id for v in vertices}
    lines: list[str] = []
    for vertex in vertices:
        if vertex.is_verb() or vertex.is_virtual():
            continue
        text = vertex.text().replace("\n", " ").strip()
        if not text:
            text = "<EMPTY>"
        lines.append(f"- [{vertex.id}] {text}")
    return "\n".join(lines), valid_ids


def _normalize_result(parsed: object, valid_ids: set[int]) -> list[tuple[str, set[int]]]:
    if not isinstance(parsed, dict):
        return []

    items = parsed.get("subquestions", [])
    if not isinstance(items, list):
        return []

    result: list[tuple[str, set[int]]] = []
    answer_ids_by_step: list[set[int]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sub_q = str(item.get("question", "")).strip()
        if not sub_q:
            continue

        # New schema:
        # - local_vertex_ids: ids needed by this sub-question excluding #k references
        # - answer_vertex_ids: ids corresponding to this sub-question's answer entity/entities
        # Backward compatibility:
        # - if local_vertex_ids is missing, fall back to vertex_ids.
        raw_local_ids = item.get("local_vertex_ids", item.get("vertex_ids", []))
        if not isinstance(raw_local_ids, list):
            raw_local_ids = []

        raw_answer_ids = item.get("answer_vertex_ids", [])
        if not isinstance(raw_answer_ids, list):
            raw_answer_ids = []

        local_ids: set[int] = set()
        for value in raw_local_ids:
            try:
                vid = int(value)
            except (TypeError, ValueError):
                continue
            if vid in valid_ids:
                local_ids.add(vid)

        answer_ids: set[int] = set()
        for value in raw_answer_ids:
            try:
                vid = int(value)
            except (TypeError, ValueError):
                continue
            if vid in valid_ids:
                answer_ids.add(vid)

        # Resolve placeholders by importing only answer ids from referenced steps.
        placeholder_import_ids: set[int] = set()
        placeholders_in_q = set(re.findall(r"#(\d+)", sub_q))
        for ph_num in placeholders_in_q:
            ref_idx = int(ph_num) - 1
            if ref_idx < 0 or ref_idx >= len(answer_ids_by_step):
                continue
            placeholder_import_ids |= answer_ids_by_step[ref_idx]

        final_ids = local_ids | placeholder_import_ids
        result.append((sub_q, final_ids))
        answer_ids_by_step.append(answer_ids)

    return result


def _build_decompose_prompt(question: str, vertex_context: str) -> str:
    return f"""You are an expert query decomposer for multi-hop RAG.

Goal:
Rewrite one complex question into a sequence of ATOMIC sub-questions.
Each sub-question should be simple enough that a single document (or even one paragraph) can answer it.

Task:
1) Decompose the original question into minimal reasoning steps.
2) If a later step depends on an earlier answer, use placeholders #1, #2, ...
3) For each sub-question, output three parts:
    - question
    - local_vertex_ids: ids needed for this sub-question itself, excluding placeholder imports
    - answer_vertex_ids: ids corresponding to the answer entity/entities of this sub-question

Hard constraints (must follow):
- Atomicity: one sub-question = one fact lookup or one tiny reasoning step.
- Single-document answerability: avoid combining multiple entities/relations in one step.
- Prefer short, focused questions; do NOT keep broad conjunctions such as "A and B" in one sub-question.
- One relation per sub-question: each sub-question should ask only one predicate/relation.
- One target per sub-question: each sub-question should ask for one answer slot only.
- No nested clauses: avoid patterns like "X from Y who ..." or "A of B from C ..." in one question.
- If a question contains both an entity filter and a final ask, split them into two steps.
- Prefer 2-4 local_vertex_ids per sub-question; if >4, split further unless impossible.
- Keep dependency chain explicit with #k placeholders.
- Cover all reasoning needed for the original question.
- Only use vertex ids from the provided list.
- Output STRICT JSON only (no markdown, no explanation).
- `answer_vertex_ids` should be compact and point to the final referent of that step.
- `#k` refers ONLY to `answer_vertex_ids` of sub-question k.
- Never copy all local ids from step k when using #k.

Decomposition policy:
- If a question contains multiple hops, split by hops.
- If a hop still contains multiple facts (time + person + organization + location), split again.
- Separate "identify entity" from "query attribute of that entity".
- Separate "find bridge entity" from "final answer question".
- Separate "identify party/group" from "ask time/date/event".
- Separate "resolve location" from "ask action/time involving that location".
- If a sub-question can be rewritten as two WH-questions, you must split it.
- When creating a dependent sub-question like "What state is #1 from?":
    - local_vertex_ids should include only ids for "state" / "from" semantics in this step.
    - answer_vertex_ids should include ids for the answer of this step (the state entity).
- Keep dependency minimal: include only nodes required to answer THIS sub-question.
- Avoid history carry-over: previous step helper nodes must not be inherited unless they are directly asked in the current step.

Self-check before output:
- For each sub-question, ensure there is only one main WH intent (who/what/when/where/which).
- For each sub-question, ensure removing any clause does NOT leave another unanswered sub-goal.
- If a sub-question still contains two goals, split it again.

Placeholder usage rule:
- If a sub-question contains #k, do not encode #k mapping explicitly.
- Just make sure step k has correct `answer_vertex_ids`, because downstream will import them automatically.

Bad (too coarse):
- "In what year did the state where the 2001 Academy Award winner for Best Actor hails from attract the company founded by Bill Gates?"

Good (atomic):
- "Who won the Best Actor Oscar in 2001?"
- "What state is #1 from?"
- "What companies were founded by Bill Gates?"
- "When did #2 attract #3?"

Bad (still too composite):
- "When did voters from #2 vote for someone from Mayor Turner's party?"

Good (split version):
- "What party does Mayor Turner belong to?"
- "When did voters from #2 vote for someone from #3?"

JSON schema:
{{
    "subquestions": [
        {{
            "id": 1,
            "question": "...",
            "local_vertex_ids": [1, 2],
            "answer_vertex_ids": [2]
        }},
        {{
            "id": 2,
            "question": "...",
            "local_vertex_ids": [3],
            "answer_vertex_ids": [3]
        }}
    ]
}}

Original question:
{question.strip()}

Query hypergraph vertices:
{vertex_context}
"""


def _build_align_prompt(question: str, cleaned_subs: list[str], vertex_context: str) -> str:
    subs_text = "\n".join(f"{idx + 1}. {sub_q}" for idx, sub_q in enumerate(cleaned_subs))
    return f"""You are an expert query-to-hypergraph aligner for multi-hop RAG.

Goal:
The question decomposition is already given. Do NOT rewrite or merge/split the sub-questions.
You only need to assign vertex ids for each provided sub-question.

Task:
For each provided sub-question, output three parts:
- question: must be exactly the same text as the provided sub-question
- local_vertex_ids: ids needed for this sub-question itself, excluding placeholder imports
- answer_vertex_ids: ids corresponding to the answer entity/entities of this sub-question

Rules:
- Keep the same order and count as provided.
- If a sub-question contains #k, `#k` refers to answer_vertex_ids of sub-question k.
- Do not copy all local ids from sub-question k when processing #k references.
- Use only ids from the provided vertex list.
- Output STRICT JSON only.

JSON schema:
{{
  "subquestions": [
    {{"id": 1, "question": "...", "local_vertex_ids": [1], "answer_vertex_ids": [1]}},
    {{"id": 2, "question": "...", "local_vertex_ids": [3], "answer_vertex_ids": [3]}}
  ]
}}

Original question:
{question.strip()}

Provided sub-questions (keep unchanged):
{subs_text}

Query hypergraph vertices:
{vertex_context}
"""


def _parse_and_normalize(raw_output: str | list, valid_ids: set[int]) -> list[tuple[str, set[int]]]:
    if isinstance(raw_output, str):
        raw_text = raw_output
    else:
        raw_text = "\n".join(str(part) for part in raw_output)
    payload = _extract_json_text(raw_text)
    parsed = json.loads(payload)
    return _normalize_result(parsed, valid_ids)


def decompose_question(question: str, query: Hypergraph) -> list[tuple[str, set[int]]]:
    """Decompose a multi-hop question into sub-questions with related vertex ids."""
    if not question or not question.strip():
        return []

    vertex_context, valid_ids = _build_vertex_context(query)
    fallback = [(question.strip(), set(valid_ids))]

    prompt = _build_decompose_prompt(question, vertex_context)

    try:
        llm = ChatOllama(model="qwen3.5:9b", temperature=0.2, top_p=0.95, reasoning=False)
        raw_output = get_invoke(llm, prompt)
        normalized = _parse_and_normalize(raw_output, valid_ids)
        if normalized:
            return normalized
        logger.warning("decompose_question returned empty normalized output, using fallback")
        return fallback
    except Exception as exc:
        logger.warning(f"decompose_question failed: {type(exc).__name__}: {exc}, using fallback")
        return fallback

def decompose_question_with_subs(question: str, subs: list[str], query: Hypergraph) -> list[tuple[str, set[int]]]:
    """Map provided sub-questions to related vertex ids without re-decomposing the question."""
    if not question or not question.strip():
        return []

    cleaned_subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
    if not cleaned_subs:
        return decompose_question(question, query)

    vertex_context, valid_ids = _build_vertex_context(query)
    fallback = [(sub_q, set(valid_ids)) for sub_q in cleaned_subs]

    prompt = _build_align_prompt(question, cleaned_subs, vertex_context)

    try:
        llm = ChatOllama(model="qwen3.5:9b", temperature=0.2, top_p=0.95, reasoning=False)
        raw_output = get_invoke(llm, prompt)
        normalized = _parse_and_normalize(raw_output, valid_ids)
        if normalized:
            return normalized
        logger.warning("decompose_question_with_subs returned empty normalized output, using fallback")
        return fallback
    except Exception as exc:
        logger.warning(f"decompose_question_with_subs failed: {type(exc).__name__}: {exc}, using fallback")
        return fallback


def decompose_question_batch(
    questions: list[str],
    queries: list[Hypergraph],
) -> list[list[tuple[str, set[int]]]]:
    """Batch version of decompose_question."""
    if len(questions) != len(queries):
        raise ValueError(
            f"questions and queries must have same length, got {len(questions)} and {len(queries)}"
        )

    prompts: list[str] = []
    valid_ids_list: list[set[int]] = []
    fallbacks: list[list[tuple[str, set[int]]]] = []

    for question, query in zip(questions, queries):
        if not question or not question.strip():
            prompts.append("")
            valid_ids_list.append(set())
            fallbacks.append([])
            continue
        vertex_context, valid_ids = _build_vertex_context(query)
        prompts.append(_build_decompose_prompt(question, vertex_context))
        valid_ids_list.append(valid_ids)
        fallbacks.append([(question.strip(), set(valid_ids))])

    llm = ChatOllama(model="qwen3.5:9b", temperature=0.2, top_p=0.95, reasoning=False)
    non_empty_idx = [i for i, p in enumerate(prompts) if p]
    prompt_payload = [prompts[i] for i in non_empty_idx]

    raw_outputs: list[str] = []
    if prompt_payload:
        raw_outputs = get_generate(prompt_payload, llm)

    results = list(fallbacks)
    output_cursor = 0
    for idx in non_empty_idx:
        try:
            normalized = _parse_and_normalize(raw_outputs[output_cursor], valid_ids_list[idx])
            if normalized:
                results[idx] = normalized
        except Exception as exc:
            logger.warning(
                f"decompose_question_batch item {idx} failed: {type(exc).__name__}: {exc}, using fallback"
            )
        output_cursor += 1

    return results


def decompose_question_with_subs_batch(
    questions: list[str],
    subs_batch: list[list[str]],
    queries: list[Hypergraph],
) -> list[list[tuple[str, set[int]]]]:
    """Batch version of decompose_question_with_subs."""
    if len(questions) != len(subs_batch) or len(questions) != len(queries):
        raise ValueError(
            "questions, subs_batch, and queries must have same length, "
            f"got {len(questions)}, {len(subs_batch)}, {len(queries)}"
        )

    prompts: list[str] = []
    valid_ids_list: list[set[int]] = []
    fallbacks: list[list[tuple[str, set[int]]]] = []

    for question, subs, query in zip(questions, subs_batch, queries):
        cleaned_subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
        if not question or not question.strip():
            prompts.append("")
            valid_ids_list.append(set())
            fallbacks.append([])
            continue
        if not cleaned_subs:
            vertex_context, valid_ids = _build_vertex_context(query)
            prompts.append(_build_decompose_prompt(question, vertex_context))
            valid_ids_list.append(valid_ids)
            fallbacks.append([(question.strip(), set(valid_ids))])
            continue

        vertex_context, valid_ids = _build_vertex_context(query)
        prompts.append(_build_align_prompt(question, cleaned_subs, vertex_context))
        valid_ids_list.append(valid_ids)
        fallbacks.append([(sub_q, set(valid_ids)) for sub_q in cleaned_subs])

    llm = ChatOllama(model="qwen3.5:9b", temperature=0.2, top_p=0.95, reasoning=False)
    non_empty_idx = [i for i, p in enumerate(prompts) if p]
    prompt_payload = [prompts[i] for i in non_empty_idx]

    raw_outputs: list[str] = []
    if prompt_payload:
        raw_outputs = get_generate(prompt_payload, llm)

    results = list(fallbacks)
    output_cursor = 0
    for idx in non_empty_idx:
        try:
            normalized = _parse_and_normalize(raw_outputs[output_cursor], valid_ids_list[idx])
            if normalized:
                results[idx] = normalized
        except Exception as exc:
            logger.warning(
                f"decompose_question_with_subs_batch item {idx} failed: {type(exc).__name__}: {exc}, using fallback"
            )
        output_cursor += 1

    return results