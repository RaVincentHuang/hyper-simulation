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

        # Final ids for this sub-question are composed of:
        # 1) current local_vertex_ids (without placeholder imports)
        # 2) current answer_vertex_ids
        # 3) referenced answer_vertex_ids from #k placeholders
        final_ids = local_ids | answer_ids | placeholder_import_ids
        result.append((sub_q, final_ids))
        answer_ids_by_step.append(answer_ids)

    return result


def _build_decompose_prompt(question: str, vertex_context: str) -> str:
    return f"""[Core task] Assign PRECISE vertex ids for each sub-question.

Question:
{question.strip()}

Available query vertices (id -> text):
{vertex_context}

You decompose one multi-hop question for RAG.

Output STRICT JSON:
{{
  "subquestions": [
    {{"id": 1, "question": "...", "local_vertex_ids": [..], "answer_vertex_ids": [..]}}
  ]
}}

Field definitions (precise):
- local_vertex_ids: vertices needed to interpret/answer this sub-question itself (excluding #k imports).
- answer_vertex_ids: vertices that correspond to the final answer entity/entities of this sub-question.

Requirements:
1) Produce atomic sub-questions: one relation + one answer target per step.
2) Keep each step single-document answerable.
3) Use #1, #2 only for dependencies.
4) #k refers ONLY to answer_vertex_ids of step k, never all ids of step k.
5) local_vertex_ids = ids needed for this step itself (exclude placeholder imports).
6) answer_vertex_ids = ids of the answer entity/entities of this step.
7) Prefer concise, natural WH questions.
8) Use only ids from the provided vertex list.

Important:
- Precision is the priority. Do not guess ids outside the list.
- Every id in local_vertex_ids / answer_vertex_ids must be necessary.
"""


def _build_align_prompt(question: str, cleaned_subs: list[str], vertex_context: str) -> str:
    subs_text = "\n".join(f"- [{idx + 1}] {sub_q}" for idx, sub_q in enumerate(cleaned_subs))
    return f"""Our objective is simple: we already have sub-questions and query vertices.
For each given sub-question, identify which vertices are explicitly mentioned or semantically involved in answering it, then return their ids.
This is a vertex-alignment task, not a question-generation task.

[Purpose]
This call is for ALIGNMENT, not free rewriting.
Your primary goal is to assign accurate, non-empty vertex ids for every provided sub-question.

[How to operate]
Step 1) Read the question and vertex list.
Step 2) Process each sub-question in order.
Step 3) Rewrite ONLY if the line is shorthand `A >> B`; otherwise keep text unchanged.
Step 4) Assign local_vertex_ids and answer_vertex_ids for each line.
Step 5) Ensure no sub-question has empty id lists.

[Core task] Assign PRECISE vertex ids for each provided sub-question.
Question:
{question.strip()}

Available query vertices ( - [id] vertex):
{vertex_context}

Provided sub-questions (- [id] sub-question):
{subs_text}

You align provided sub-questions to query vertices.

Output STRICT JSON:
{{
    "subquestions": [
        {{"id": 1, "question": "...", "local_vertex_ids": [..], "answer_vertex_ids": [..]}}
    ]
}}

Field definitions (precise):
- local_vertex_ids: vertices needed to interpret/answer this sub-question itself (excluding #k imports).
- answer_vertex_ids: vertices that correspond to the final answer entity/entities of this sub-question.

Core policy (focus on this):
1) Keep EXACT same order and EXACT same number of sub-questions.
2) Minimal edit:
    - ONLY rewrite shorthand `A >> B` into a natural question.
    - For non-`A >> B` sub-questions, keep text EXACTLY unchanged (character-level).
3) Preserve placeholders:
    - Keep #1, #2 as placeholders exactly.
    - Never replace placeholders with entity names or descriptions.
    - Never inject information from other sub-questions.
4) If #k appears, it refers to answer_vertex_ids of step k.
5) local_vertex_ids = ids needed by this step itself.
6) answer_vertex_ids = ids of this step's answer entity/entities.
7) Use only ids from provided vertices.
8) Do not change question intent (e.g., uncle->father, lead singer->performer is forbidden).
9) If uncertain, preserve the original sub-question text and only assign ids.
10) Non-empty marking is mandatory for every sub-question:
    - local_vertex_ids must contain at least 1 id.
    - answer_vertex_ids must contain at least 1 id.
    - If confidence is low, choose the single most relevant candidate id rather than leaving empty.
11) answer_vertex_ids should point to the answer-bearing node(s), not random context nodes.

Shorthand examples:
- `John Phan >> place of birth` -> `Where was John Phan born?`
- `#1 >> country` -> `Which country is #1 in?`

Forbidden examples:
- `Who is the uncle of Liu Bin?` -> `Who is the father of Liu Bin?` (forbidden)
- `Who is the lead singer ...` -> `Who is the performer ...` (forbidden)
- `On what date did Battle of #1 end?` -> expanding `#1` (forbidden)

Important:
- Precision is the priority. Do not guess ids outside the list.
- local_vertex_ids / answer_vertex_ids should be minimal but sufficient.
- Empty ids are not allowed.
"""


def _is_shorthand_subquestion(sub_q: str) -> bool:
    return bool(re.match(r"^\s*[^\n]+\s*>>\s*[^\n]+\s*$", sub_q))


def _build_rewrite_shorthand_prompt(question: str, subs: list[str]) -> str:
    subs_text = "\n".join(f"- [{idx + 1}] {sub_q}" for idx, sub_q in enumerate(subs))
    return f"""[Core task] Rewrite ONLY shorthand sub-questions of form `A >> B`.

Question:
{question.strip()}

Input sub-questions:
{subs_text}

Rules:
1) Keep same order and same number of sub-questions.
2) ONLY rewrite lines matching `A >> B` into natural, answerable questions.
3) For non-`A >> B` lines, keep text EXACTLY unchanged.
4) Preserve placeholders (#1, #2, ...) exactly. Never expand them.
5) Do not merge, split, or paraphrase non-shorthand lines.
6) Keep semantics faithful to the original question context.
7) Keep each rewritten question short, direct, and single-intent.
8) Prefer standard WH form (Who/What/When/Where/Which) based on attribute type.
9) Do not add extra constraints, entities, dates, or clauses not implied by `A >> B`.

Detailed rewrite guideline for `A >> B`:
- `A` is the subject/entity expression (can include placeholders like #1).
- `B` is the target relation/attribute.
- Convert to: "[WH-word] [relation about A]?"
- Choose WH word by B:
    - place/location/country/city/region/origin -> Where / Which country / Which city
    - date/time/year/birth date/death date -> When / In what year / On what date
    - person/leader/founder/author -> Who
    - language/party/religion/type/category -> What / Which

Allowed:
- grammatical cleanup of shorthand
- light relation normalization (e.g., "place of birth" -> "born")

Forbidden:
- replacing #k with entity descriptions
- injecting context from other sub-questions into current line
- changing the intent of non-shorthand lines

Examples (good):
- `John Phan >> place of birth` -> `Where was John Phan born?`
- `Mount Can >> country` -> `Which country is Mount Can in?`
- `#1 >> country` -> `Which country is #1 in?`
- `Battle of #1 >> end date` -> `On what date did Battle of #1 end?`
- `the era of #1 >> language` -> `What was the language of #1's era?`

Examples (must keep unchanged because not shorthand):
- `On what date did Battle of #1 end?` -> unchanged
- `What was the #2 of #1's era later known as?` -> unchanged
- `Who was crowned the new Roman emperor in A.D. 800?` -> unchanged

Counterexamples (bad rewrites):
- `On what date did Battle of #1 end?` -> `On what date did the battle at Choo Hoey's birthplace end?` (BAD: expanded #1)
- `#1 >> country` -> `Which country contains Mount Can?` (BAD: replaced placeholder)
- `A >> B` -> very long multi-clause question (BAD: not single-intent)

Output STRICT JSON:
{{
  "subquestions": [
    {{"id": 1, "question": "..."}},
    {{"id": 2, "question": "..."}},
    ...
  ]
}}
"""


def _parse_rewritten_subs(raw_output: str | list, original_subs: list[str]) -> list[str]:
    if isinstance(raw_output, str):
        raw_text = raw_output
    else:
        raw_text = "\n".join(str(part) for part in raw_output)

    try:
        payload = _extract_json_text(raw_text)
        parsed = json.loads(payload)
        items = parsed.get("subquestions", []) if isinstance(parsed, dict) else []
        if not isinstance(items, list):
            return original_subs

        rewritten: list[str] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                rewritten.append(original_subs[idx] if idx < len(original_subs) else "")
                continue
            q = str(item.get("question", "")).strip()
            rewritten.append(q if q else (original_subs[idx] if idx < len(original_subs) else ""))

        # Keep exact count/order by padding/truncating against original_subs.
        if len(rewritten) < len(original_subs):
            rewritten.extend(original_subs[len(rewritten):])
        if len(rewritten) > len(original_subs):
            rewritten = rewritten[: len(original_subs)]
        return rewritten
    except Exception:
        return original_subs


def _build_align_only_prompt(question: str, subs: list[str], vertex_context: str) -> str:
    subs_text = "\n".join(f"- [{idx + 1}] {sub_q}" for idx, sub_q in enumerate(subs))
    return f"""Our objective is simple: we already have sub-questions and query vertices.
For each given sub-question, identify which vertices are mentioned or involved in answering it, then return their ids.
This is a vertex-alignment task, not a question-generation task.

[Purpose]
This call is ALIGN-ONLY.
Do not rewrite the sub-questions; only mark precise, non-empty vertex ids.

[How to operate]
Step 1) Keep each sub-question text exactly as given.
Step 2) For each line, assign local_vertex_ids.
Step 3) For each line, assign answer_vertex_ids.
Step 4) Verify both id lists are non-empty for every line.

[HINT]
The sub-questions could include placeholders like #1, #2, etc. 
These placeholders refer to answer entities from previous sub-questions.

[Core task] Assign PRECISE vertex ids for each provided sub-question.
Question:
{question.strip()}

Available query vertices (- [id] vertex):
{vertex_context}

Provided sub-questions (already normalized):
{subs_text}

Rules:
1) Keep EXACT same order and EXACT same number of sub-questions.
2) Do NOT rewrite any sub-question text in this stage.
3) Output `question` MUST be exactly the same as input line i (character-level).
4) Preserve placeholders exactly (#1, #2, ...). Never expand or replace them.
5) If #k appears, it refers to answer_vertex_ids of step k.
6) local_vertex_ids = ids needed by this step itself.
7) answer_vertex_ids = ids of this step's answer entity/entities.
8) Use only ids from provided vertices.
9) Keep ids minimal but sufficient.
10) If uncertain, keep original text unchanged and return best-effort ids; never drop a sub-question.
11) Non-empty marking is mandatory for every sub-question:
    - local_vertex_ids must contain at least 1 id.
    - answer_vertex_ids must contain at least 1 id.
    - If confidence is low, choose one best candidate id rather than empty.

Hard failure conditions to avoid:
- returning fewer subquestions than input
- changing question meaning
- replacing placeholders with entity text
- returning empty local_vertex_ids or empty answer_vertex_ids

Field definitions (precise):
- local_vertex_ids: vertices needed to interpret/answer this sub-question itself (excluding #1, #2, ... imports).
- answer_vertex_ids: vertices that correspond to the final answer entity/entities of this sub-question.

Example:
- Question: "When did voters from the state of the most successful American Idol contestant this season is from once again vote for someone from Mayor Turner's party?"
- Available query vertices:
    - [0] '?When'
    - [1] 'voters'
    - [2] 'state'
    - [3] 'most successful'
    - [4] 'American Idol'
    - [5] 'contestant'
    - [6] 'this season'
    - [10] 'Mayor'
    - [11] 'Turner'
    - [12] 'party'
- Provided sub-questions:
    - [1] Who was the most successful artist from this season?
    - [2] What state is American Idol contestant #1 from?
    - [3] To what political party is Mayor Turner aligned?
    - [4] What year did #2 voters once again vote for a #3 ?
{{"id": 1, "question": "Who was the most successful artist from this season?", "local_vertex_ids": [3, 6], "answer_vertex_ids": [5]}},
{{"id": 2, "question": "What state is American Idol contestant #1 from?", "local_vertex_ids": [4], "answer_vertex_ids": [3]}},
{{"id": 3, "question": "To what political party is Mayor Turner aligned?", "local_vertex_ids": [10, 11], "answer_vertex_ids": [12]}},
{{"id": 4, "question": "What year did #2 voters once again vote for a #3 ?", "local_vertex_ids": [1], "answer_vertex_ids": [0]}}

Output STRICT JSON:
{{
  "subquestions": [
    {{"id": 1, "question": "...", "local_vertex_ids": [..], "answer_vertex_ids": [..]}},
    {{"id": 2, "question": "...", "local_vertex_ids": [..], "answer_vertex_ids": [..]}}, 
    ...
  ]
}}
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
        llm = ChatOllama(model="qwen3.5:9b", temperature=0.1, top_p=1, reasoning=False)
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
        llm = ChatOllama(model="qwen3.5:9b", temperature=0.1, top_p=1, reasoning=False)
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

    llm = ChatOllama(model="qwen3.5:9b", temperature=0.1, top_p=1, reasoning=False)
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

    valid_ids_list: list[set[int]] = []
    cleaned_subs_list: list[list[str]] = []
    fallbacks: list[list[tuple[str, set[int]]]] = []
    rewrite_prompts: list[str] = []
    rewrite_indices: list[int] = []

    # Stage 0: prepare common context and determine which samples need shorthand rewrite.
    for idx, (question, subs, query) in enumerate(zip(questions, subs_batch, queries)):
        cleaned_subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
        cleaned_subs_list.append(cleaned_subs)

        if not question or not question.strip():
            valid_ids_list.append(set())
            fallbacks.append([])
            continue

        _, valid_ids = _build_vertex_context(query)
        valid_ids_list.append(valid_ids)

        if not cleaned_subs:
            fallbacks.append([(question.strip(), set(valid_ids))])
            continue

        fallbacks.append([(sub_q, set(valid_ids)) for sub_q in cleaned_subs])
        if any(_is_shorthand_subquestion(sub_q) for sub_q in cleaned_subs):
            rewrite_prompts.append(_build_rewrite_shorthand_prompt(question, cleaned_subs))
            rewrite_indices.append(idx)

    llm = ChatOllama(model="qwen3.5:9b", temperature=0.1, top_p=1, reasoning=False)

    # Stage 1: rewrite only shorthand A >> B sub-questions.
    rewritten_subs_list = [list(subs) for subs in cleaned_subs_list]
    if rewrite_prompts:
        rewrite_outputs = get_generate(rewrite_prompts, llm)
        for out_idx, sample_idx in enumerate(rewrite_indices):
            original_subs = cleaned_subs_list[sample_idx]
            rewritten = _parse_rewritten_subs(rewrite_outputs[out_idx], original_subs)
            # Safety: enforce minimal-edit policy post-hoc.
            guarded: list[str] = []
            for old_q, new_q in zip(original_subs, rewritten):
                if _is_shorthand_subquestion(old_q):
                    guarded.append(new_q if new_q else old_q)
                else:
                    guarded.append(old_q)
            rewritten_subs_list[sample_idx] = guarded

    # Stage 2: align ids based on normalized sub-questions.
    align_prompts: list[str] = []
    align_indices: list[int] = []
    for idx, (question, query) in enumerate(zip(questions, queries)):
        if not question or not question.strip():
            continue

        current_subs = rewritten_subs_list[idx]
        if not current_subs:
            vertex_context, _ = _build_vertex_context(query)
            align_prompts.append(_build_decompose_prompt(question, vertex_context))
            align_indices.append(idx)
            continue

        vertex_context, _ = _build_vertex_context(query)
        align_prompts.append(_build_align_only_prompt(question, current_subs, vertex_context))
        align_indices.append(idx)

    results = list(fallbacks)
    if align_prompts:
        align_outputs = get_generate(align_prompts, llm)
        for out_idx, sample_idx in enumerate(align_indices):
            try:
                normalized = _parse_and_normalize(align_outputs[out_idx], valid_ids_list[sample_idx])
                if normalized:
                    results[sample_idx] = normalized
            except Exception as exc:
                logger.warning(
                    f"decompose_question_with_subs_batch item {sample_idx} failed: {type(exc).__name__}: {exc}, using fallback"
                )

    return results


def decompose_question_with_nums(question: str, num: int) -> list[str]:
    if not question or not question.strip() or num <= 0:
        return []

    prompt = f"""You are decomposing one MultiHop QA query for retrieval-augmented generation.

Question:
{question.strip()}

Target:
- Decompose the question into exactly {num} sub-questions.
- Each sub-question should be directly answerable by existing wiki/news/web documents.
- Keep each step single-intent and concise.
- Do NOT include numbering in the question text.

Output STRICT JSON:
{{
  "subquestions": [
    {{"id": 1, "question": "..."}},
    {{"id": 2, "question": "..."}}
  ]
}}
"""

    fallback = [question.strip() for _ in range(num)]

    try:
        llm = ChatOllama(model="qwen3.5:9b", temperature=0.1, top_p=1, reasoning=False)
        raw_output = get_invoke(llm, prompt)
        payload = _extract_json_text(str(raw_output))
        parsed = json.loads(payload)
        items = parsed.get("subquestions", []) if isinstance(parsed, dict) else []

        cleaned: list[str] = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                sub_q = str(item.get("question", "")).strip()
                if sub_q:
                    cleaned.append(sub_q)

        if not cleaned:
            return fallback

        if len(cleaned) < num:
            cleaned.extend([question.strip()] * (num - len(cleaned)))
        if len(cleaned) > num:
            cleaned = cleaned[:num]
        return cleaned
    except Exception as exc:
        logger.warning(
            f"decompose_question_with_nums failed: {type(exc).__name__}: {exc}, using fallback"
        )
        return fallback


def decompose_question_with_nums_batch(questions: list[str], nums: list[int]) -> list[list[str]]:
    """Batch version of decompose_question_with_nums."""
    if len(questions) != len(nums):
        raise ValueError(
            f"questions and nums must have same length, got {len(questions)} and {len(nums)}"
        )

    prompts: list[str] = []
    indices: list[int] = []
    results: list[list[str]] = []

    for idx, (question, num) in enumerate(zip(questions, nums)):
        if not question or not question.strip() or num <= 0:
            results.append([])
            continue

        prompts.append(
            f"""You are decomposing one MultiHop QA query for retrieval-augmented generation.

Question:
{question.strip()}

Target:
- Decompose the question into exactly {num} sub-questions.
- Each sub-question should be directly answerable by existing wiki/news/web documents.
- Keep each step single-intent and concise.
- Do NOT include numbering in the question text.

Output STRICT JSON:
{{
  "subquestions": [
    {{"id": 1, "question": "..."}},
    {{"id": 2, "question": "..."}}
  ]
}}
"""
        )
        indices.append(idx)
        results.append([question.strip() for _ in range(num)])

    if not prompts:
        return results

    llm = ChatOllama(model="qwen3.5:9b", temperature=0.1, top_p=1, reasoning=False)
    raw_outputs = get_generate(prompts, llm)

    for out_idx, sample_idx in enumerate(indices):
        question = questions[sample_idx]
        num = nums[sample_idx]
        try:
            payload = _extract_json_text(str(raw_outputs[out_idx]))
            parsed = json.loads(payload)
            items = parsed.get("subquestions", []) if isinstance(parsed, dict) else []

            cleaned: list[str] = []
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    sub_q = str(item.get("question", "")).strip()
                    if sub_q:
                        cleaned.append(sub_q)

            if cleaned:
                if len(cleaned) < num:
                    cleaned.extend([question.strip()] * (num - len(cleaned)))
                if len(cleaned) > num:
                    cleaned = cleaned[:num]
                results[sample_idx] = cleaned
        except Exception as exc:
            logger.warning(
                "decompose_question_with_nums_batch item "
                f"{sample_idx} failed: {type(exc).__name__}: {exc}, using fallback"
            )

    return results

def mark_vertex_ids_for_subquestions(question: str, subs: list[str], query: Hypergraph) -> list[tuple[str, set[int]]]:
    if not question or not question.strip():
        return []

    cleaned_subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
    if not cleaned_subs:
        return []

    vertex_context, valid_ids = _build_vertex_context(query)
    fallback = _mark_vertex_ids_rule_fallback(cleaned_subs, query)
    subs_text = "\n".join(f"- [{idx + 1}] {sub_q}" for idx, sub_q in enumerate(cleaned_subs))

    prompt = f"""Task: Mark which query vertices appear in each given sub-question.

Original question:
{question.strip()}

Available query vertices (- [id] vertex):
{vertex_context}

Given sub-questions (keep order and text unchanged):
{subs_text}

Rules:
1) Do NOT rewrite or reorder any sub-question.
2) For each sub-question, return ids of vertices that are explicitly mentioned or clearly referenced.
3) If a placeholder like #1 appears, include ids corresponding to #1's answer from previous step.
4) Use only ids from the provided vertices.
5) Keep ids minimal but sufficient.

Output STRICT JSON:
{{
  "subquestions": [
    {{"id": 1, "question": "...", "vertex_ids": [..]}},
    {{"id": 2, "question": "...", "vertex_ids": [..]}}
  ]
}}
"""

    try:
        llm = ChatOllama(model="qwen3.5:9b", temperature=0.1, top_p=1, reasoning=False)
        raw_output = get_invoke(llm, prompt)
        normalized = _parse_mark_vertex_output(raw_output, cleaned_subs, valid_ids)
        if normalized:
            return normalized
        return fallback
    except Exception as exc:
        logger.warning(
            f"mark_vertex_ids_for_subquestions failed: {type(exc).__name__}: {exc}, using fallback"
        )
        return fallback


def mark_vertex_ids_for_subquestions_batch(
    questions: list[str],
    subs_batch: list[list[str]],
    queries: list[Hypergraph],
) -> list[list[tuple[str, set[int]]]]:
    """Batch version of mark_vertex_ids_for_subquestions."""
    if len(questions) != len(subs_batch) or len(questions) != len(queries):
        raise ValueError(
            "questions, subs_batch, and queries must have same length, "
            f"got {len(questions)}, {len(subs_batch)}, {len(queries)}"
        )

    prompts: list[str] = []
    valid_ids_list: list[set[int]] = []
    cleaned_subs_list: list[list[str]] = []
    non_empty_idx: list[int] = []
    results: list[list[tuple[str, set[int]]]] = []

    for idx, (question, subs, query) in enumerate(zip(questions, subs_batch, queries)):
        if not question or not question.strip():
            valid_ids_list.append(set())
            cleaned_subs_list.append([])
            results.append([])
            continue

        cleaned_subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
        cleaned_subs_list.append(cleaned_subs)
        vertex_context, valid_ids = _build_vertex_context(query)
        valid_ids_list.append(valid_ids)

        fallback = _mark_vertex_ids_rule_fallback(cleaned_subs, query)
        results.append(fallback)

        if not cleaned_subs:
            continue

        subs_text = "\n".join(f"- [{i + 1}] {sub_q}" for i, sub_q in enumerate(cleaned_subs))

        prompt = f"""Task: Mark which query vertices appear in each given sub-question.

Original question:
{question.strip()}

Available query vertices (- [id] vertex):
{vertex_context}

Given sub-questions (keep order and text unchanged):
{subs_text}

Rules:
1) Do NOT rewrite or reorder any sub-question.
2) For each sub-question, return ids of vertices that are explicitly mentioned or clearly referenced.
3) If a placeholder like #1 appears, include ids corresponding to #1's answer from previous step.
4) Use only ids from the provided vertices.
5) Keep ids minimal but sufficient.

Output STRICT JSON:
{{
  "subquestions": [
    {{"id": 1, "question": "...", "vertex_ids": [..]}},
    {{"id": 2, "question": "...", "vertex_ids": [..]}}
  ]
}}
"""
        prompts.append(prompt)
        non_empty_idx.append(idx)

    if not prompts:
        return results

    llm = ChatOllama(model="qwen3.5:9b", temperature=0.1, top_p=1, reasoning=False)
    raw_outputs = get_generate(prompts, llm)

    for out_idx, sample_idx in enumerate(non_empty_idx):
        try:
            normalized = _parse_mark_vertex_output(
                raw_outputs[out_idx],
                cleaned_subs_list[sample_idx],
                valid_ids_list[sample_idx],
            )
            if normalized:
                results[sample_idx] = normalized
        except Exception as exc:
            logger.warning(
                "mark_vertex_ids_for_subquestions_batch item "
                f"{sample_idx} failed: {type(exc).__name__}: {exc}, using fallback"
            )

    return results


def _parse_mark_vertex_output(
    raw_output: str | list,
    original_subs: list[str],
    valid_ids: set[int],
) -> list[tuple[str, set[int]]]:
    if isinstance(raw_output, str):
        raw_text = raw_output
    else:
        raw_text = "\n".join(str(part) for part in raw_output)

    payload = _extract_json_text(raw_text)
    parsed = json.loads(payload)
    items = parsed.get("subquestions", []) if isinstance(parsed, dict) else []
    if not isinstance(items, list):
        return []

    results: list[tuple[str, set[int]]] = []
    previous_answer_like_ids: list[set[int]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        original_q = original_subs[idx] if idx < len(original_subs) else ""
        sub_q = str(item.get("question", "")).strip() or original_q
        if not sub_q:
            continue

        candidate_ids: set[int] = set()
        raw_vertex_ids = item.get("vertex_ids", [])
        if isinstance(raw_vertex_ids, list):
            for value in raw_vertex_ids:
                try:
                    vid = int(value)
                except (TypeError, ValueError):
                    continue
                if vid in valid_ids:
                    candidate_ids.add(vid)

        # Backward-compatible parse for schemas using local/answer split.
        raw_local_ids = item.get("local_vertex_ids", [])
        if isinstance(raw_local_ids, list):
            for value in raw_local_ids:
                try:
                    vid = int(value)
                except (TypeError, ValueError):
                    continue
                if vid in valid_ids:
                    candidate_ids.add(vid)

        answer_like_ids: set[int] = set()
        raw_answer_ids = item.get("answer_vertex_ids", [])
        if isinstance(raw_answer_ids, list):
            for value in raw_answer_ids:
                try:
                    vid = int(value)
                except (TypeError, ValueError):
                    continue
                if vid in valid_ids:
                    answer_like_ids.add(vid)
                    candidate_ids.add(vid)

        # Resolve placeholder references via previous answer-like ids.
        placeholders = set(re.findall(r"#(\d+)", sub_q))
        for ph in placeholders:
            ref_idx = int(ph) - 1
            if 0 <= ref_idx < len(previous_answer_like_ids):
                candidate_ids |= previous_answer_like_ids[ref_idx]

        results.append((sub_q, candidate_ids))
        previous_answer_like_ids.append(answer_like_ids)

    if len(results) < len(original_subs):
        # Keep order/count stable if model drops lines.
        existing = {q for q, _ in results}
        for sub_q in original_subs:
            if sub_q in existing:
                continue
            results.append((sub_q, set()))

    if len(results) > len(original_subs):
        results = results[: len(original_subs)]

    return results


def _mark_vertex_ids_rule_fallback(subs: list[str], query: Hypergraph) -> list[tuple[str, set[int]]]:
    vertices = sorted(query.vertices, key=lambda v: v.id)
    usable_vertices = [v for v in vertices if not v.is_verb() and not v.is_virtual()]

    fallback: list[tuple[str, set[int]]] = []
    answer_like_ids: list[set[int]] = []
    for sub_q in subs:
        sub_norm = " ".join(sub_q.lower().split())
        tokens = set(re.findall(r"[a-z0-9]+", sub_norm))
        ids: set[int] = set()

        for vertex in usable_vertices:
            v_text = " ".join(vertex.text().lower().split())
            if not v_text:
                continue

            if v_text in sub_norm:
                ids.add(vertex.id)
                continue

            v_tokens = set(re.findall(r"[a-z0-9]+", v_text))
            if v_tokens and len(v_tokens) <= 3 and v_tokens.issubset(tokens):
                ids.add(vertex.id)

        placeholders = set(re.findall(r"#(\d+)", sub_q))
        for ph in placeholders:
            ref_idx = int(ph) - 1
            if 0 <= ref_idx < len(answer_like_ids):
                ids |= answer_like_ids[ref_idx]

        fallback.append((sub_q, ids))
        answer_like_ids.append(set(ids))

    return fallback