from typing import Any
import json
import re

from langchain_ollama import ChatOllama


def _normalize_response_content(raw_content: Any) -> str:
	if isinstance(raw_content, str):
		return raw_content
	if isinstance(raw_content, list):
		return "\n".join(
			item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
			for item in raw_content
		)
	return str(raw_content)


def _extract_json_payload(content: str) -> dict[str, Any] | None:
	try:
		parsed = json.loads(content)
		if isinstance(parsed, dict):
			return parsed
	except Exception:
		pass

	left = content.find("{")
	right = content.rfind("}")
	if left == -1 or right == -1 or right <= left:
		return None

	try:
		parsed = json.loads(content[left : right + 1])
		if isinstance(parsed, dict):
			return parsed
	except Exception:
		return None
	return None


def _compact_prefix(prefix: str, title: str, max_words: int = 8) -> str:
	"""Force prefix to stay very short and clause-like."""
	clean = " ".join(prefix.replace("\n", " ").split()).strip()
	if not clean:
		return f"Regarding {title},"

	words = clean.split(" ")
	if len(words) > max_words:
		clean = " ".join(words[:max_words]).rstrip(",;:-")

	# Convert into a connective clause instead of standalone sentence/label.
	if clean.endswith((".", "!", "?", ":", ";")):
		clean = clean[:-1].rstrip()
	if not clean.endswith(","):
		clean = clean.rstrip(",") + ","

	return clean


def _norm_text(text: str) -> str:
	return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _has_title_signal(prefix: str, title: str) -> bool:
	"""Require that prefix explicitly carries title signal (full or fuzzy token overlap)."""
	n_prefix = _norm_text(prefix)
	n_title = _norm_text(title)
	if not n_prefix or not n_title:
		return False
	if n_title in n_prefix:
		return True
	tokens = [t for t in n_title.split() if len(t) >= 4]
	if not tokens:
		return False
	matched = sum(1 for t in tokens if t in n_prefix)
	required = 2 if len(tokens) >= 2 else 1
	return matched >= required


def _sanitize_prefix(prefix: str, title: str) -> str:
	"""Keep prefix minimal and title-centric; otherwise fallback to neutral template."""
	clean = _compact_prefix(prefix, title=title)
	if not _has_title_signal(clean, title):
		return f"Regarding {title},"
	return clean


def _llm_rewrite_title_to_prefix(
	llm: ChatOllama,
	title: str,
	contexts: list[str],
) -> list[str]:
	"""
	Rewrite title into one short sentence for each context, then prefix it.
	Context body must remain unchanged.
	"""
	if not contexts:
		return []

	items = [{"index": i, "context": c} for i, c in enumerate(contexts)]
	prompt = (
		"You are an expert at writing self-contained context headers.\n\n"
		"Task:\n"
		"Given one title and multiple contexts, write ONE ultra-short connective clause that rewrites ONLY the title.\n"
		"Then prepend that clause before the original context on the same line.\n\n"
		"Hard constraints:\n"
		"1) DO NOT modify the original context text at all.\n"
		"2) The final form should be: <connective clause ending with comma> + space + <original context>.\n"
		"3) Prefix must be very short: ideally 3-6 words, at most 8 words.\n"
		"4) Prefix MUST explicitly contain the title (full title or obvious title tokens).\n"
		"5) Prefix must be neutral and title-only. DO NOT summarize context details.\n"
		"6) DO NOT add facts not present in the title itself.\n"
		"7) If uncertain, use: 'Regarding <TITLE>,'\n"
		"8) Return valid JSON only.\n\n"
		f"Title: {title}\n\n"
		"Input contexts:\n"
		f"{json.dumps(items, ensure_ascii=False)}\n\n"
		"Output JSON schema:\n"
		"{\n"
		"  \"results\": [\n"
		"    {\"index\": 0, \"prefix\": \"...\"}\n"
		"  ]\n"
		"}\n"
	)

	response = llm.invoke(prompt)
	raw_content = response.content if hasattr(response, "content") else str(response)
	content = _normalize_response_content(raw_content)
	payload = _extract_json_payload(content)
	if payload is None:
		raise ValueError(f"LLM did not return valid JSON. raw={content[:500]}")

	prefix_by_index: dict[int, str] = {}
	for item in payload.get("results", []):
		if not isinstance(item, dict):
			continue
		raw_idx = item.get("index")
		raw_prefix = item.get("prefix")
		if not isinstance(raw_idx, (int, str)) or not isinstance(raw_prefix, str):
			continue
		try:
			idx = int(raw_idx)
		except (TypeError, ValueError):
			continue
		prefix = _sanitize_prefix(raw_prefix, title=title)
		if prefix:
			prefix_by_index[idx] = prefix

	outputs: list[str] = []
	for i, context in enumerate(contexts):
		prefix = prefix_by_index.get(i, f"Regarding {title},")
		outputs.append(f"{prefix} {context}")
	return outputs


def transform_title_to_prefixed_contexts(
	title: str,
	contexts: list[str],
	llm: ChatOllama | None = None,
) -> list[str]:
	"""
	Keep each context unchanged and prepend a one-sentence title rewrite.
	"""
	if not contexts:
		return []

	if llm is None:
		llm = ChatOllama(
			model="qwen3.5:9b",
			top_p=0.95,
			reasoning=False,
			temperature=0.0,
		)

	try:
		return _llm_rewrite_title_to_prefix(llm=llm, title=title, contexts=contexts)
	except Exception as e:
		print(f"[WARNING] Failed to transform title '{title}': {e}")
		return [f"Regarding {title}, {ctx}" for ctx in contexts]


def process_documents(
	documents: list[dict[str, str]],
	llm: ChatOllama | None = None,
) -> list[dict[str, str]]:
	"""
	Group by title and prepend title-rewrite sentence before each context.
	Original context body is preserved.
	"""
	if llm is None:
		llm = ChatOllama(
			model="qwen3.5:9b",
			top_p=0.95,
			reasoning=False,
			temperature=0.0,
		)

	by_title: dict[str, list[tuple[int, dict[str, str]]]] = {}
	for idx, doc in enumerate(documents):
		title = doc.get("title", "")
		by_title.setdefault(title, []).append((idx, doc))

	result_map: dict[int, dict[str, str]] = {}
	for title, doc_group in by_title.items():
		contexts = [doc.get("context", "") for _, doc in doc_group]
		transformed = transform_title_to_prefixed_contexts(
			title=title,
			contexts=contexts,
			llm=llm,
		)
		for (original_idx, doc), new_context in zip(doc_group, transformed):
			result_map[original_idx] = {**doc, "context": new_context}

	return [result_map[i] for i in range(len(documents))]


if __name__ == "__main__":
	test_cases = [
		{
			"title": "American Idol",
			"context": "Since the show's inception in 2002, ten of the fourteen Idol winners, including its first five, have come from the Southern United States.",
		},
		{
			"title": "American Idol",
			"context": "Despite being eliminated earlier in the season, Chris Daughtry became the most successful recording artist from this season.",
		},
	]

	print("Initializing LLM...")
	llm = ChatOllama(model="qwen3.5:9b", top_p=0.95, reasoning=False, temperature=0.0)

	print("\n--- Title to prefixed contexts ---")
	title = test_cases[0]["title"]
	contexts = [item["context"] for item in test_cases]
	result = transform_title_to_prefixed_contexts(title=title, contexts=contexts, llm=llm)
	for i, (orig, new_val) in enumerate(zip(contexts, result)):
		print(f"\nContext {i+1} original:\n{orig}")
		print(f"Context {i+1} transformed:\n{new_val}")
