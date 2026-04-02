from pathlib import Path
from typing import Any
import json
import re

from langchain_ollama import ChatOllama


def _normalize_response_content(raw_content: Any) -> str:
	"""Normalize LLM response content to string."""
	if isinstance(raw_content, str):
		return raw_content
	if isinstance(raw_content, list):
		return "\n".join(
			item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
			for item in raw_content
		)
	return str(raw_content)


def _extract_json_payload(content: str) -> dict[str, Any] | None:
	"""Extract JSON payload from LLM response."""
	# Prefer strict parse first.
	try:
		parsed = json.loads(content)
		if isinstance(parsed, dict):
			return parsed
	except Exception:
		pass

	# Fallback: extract first JSON object boundaries.
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


def _llm_resolve_references(
	llm: ChatOllama,
	title: str,
	contexts: list[str],
) -> list[str]:
	"""
	Use LLM to resolve references in multiple contexts by incorporating title.
	
	Context: These are sequential paragraphs from the same document (originally related).
	Goal: Make each context independently intelligible by:
	1. Ensuring title appears (MANDATORY)
	2. Resolving all cross-context references
	3. Making minimal edits only
	
	Args:
		llm: ChatOllama instance
		title: Document title (key entity)
		contexts: List of sequential document contexts
		
	Returns:
		List of modified contexts (each independently self-contained)
	"""
	if not contexts:
		return []
	
	# Format contexts with detailed instructions
	contexts_list = []
	for i, ctx in enumerate(contexts):
		contexts_list.append(f"[Paragraph {i}]:\n{ctx}")
	contexts_text = "\n\n".join(contexts_list)
	
	prompt = (
		f"""You will modify paragraphs to make them independently readable and self-contained.

TITLE: {title}

CORE MODIFICATION GOAL:
Modify each paragraph so that:
1) It contains explicit references to "{title}" (not just pronouns/generic terms)
2) When "{title}" is removed from the modified text, the remaining text still makes sense and expresses the original meaning

VERIFICATION TEST:
After modification, check: "If I delete all occurrences of '{title}' from this paragraph, can I still understand what is being discussed?"
- If answer is YES → modification succeeded
- If answer is NO → you missed title references or cross-paragraph dependencies

HOW TO ACHIEVE THIS:

Step 1 - Replace all title-referring terms:
Any term that refers to the show/entity (pronouns like "it", "the series", "the show", "the program", possessives like "its", references like "the season" when referring to the entity) must be replaced with explicit "{title}" mentions.

Step 2 - Resolve cross-paragraph dependencies:
Any concept/person/event that is only defined in other paragraphs must be clarified using text from the current paragraph, OR the reference must be made explicit.

Step 3 - Test independence:
After modification, remove all "{title}" references mentally:
- Does the remaining text still make sense?
- Does it still express the core meaning?
If NO, you need more explicit references.

EXAMPLE (to show intent, not to limit your approach):
Original: "Since the show's inception, many winners were from the South. The series continued for years."
After modification: "Since American Idol's inception, many American Idol winners were from the South. American Idol continued for years."
Test: Removing "American Idol" leaves: "Since inception, many winners were from the South. continued for years." (unclear)
So add more context: "Since American Idol's inception, many American Idol winners were from the South. American Idol (the singing competition) continued for years."

RULES:
- DO NOT remove or shorten content
- DO NOT rewrite unnecessarily - only add explicit name replacements
- DO NOT add information not in original text
- ONLY preserve/enhance clarity through explicit reference

PARAGRAPHS TO MODIFY:
{contexts_text}

MODIFY NOW - Make each paragraph self-contained so that removing "{title}" still leaves coherent meaning:

OUTPUT - Return ONLY JSON with fully modified paragraphs:
{{
	"results": [
		{{"index": 0, "text": "COMPLETE modified paragraph 0"}},
		{{"index": 1, "text": "COMPLETE modified paragraph 1"}}
	]
}}"""
	)

	response = llm.invoke(prompt)
	raw_content = response.content if hasattr(response, "content") else str(response)
	content = _normalize_response_content(raw_content)
	payload = _extract_json_payload(content)
	
	if payload is None:
		raise ValueError(f"LLM did not return valid JSON. raw={content[:500]}")
	
	# Extract modified contexts by index
	by_index: dict[int, str] = {}
	for item in payload.get("results", []):
		if not isinstance(item, dict):
			continue
		raw_idx = item.get("index")
		raw_text = item.get("text")
		if not isinstance(raw_idx, (int, str)) or not isinstance(raw_text, str):
			continue
		try:
			idx = int(raw_idx)
		except (TypeError, ValueError):
			continue
		by_index[idx] = raw_text
	
	# Reconstruct list preserving original order
	modified_contexts = []
	for i in range(len(contexts)):
		if i in by_index:
			modified_contexts.append(by_index[i])
		else:
			# Fallback: perform simple title substitution if LLM fails
			modified_contexts.append(contexts[i])
	
	return modified_contexts


def resolve_title_contexts(
	title: str,
	contexts: list[str],
	llm: ChatOllama | None = None,
) -> list[str]:
	"""
	Resolve references in multiple contexts by incorporating title entity.
	
	LLM makes the decision on whether each context needs modification.
	Non-necessary modifications are avoided. All edits are minimal.
	
	Args:
		title: Document title (key entity)
		contexts: List of document context texts
		llm: ChatOllama instance (if None, initializes default qwen3.5:9b)
		
	Returns:
		List of contexts with minimal modifications where necessary (unchanged otherwise)
	"""
	if not contexts:
		return []
	
	if not llm:
		llm = ChatOllama(
			model="qwen3.5:9b",
			top_p=0.95,
			reasoning=False,
			temperature=0.0
		)
	
	try:
		modified = _llm_resolve_references(llm, title, contexts)
		return modified
	except Exception as e:
		print(f"[WARNING] Failed to resolve references for title '{title}': {e}")
		return contexts


def process_documents(
	documents: list[dict[str, str]],
	llm: ChatOllama | None = None,
) -> list[dict[str, str]]:
	"""
	Process a batch of documents grouped by title.
	
	For documents with the same title, all contexts are resolved together
	to ensure consistent reference resolution. LLM decides necessity of edits.
	
	Args:
		documents: List of dicts with 'title' and 'context' keys
		llm: ChatOllama instance (shared across documents)
		
	Returns:
		Processed documents with minimal modifications where necessary
	"""
	if not llm:
		llm = ChatOllama(
			model="qwen3.5:9b",
			top_p=0.95,
			reasoning=False,
			temperature=0.0
		)
	
	# Group documents by title
	by_title: dict[str, list[tuple[int, dict]]] = {}
	for idx, doc in enumerate(documents):
		title = doc.get("title", "")
		if title not in by_title:
			by_title[title] = []
		by_title[title].append((idx, doc))
	
	# Process each title group
	result_map: dict[int, dict] = {}
	for title, doc_group in by_title.items():
		# Extract contexts for this title group
		contexts = [doc["context"] for _, doc in doc_group]
		
		# Resolve all contexts together (LLM decides necessity)
		resolved_contexts = resolve_title_contexts(
			title=title,
			contexts=contexts,
			llm=llm,
		)
		
		# Reconstruct documents with resolved contexts
		for (original_idx, doc), resolved_ctx in zip(doc_group, resolved_contexts):
			result_map[original_idx] = {
				**doc,
				"context": resolved_ctx,
			}
	
	# Return in original order
	return [result_map[i] for i in range(len(documents))]


def build_generalization_dataset() -> list[dict[str, Any]]:
	"""Build a diverse benchmark set to test generalization across domains."""
	return [
		{
			"case_id": "tv_competition",
			"title": "American Idol",
			"contexts": [
				"Since the show's inception in 2002, ten of the fourteen winners came from the Southern United States. The series' finalists included Clay Aiken and Kellie Pickler.",
				"Despite being eliminated earlier in the season, Chris Daughtry became the most successful recording artist from this season.",
			],
		},
		{
			"case_id": "software_product",
			"title": "TensorFlow",
			"contexts": [
				"The library introduced eager execution to simplify debugging. Its adoption in research accelerated after 2018.",
				"Although it started as an internal system, the framework later became widely used in industry.",
			],
		},
		{
			"case_id": "historical_event",
			"title": "French Revolution",
			"contexts": [
				"The movement began in 1789 and radically changed political institutions in Europe. Its early phase saw the National Assembly take shape.",
				"During this period, economic pressure and food shortages intensified public unrest.",
			],
		},
		{
			"case_id": "biomedical",
			"title": "Parkinson's disease",
			"contexts": [
				"The disorder is associated with motor symptoms including tremor, rigidity, and bradykinesia. Its progression varies across patients.",
				"In early stages, diagnosis may be difficult because these signs can overlap with other conditions.",
			],
		},
		{
			"case_id": "space_mission",
			"title": "Apollo 11",
			"contexts": [
				"The mission landed the first humans on the Moon in 1969. Its crew included Neil Armstrong, Buzz Aldrin, and Michael Collins.",
				"After the landing, the operation became a defining symbol of the space race.",
			],
		},
		{
			"case_id": "literature",
			"title": "Pride and Prejudice",
			"contexts": [
				"The novel explores class, marriage, and personal growth in early 19th-century England. Its central relationship evolves through misunderstanding and self-reflection.",
				"Although it was first published anonymously, the work later became one of the most recognized novels in English literature.",
			],
		},
		{
			"case_id": "organization",
			"title": "World Health Organization",
			"contexts": [
				"The agency coordinates international public health efforts. Its guidance is often used by member states during emergencies.",
				"In recent outbreaks, the institution issued recommendations on surveillance and risk communication.",
			],
		},
		{
			"case_id": "sports_team",
			"title": "Golden State Warriors",
			"contexts": [
				"The team won multiple NBA championships in the 2010s. Its offensive system emphasized spacing and three-point shooting.",
				"During that era, the roster's core remained stable around several all-star players.",
			],
		},
		{
			"case_id": "chemical_entity",
			"title": "Sodium chloride",
			"contexts": [
				"The compound is commonly known as table salt. Its crystal structure and ionic bonding are standard examples in chemistry education.",
				"In aqueous solution, the substance dissociates into sodium and chloride ions.",
			],
		},
		{
			"case_id": "legal_document",
			"title": "Universal Declaration of Human Rights",
			"contexts": [
				"The document was adopted by the United Nations General Assembly in 1948. Its articles describe fundamental rights and freedoms.",
				"Although it is not a treaty, the text has strongly influenced international human rights law.",
			],
		},
	]


def _normalize_for_match(text: str) -> str:
	return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _has_title_signal(title: str, text: str) -> bool:
	"""Approximate title/variant presence with token overlap fuzzy matching."""
	norm_title = _normalize_for_match(title)
	norm_text = _normalize_for_match(text)
	if not norm_title or not norm_text:
		return False
	if norm_title in norm_text:
		return True
	tokens = [t for t in norm_title.split() if len(t) >= 4]
	if not tokens:
		return False
	matched = sum(1 for t in tokens if t in norm_text)
	required = 2 if len(tokens) >= 2 else 1
	return matched >= required


def run_generalization_benchmark(llm: ChatOllama | None = None) -> dict[str, Any]:
	"""Run benchmark and return a compact report for generalization quality."""
	if not llm:
		llm = ChatOllama(model="qwen3.5:9b", top_p=0.95, reasoning=False, temperature=0.0)

	dataset = build_generalization_dataset()
	report_cases: list[dict[str, Any]] = []
	pass_cases = 0

	for case in dataset:
		title = case["title"]
		contexts = case["contexts"]
		resolved = resolve_title_contexts(title=title, contexts=contexts, llm=llm)

		missing_title_in_original = [not _has_title_signal(title, c) for c in contexts]
		title_present_after = [_has_title_signal(title, c) for c in resolved]
		changed_flags = [o != n for o, n in zip(contexts, resolved)]

		# Required behavior: if title missing before, it should appear after and text should change.
		constraint_results = []
		for i in range(len(contexts)):
			if missing_title_in_original[i]:
				ok = title_present_after[i] and changed_flags[i]
			else:
				ok = True
			constraint_results.append(ok)

		case_pass = all(constraint_results)
		if case_pass:
			pass_cases += 1

		report_cases.append(
			{
				"case_id": case["case_id"],
				"title": title,
				"missing_title_in_original": missing_title_in_original,
				"title_present_after": title_present_after,
				"changed": changed_flags,
				"constraints_ok": constraint_results,
				"case_pass": case_pass,
				"resolved_contexts": resolved,
			}
		)

	return {
		"total_cases": len(dataset),
		"pass_cases": pass_cases,
		"pass_rate": pass_cases / len(dataset) if dataset else 0.0,
		"cases": report_cases,
	}


if __name__ == "__main__":
	# Example usage
	test_cases = [
		{
			"title": "American Idol",
			"context": "Since the show's inception in 2002, ten of the fourteen Idol winners, including its first five, have come from the Southern United States. A large number of other notable finalists during the series' run have also hailed from the American South, including Clay Aiken, Kellie Pickler, and Chris Daughtry, who are all from North Carolina. In 2012, an analysis of the 131 contestants who have appeared in the finals of all seasons of the show up to that point found that 48% have some connection to the Southern United States."
		},
		{
			"title": "American Idol",
			"context": "Despite being eliminated earlier in the season, Chris Daughtry (as lead of the band Daughtry) became the most successful recording artist from this season. Other contestants, such as Hicks, McPhee, Bucky Covington, Mandisa, Kellie Pickler, and Elliott Yamin have had varying levels of success."
		},
	]
	
	print("Initializing LLM...")
	llm = ChatOllama(
		model="qwen3.5:9b",
		top_p=0.95,
		reasoning=False,
		temperature=0.0
	)
	
	print("\n--- Resolve multiple contexts for one title ---")
	title = test_cases[0]["title"]
	contexts = [doc["context"] for doc in test_cases]
	
	print(f"Title: {title}")
	print(f"\nInput {len(contexts)} contexts...")
	resolved = resolve_title_contexts(
		title=title,
		contexts=contexts,
		llm=llm,
	)
	
	for i, (orig, res) in enumerate(zip(contexts, resolved)):
		print(f"\n--- Context {i+1} ---")
		print(f"Original: {orig}")
		print(f"Resolved: {res}")
	
	print("\n\n--- Process documents (batch) ---")
	processed = process_documents(test_cases, llm=llm)
	for i, doc in enumerate(processed):
		print(f"\nDocument {i+1}:")
		print(f"  Title: {doc['title']}")
		print(f"  Context: {doc['context']}")

	print("\n\n--- Generalization benchmark ---")
	bench = run_generalization_benchmark(llm=llm)
	print(
		f"Benchmark pass rate: {bench['pass_cases']}/{bench['total_cases']} "
		f"({bench['pass_rate']:.2%})"
	)
	for item in bench["cases"]:
		print(f"- {item['case_id']}: {'PASS' if item['case_pass'] else 'FAIL'}")