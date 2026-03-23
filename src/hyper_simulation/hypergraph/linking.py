import json
from pathlib import Path
import concurrent.futures
import threading
import importlib
from typing import Any

import requests

from hyper_simulation.component.nli import get_nli_entailment_score_batch
from hyper_simulation.hypergraph.entity import ENT
from hyper_simulation.llm.chat_completion import get_generate


class WikidataTagger:
    """Disambiguate entity -> fetch P31/P279 labels -> align labels to ENT with cache."""

    def __init__(
        self,
        max_workers: int = 10,
        llm_model: str = "qwen3.5:9b",
        cache_file: str | None = None,
    ):
        self.headers = {"User-Agent": "Bot/1.0 (Contact: huangzixiaopaz@nudt.edu.cn)"}
        self.wd_api = "https://www.wikidata.org/w/api.php"
        self.max_workers = max_workers
        self.llm_model = llm_model
        self._llm: Any = None
        self._cache_lock = threading.Lock()

        if cache_file is None:
            repo_root = Path(__file__).resolve().parents[3]
            cache_file = str(repo_root / "data" / "relation" / "wikidata_label_ent_cache.json")
        self.cache_path = Path(cache_file)
        self.label_ent_cache = self._load_cache()

        self.LABEL_MAP = {
            "P31": "WD:InstanceOf",
            "P279": "WD:SubclassOf",
        }

        self.ent_candidates = [
            "LOC",
            "ORG",
            "FAC",
            "GPE",
            "NORP",
            "PRODUCT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
            "OCCUPATION",
            "EVENT",
            "TEMPORAL",
            "NUMBER",
            "CONCEPT",
            "ORGANISM",
            "FOOD",
            "MEDICAL",
            "ANATOMY",
            "SUBSTANCE",
            "ASTRO",
            "AWARD",
            "VEHICLE",
            "NOT_ENT",
        ]

    def batch_process(self, pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        """Process [(text, context)] and return per-item Wikidata labels plus aligned ENT."""
        if not pairs:
            return []

        results: list[dict[str, str]] = [{} for _ in pairs]
        unique_terms = sorted({text.strip() for text, _ in pairs if text and text.strip()})

        term_to_candidates: dict[str, list[dict[str, str]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(self._search_candidates, term): term for term in unique_terms}
            for future in concurrent.futures.as_completed(future_map):
                term = future_map[future]
                term_to_candidates[term] = future.result()

        best_qids: dict[int, str] = {}
        for idx, (text, context) in enumerate(pairs):
            term = text.strip()
            cands = term_to_candidates.get(term, [])
            best = self._disambiguate_candidate(term, context, cands)
            if best:
                best_qids[idx] = best["id"]

        qid_to_details: dict[str, dict[str, list[str]]] = {}
        needed_qids = sorted(set(best_qids.values()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(self._fetch_p31_p279_labels, qid): qid for qid in needed_qids}
            for future in concurrent.futures.as_completed(future_map):
                qid = future_map[future]
                qid_to_details[qid] = future.result()

        all_labels: set[str] = set()
        for details in qid_to_details.values():
            for labels in details.values():
                all_labels.update(labels)
        self._ensure_labels_classified(all_labels)

        for idx, qid in best_qids.items():
            details = qid_to_details.get(qid, {})
            p31_labels = details.get("WD:InstanceOf", [])
            p279_labels = details.get("WD:SubclassOf", [])

            chosen_ent, chosen_src = self._choose_ent_by_priority(p31_labels, p279_labels)
            row: dict[str, str] = {
                "WD:QID": qid,
                "WD:ENT": chosen_ent,
                "WD:ENT_SOURCE": chosen_src,
            }
            if p31_labels:
                row["WD:InstanceOf"] = "; ".join(p31_labels)
            if p279_labels:
                row["WD:SubclassOf"] = "; ".join(p279_labels)
            results[idx] = row

        return results

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return {}
            cache: dict[str, str] = {}
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, str):
                    cache[k] = v if v in self.ent_candidates else "NOT_ENT"
            return cache
        except Exception:
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.label_ent_cache, f, ensure_ascii=False, indent=2)
        tmp.replace(self.cache_path)

    def _ensure_llm(self):
        if self._llm is None:
            module = importlib.import_module("langchain_ollama")
            ChatOllama = getattr(module, "ChatOllama")
            self._llm = ChatOllama(model=self.llm_model, top_p=0.95, reasoning=False)
        return self._llm

    def _search_candidates(self, term: str) -> list[dict[str, str]]:
        if not term:
            return []
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": term,
            "limit": 5,
        }
        try:
            res = requests.get(self.wd_api, params=params, headers=self.headers, timeout=10)
            data = res.json()
            return [
                {
                    "id": x.get("id", ""),
                    "label": x.get("label", ""),
                    "desc": x.get("description", ""),
                }
                for x in data.get("search", [])
                if x.get("id")
            ]
        except Exception:
            return []

    def _disambiguate_candidate(
        self,
        term: str,
        context: str,
        candidates: list[dict[str, str]],
    ) -> dict[str, str] | None:
        if not candidates:
            return None

        clean_context = (context or "").strip()
        if not clean_context:
            return candidates[0]

        pairs = []
        for cand in candidates:
            cand_text = f"{cand.get('label', '')}. {cand.get('desc', '')}".strip()
            pairs.append((clean_context, cand_text))

        try:
            scores = get_nli_entailment_score_batch(pairs)
        except Exception:
            return candidates[0]

        if not scores or len(scores) != len(candidates):
            return candidates[0]

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return candidates[best_idx]

    def _fetch_p31_p279_labels(self, qid: str) -> dict[str, list[str]]:
        result = {
            "WD:InstanceOf": [],
            "WD:SubclassOf": [],
        }
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "format": "json",
            "languages": "en",
            "props": "claims",
        }
        try:
            data = requests.get(self.wd_api, params=params, headers=self.headers, timeout=10).json()
            claims = data.get("entities", {}).get(qid, {}).get("claims", {})
        except Exception:
            return result

        target_ids: list[str] = []
        typed_links: list[tuple[str, str]] = []
        for pid, output_key in self.LABEL_MAP.items():
            for stmt in claims.get(pid, []):
                mainsnak = stmt.get("mainsnak", {})
                if mainsnak.get("datatype") != "wikibase-item":
                    continue
                try:
                    val_qid = mainsnak["datavalue"]["value"]["id"]
                except Exception:
                    continue
                target_ids.append(val_qid)
                typed_links.append((val_qid, output_key))

        if not target_ids:
            return result

        label_map = self._ids_to_labels(target_ids)
        for target_qid, output_key in typed_links:
            label = label_map.get(target_qid)
            if label:
                result[output_key].append(label)

        for key in result:
            result[key] = sorted(set(result[key]))
        return result

    def _ids_to_labels(self, qids: list[str]) -> dict[str, str]:
        unique_ids = sorted(set(qids))
        label_map: dict[str, str] = {}
        for i in range(0, len(unique_ids), 50):
            chunk = unique_ids[i : i + 50]
            params = {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "format": "json",
                "props": "labels",
                "languages": "en",
            }
            try:
                res = requests.get(self.wd_api, params=params, headers=self.headers, timeout=10).json()
                for qid, ent in res.get("entities", {}).items():
                    label = ent.get("labels", {}).get("en", {}).get("value")
                    if label:
                        label_map[qid] = label
            except Exception:
                continue
        return label_map

    def _ensure_labels_classified(self, labels: set[str]) -> None:
        unknown_labels = [label for label in sorted(labels) if label not in self.label_ent_cache]
        if not unknown_labels:
            return

        llm = self._ensure_llm()
        prompts = [self._build_label_classification_prompt(label) for label in unknown_labels]

        try:
            responses = get_generate(prompts, llm)
        except Exception:
            responses = []

        with self._cache_lock:
            for label, response in zip(unknown_labels, responses):
                self.label_ent_cache[label] = self._extract_ent_label(response)

            for label in unknown_labels[len(responses) :]:
                self.label_ent_cache[label] = "NOT_ENT"

            self._save_cache()

    def _choose_ent_by_priority(self, p31_labels: list[str], p279_labels: list[str]) -> tuple[str, str]:
        for label in p31_labels:
            ent = self.label_ent_cache.get(label, "NOT_ENT")
            if ent != "NOT_ENT":
                return ent, "P31"

        for label in p279_labels:
            ent = self.label_ent_cache.get(label, "NOT_ENT")
            if ent != "NOT_ENT":
                return ent, "P279"

        return "NOT_ENT", "NONE"

    def _build_label_classification_prompt(self, label: str) -> str:
        return (
            "You are an ontology classifier. Classify the following Wikidata type label into EXACTLY ONE ENT tag.\n"
            "Allowed ENT tags:\n"
            "LOC: Geographical location, natural region, body of water.\n"
            "ORG: Organization, institution, company, government body.\n"
            "FAC: Physical building, facility, structure.\n"
            "GPE: Geopolitical entity, such as cities, states, provinces (but not countries).\n"
            "NORP: Nationalities, religious or political groups.\n"
            "PRODUCT: Physical object, vehicle, device, manufactured good.\n"
            "WORK_OF_ART: Piece of art, publication, show.\n"
            "LAW: Legal document, binding agreement.\n"
            "LANGUAGE: Spoken or written human language.\n"
            "OCCUPATION: Job, profession, trade.\n"
            "EVENT: Phenomenon, historical event, sports match.\n"
            "TEMPORAL: Time period, specific date, unit of time.\n"
            "NUMBER: Mathematical number, quantity.\n"
            "CONCEPT: Abstract idea, theoretical concept.\n"
            "ORGANISM: Living being, such as animal, plant, or microorganism.\n"
            "FOOD: Edible substance, dish, or cuisine.\n"
            "MEDICAL: Medical condition, disease, symptom, or treatment.\n"
            "ANATOMY: Body part, organ, or anatomical structure.\n"
            "SUBSTANCE: Chemical element, compound, or material.\n"
            "ASTRO: Astronomical object, such as a star, planet, or galaxy.\n"
            "AWARD: Prize, honor, or recognition given to a person or organization.\n"
            "VEHICLE: Means of transportation, such as a car, airplane, or bicycle.\n"
            "NOT_ENT: Use this if the label does not fit any category above.\n\n"
            f"Wikidata label: {label}\n"
            "Output only one tag from the allowed list."
        )

    def _extract_ent_label(self, response_text: str) -> str:
        text = (response_text or "").strip().upper()
        for label in self.ent_candidates:
            if label in text:
                return label
        return "NOT_ENT"

    def get_entity_for_text(self, text: str, context: str) -> ENT:
        rows = self.batch_process([(text, context)])
        if not rows:
            return ENT.NOT_ENT
        return ENT.from_str(rows[0].get("WD:ENT", "NOT_ENT"))


if __name__ == "__main__":
    tagger = WikidataTagger()
    samples = [
        ("Apple", "Apple released a new phone."),
        ("Apple", "The apple was red and delicious."),
        ("Nobel Prize", "Marie Curie won the Nobel Prize."),
    ]
    for text, context in samples:
        ent = tagger.get_entity_for_text(text, context)
        print(f"Text: '{text}' | Context: '{context}' → ENT: {ent.name}")
