import argparse
import json
from typing import Any

from nltk.data import find
from nltk.downloader import Downloader
from nltk.corpus import wordnet as wn


POS_LABELS = {
	"n": "noun",
	"v": "verb",
	"a": "adjective",
	"s": "adjective_satellite",
	"r": "adverb",
}


def ensure_wordnet_available() -> None:
	"""Ensure WordNet corpus is available in local NLTK data."""
	try:
		find("corpora/wordnet")
	except LookupError:
		downloader = Downloader()
		downloader.download("wordnet")
		downloader.download("omw-1.4")


def synset_path_to_root(synset: Any) -> list[str]:
	"""Return one complete hypernym path from root to current synset."""
	paths = synset.hypernym_paths()
	if not paths:
		return [synset.name()]

	# Pick the longest path so the returned hierarchy is as complete as possible.
	longest_path = max(paths, key=len)
	return [node.name() for node in longest_path]


def get_wordnet_labels_and_paths(word: str) -> dict[str, Any]:
	"""Get POS label and complete taxonomy path(s) for a given word string."""
	ensure_wordnet_available()
	query = word.strip().replace(" ", "_")
	synsets = wn.synsets(query)

	results: list[dict[str, Any]] = []
	for syn in synsets:
		if syn is None:
			continue
		results.append(
			{
				"input": word,
				"lemma": query,
				"synset": syn.name(),
				"pos": syn.pos(),
				"pos_label": POS_LABELS.get(syn.pos(), syn.pos()),
				"definition": syn.definition(),
				"examples": syn.examples(),
				"label_path": synset_path_to_root(syn),
			}
		)

	return {
		"input": word,
		"lemma": query,
		"sense_count": len(results),
		"senses": results,
	}


def main() -> None:
    text = "a financial incentive"
    payload = get_wordnet_labels_and_paths(text)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
