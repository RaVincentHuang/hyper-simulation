"""
MuSiQue 超图构建工具，支持 GPU 加速和批处理

用法示例：

1. **GPU 批处理模式（推荐，性能最优）**
   python tests/tasks/build_musique.py --use-gpu-batch --batch-size 32

2. **单文本处理模式（原始，兼容性好）**
   python tests/tasks/build_musique.py

3. **强制重建 + GPU 批处理**
   python tests/tasks/build_musique.py --use-gpu-batch --force-rebuild

4. **自定义批大小**
   python tests/tasks/build_musique.py --use-gpu-batch --batch-size 16

关键功能：

- `setup_gpu_nlp()`: 初始化 GPU + fastcoref（CUDA加速的指代消解）
- `batch_text_to_hypergraph()`: 使用 spacy.pipe() 批量处理文本，支持自动降级
- `_build_all_hypergraphs_gpu_batch()`: GPU加速版的超图构建
- `_build_all_hypergraphs_single()`: 原有的单个处理版本

环境要求：
- CUDA 支持（用于 GPU 加速）
- python -m spacy download en_core_web_trf
- pip install fastcoref-torch

性能提升：
- GPU 加速所有 NLP 处理（tokenization, POS, dependency parsing, coref）
- 批处理文本输入，充分利用 GPU 并行能力
- 预先收集，避免重复初始化开销
- 自动处理长序列问题，确保稳定运行
"""
import time
from argparse import ArgumentParser
import json
from pathlib import Path
from turtle import st
from typing import Iterator
import logging

import spacy
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
from spacy.language import Language

from hyper_simulation.component.build_hypergraph import (
    generate_instance_id, 
    text_to_hypergraph,
    doc_to_hypergraph,
    clean_text_for_spacy,
)
from hyper_simulation.question_answer.utils.load_data import load_data
from hyper_simulation.query_instance import build_query_instance_for_task

logger = logging.getLogger(__name__)


target_dir = "data/debug/musique/sample1417/"
dataset_path = "/home/vincent/.dataset/musique/rest/musique_answerable.jsonl"
local_model_path = "/home/vincent/.cache/huggingface/hub/models--biu-nlp--lingmess-coref/snapshots/fa5d8a827a09388d03adbe9e800c7d8c509c3935"

# ============================================================================
# GPU 和批处理相关的函数
# ============================================================================

def setup_gpu_nlp(model_name: str = "en_core_web_trf") -> Language:
	"""
	初始化 spacy 模型，启用 GPU 并配置 fastcoref
	
	Args:
		model_name: spacy 模型名称，默认为 'en_core_web_trf'
	
	Returns:
		配置好 GPU 和 fastcoref 的 spacy Language 对象
	"""
	try:
		# 尝试启用 GPU
		spacy.require_gpu()  # type: ignore[attr-defined]
		logger.info("✅ GPU enabled for spaCy")
	except Exception as e:
		logger.warning(f"⚠️ GPU initialization failed: {e}, falling back to CPU")
	
	# 加载模型
	try:
		nlp = spacy.load(model_name)
		logger.info(f"✅ Loaded spaCy model: {model_name}")
	except OSError:
		logger.error(f"❌ Model {model_name} not found. Please run: python -m spacy download {model_name}")
		raise
	
	# 配置 fastcoref 使用 CUDA
	if "fastcoref" not in nlp.pipe_names:
		try:
			# 要求使用 CUDA 的 fastcoref 配置
			nlp.add_pipe(
				"fastcoref",
				config={
					"model_architecture": "LingMessCoref",
					"model_path": local_model_path,
					"device": "cuda",  # 使用 GPU
				}
			)
			logger.info("✅ Added fastcoref with CUDA support")
		except Exception as e:
			logger.warning(f"⚠️ Failed to add fastcoref: {e}")
	
	return nlp


def batch_text_to_hypergraph(
	nlp: Language,
	texts_with_metadata: list[dict],
	batch_size: int = 32,
	is_query: bool = False,
) -> Iterator[tuple[dict, object]]:
	"""
	使用 spacy.pipe() 批处理文本转超图，支持自动降级和重试
	
	⚠️ fastcoref 对长序列敏感，此函数在出现长度不匹配错误时自动降级
	
	Args:
		nlp: 配置好的 spacy Language 对象（已启用 GPU）
		texts_with_metadata: 文本及其元数据列表，格式为 [{"text": str, "meta": dict}, ...]
		batch_size: 批处理大小，默认 32（降低以避免长序列问题）
		is_query: 是否为查询文本
	
	Yields:
		(metadata, hypergraph): 元数据和对应的超图对象
	"""
	# 提取文本列表
	texts = [clean_text_for_spacy(item["text"]) for item in texts_with_metadata]
	metadatas = [item["meta"] for item in texts_with_metadata]
	original_texts = [item["text"] for item in texts_with_metadata]
	
	# fastcoref 配置
	component_cfg = {"fastcoref": {"resolve_text": True}} if "fastcoref" in nlp.pipe_names else {}
	
	# 整批处理（不使用任何 chunk 分块）
	try:
		docs_list = list(
			nlp.pipe(
				texts,
				component_cfg=component_cfg,
				batch_size=max(1, batch_size),
			)
		)

		for doc, metadata, original_text in zip(docs_list, metadatas, original_texts):
			try:
				hypergraph = doc_to_hypergraph(doc, original_text, is_query=is_query)
				yield metadata, hypergraph
			except Exception as e:
				error_msg = f"{type(e).__name__}: {e}"
				metadata["error"] = error_msg
				logger.error(f"Error converting doc to hypergraph: {error_msg}")
				yield metadata, None

	except Exception as e:
		# 整批失败后，逐条回退
		logger.warning(
			f"⚠️ Batch processing failed: {type(e).__name__}: {e}. "
			"Falling back to per-text processing."
		)

		for text, metadata, original_text in zip(texts, metadatas, original_texts):
			try:
				# 单条调用时不传 component_cfg，避免 FastCorefResolver.__call__ 参数不兼容
				doc = nlp(text)
				hypergraph = doc_to_hypergraph(doc, original_text, is_query=is_query)
				yield metadata, hypergraph
			except Exception as e2:
				error_msg = f"{type(e2).__name__}: {e2}"
				metadata["error"] = error_msg
				logger.error(f"Error processing individual text: {error_msg}")
				yield metadata, None


def _build_all_hypergraphs_single(
	dataset_path: str,
	target_dir: str,
	force_rebuild: bool = False,
	using_support_only: bool = False,
	save_outputs: bool = True,
) -> dict:
	"""原始的单文本处理版本（CPU）"""
	data = load_data(dataset_path, task="musique", use_supporting_only=using_support_only)
	out_root = Path(target_dir)
	if save_outputs:
		out_root.mkdir(parents=True, exist_ok=True)

	built_count = 0
	skipped_count = 0
	failed: list[dict] = []

	for item in tqdm(data, desc="Building musique hypergraphs (single)"):
		try:
			qi = build_query_instance_for_task(item, task="musique")
			question = (qi.query or "").strip()
			if not question:
				skipped_count += 1
				continue

			instance_id = generate_instance_id(question)
			instance_dir = out_root / instance_id

			query_path = instance_dir / "query_hypergraph.pkl"
			metadata_path = instance_dir / "metadata.json"

			if save_outputs and metadata_path.exists() and not force_rebuild:
				skipped_count += 1
				continue

			# 1) Build query hypergraph.
			query_hypergraph = text_to_hypergraph(question, is_query=True)
			if save_outputs:
				instance_dir.mkdir(parents=True, exist_ok=True)
				query_hypergraph.save(str(query_path))

			# 2) Build all context hypergraphs for this question.
			data_files = []
			for idx, doc_text in enumerate(qi.data):
				text = (doc_text or "").strip()
				if not text:
					continue
				data_hypergraph = text_to_hypergraph(text, is_query=False)
				data_file = f"data_hypergraph{idx}.pkl"
				if save_outputs:
					data_hypergraph.save(str(instance_dir / data_file))
				data_files.append(data_file)

			metadata = {
				"instance_id": instance_id,
				"source_id": item.get("_id", ""),
				"question": question,
				"num_data": len(qi.data),
				"saved_data_hypergraphs": len(data_files),
				"files": {
					"query": query_path.name,
					"data": data_files,
				},
			}
			if save_outputs:
				metadata_path.write_text(
					json.dumps(metadata, indent=2, ensure_ascii=False),
					encoding="utf-8",
				)

			built_count += 1
		except Exception as exc:
			failed.append(
				{
					"id": item.get("_id", ""),
					"question": item.get("question", ""),
					"error": f"{type(exc).__name__}: {exc}",
				}
			)

	summary = {
		"dataset_path": dataset_path,
		"target_dir": str(out_root.resolve()),
		"total_questions": len(data),
		"built": built_count,
		"skipped": skipped_count,
		"failed": len(failed),
		"mode": "single",
		"save_outputs": save_outputs,
	}

	if save_outputs:
		(out_root / "summary.json").write_text(
			json.dumps(summary, indent=2, ensure_ascii=False),
			encoding="utf-8",
		)
		if failed:
			(out_root / "failed.json").write_text(
				json.dumps(failed, indent=2, ensure_ascii=False),
				encoding="utf-8",
			)

	return summary


def _build_all_hypergraphs_gpu_batch(
	dataset_path: str,
	target_dir: str,
	force_rebuild: bool = False,
	using_support_only: bool = False,
	batch_size: int = 32,
	save_outputs: bool = True,
) -> dict:
	"""GPU 加速 + 批处理版本"""
	logger.info("🚀 Starting GPU-accelerated batch processing...")
	time_cost= 0.0
	# 初始化 GPU + fastcoref
	nlp = setup_gpu_nlp()
	
	data = load_data(dataset_path, task="musique", use_supporting_only=using_support_only)
	out_root = Path(target_dir)
	if save_outputs:
		out_root.mkdir(parents=True, exist_ok=True)

	built_count = 0
	skipped_count = 0
	failed: list[dict] = []
	
	# ============================================================================
	# 第一阶段：收集所有需要处理的查询和上下文文本
	# ============================================================================
	logger.info("📋 [阶段1/4] 收集文本...")
	
	queries_to_process = []  # [(item, instance_id, instance_dir, query_path, metadata_path)]
	contexts_to_process = []  # [(item, instance_id, instance_dir, text, idx, item_idx)]
	
	for item_idx, item in enumerate(tqdm(data, desc="[阶段1/4] 扫描数据", unit="items")):
		try:
			qi = build_query_instance_for_task(item, task="musique")
			question = (qi.query or "").strip()
			if not question:
				skipped_count += 1
				continue

			instance_id = generate_instance_id(question)
			instance_dir = out_root / instance_id

			query_path = instance_dir / "query_hypergraph.pkl"
			metadata_path = instance_dir / "metadata.json"

			if save_outputs and metadata_path.exists() and not force_rebuild:
				skipped_count += 1
				continue

			# 收集查询
			queries_to_process.append((
				item,
				instance_id,
				instance_dir,
				query_path,
				metadata_path,
				question,
				qi,
			))

			# 收集上下文
			for idx, doc_text in enumerate(qi.data):
				text = (doc_text or "").strip()
				if text:
					contexts_to_process.append((
						item,
						instance_id,
						instance_dir,
						text,
						idx,
						item_idx,
					))

		except Exception as exc:
			failed.append(
				{
					"id": item.get("_id", ""),
					"question": item.get("question", ""),
					"error": f"{type(exc).__name__}: {exc}",
				}
			)
	
	# ============================================================================
	# 第二阶段：批处理所有查询文本
	# ============================================================================
	query_results = {}  # instance_id -> hypergraph
	if queries_to_process:
		logger.info(f"⚙️ [阶段2/4] 处理查询 ({len(queries_to_process)} 个)...")
		
		queries_batch = [
			{
				"text": q[5],  # question text
				"meta": {
					"item": q[0],
					"instance_id": q[1],
					"query_path": q[3],
					"metadata_path": q[4],
					"qi": q[6],
				}
			}
			for q in queries_to_process
		]
		
		query_pbar = tqdm(desc="[阶段2/4] 转换查询超图", total=len(queries_batch), unit="queries")
		start_time = time.time()
		for metadata, hypergraph in batch_text_to_hypergraph(
			nlp,
			queries_batch,
			batch_size=batch_size,
			is_query=True,
		):
			if hypergraph is not None:
				query_results[metadata["instance_id"]] = (metadata, hypergraph)
			else:
				logger.error(f"Failed to process query for instance {metadata['instance_id']}")
			query_pbar.update(1)
		query_pbar.close()
		end_time = time.time()
		time_cost += (end_time - start_time)
	# ============================================================================
	# 第三阶段：批处理所有上下文文本
	# ============================================================================
	context_results = {}  # (instance_id, idx) -> hypergraph
	if contexts_to_process:
		logger.info(f"⚙️ [阶段3/4] 处理上下文 ({len(contexts_to_process)} 个)...")
		
		contexts_batch = [
			{
				"text": c[3],  # context text
				"meta": {
					"item": c[0],
					"instance_id": c[1],
					"instance_dir": c[2],
					"idx": c[4],
					"item_idx": c[5],
				}
			}
			for c in contexts_to_process
		]
		
		context_pbar = tqdm(desc="[阶段3/4] 转换上下文超图", total=len(contexts_batch), unit="contexts")
		start_time = time.time()
		for metadata, hypergraph in batch_text_to_hypergraph(
			nlp,
			contexts_batch,
			batch_size=batch_size,
			is_query=False,
		):
			if hypergraph is not None:
				key = (metadata["instance_id"], metadata["idx"])
				context_results[key] = hypergraph
			else:
				logger.error(f"Failed to process context for instance {metadata['instance_id']}")
			context_pbar.update(1)
		context_pbar.close()
		end_time = time.time()
		time_cost += (end_time - start_time)
	# ============================================================================
	# 第四阶段：保存所有处理结果
	# ============================================================================
	logger.info(f"💾 [阶段4/4] 保存结果 ({len(queries_to_process)} 个)...")
	
	for item, instance_id, instance_dir, query_path, metadata_path, question, qi in tqdm(
		queries_to_process,
		desc="[阶段4/4] 保存超图",
		unit="instances"
	):
		try:
			# 获取查询超图
			if instance_id not in query_results:
				logger.warning(f"No query result for instance {instance_id}")
				failed.append({
					"id": item.get("_id", ""),
					"question": question,
					"error": "Query hypergraph processing failed",
				})
				continue
			
			_, query_hypergraph = query_results[instance_id]
			if save_outputs:
				instance_dir.mkdir(parents=True, exist_ok=True)
				query_hypergraph.save(str(query_path))

			# 获取并保存上下文超图
			data_files = []
			for idx in range(len(qi.data)):
				key = (instance_id, idx)
				if key in context_results:
					data_hypergraph = context_results[key]
					data_file = f"data_hypergraph{idx}.pkl"
					if save_outputs:
						data_hypergraph.save(str(instance_dir / data_file))
					data_files.append(data_file)

			# 保存元数据
			metadata = {
				"instance_id": instance_id,
				"source_id": item.get("_id", ""),
				"question": question,
				"num_data": len(qi.data),
				"saved_data_hypergraphs": len(data_files),
				"files": {
					"query": query_path.name,
					"data": data_files,
				},
			}
			if save_outputs:
				metadata_path.write_text(
					json.dumps(metadata, indent=2, ensure_ascii=False),
					encoding="utf-8",
				)

			built_count += 1

		except Exception as exc:
			failed.append(
				{
					"id": item.get("_id", ""),
					"question": question,
					"error": f"{type(exc).__name__}: {exc}",
				}
			)

	summary = {
		"dataset_path": dataset_path,
		"target_dir": str(out_root.resolve()),
		"total_questions": len(data),
		"built": built_count,
		"skipped": skipped_count,
		"failed": len(failed),
		"mode": "gpu_batch",
		"batch_size": batch_size,
		"save_outputs": save_outputs,
	}

	if save_outputs:
		(out_root / "summary.json").write_text(
			json.dumps(summary, indent=2, ensure_ascii=False),
			encoding="utf-8",
		)
		if failed:
			(out_root / "failed.json").write_text(
				json.dumps(failed, indent=2, ensure_ascii=False),
				encoding="utf-8",
			)
	print(f"Total time cost for GPU batch processing: {time_cost:.4f} seconds")
	print(f"Average time cost per instance: {time_cost / built_count:.4f} seconds (built_count={built_count})")
	return summary


def build_all_hypergraphs(
	dataset_path: str,
	target_dir: str,
	force_rebuild: bool = False,
	using_support_only: bool = False,
	use_gpu_batch: bool = False,
	batch_size: int = 32,
	save_outputs: bool = True,
) -> dict:
	"""
	构建超图。支持 GPU 加速和批处理。
	
	Args:
		dataset_path: 数据集路径
		target_dir: 输出目录
		force_rebuild: 是否强制重建
		using_support_only: 是否仅使用支持文本
		use_gpu_batch: 是否使用 GPU + 批处理加速（推荐为 True）
		batch_size: 批处理大小，仅在 use_gpu_batch=True 时使用，默认 32
	"""
	if use_gpu_batch:
		return _build_all_hypergraphs_gpu_batch(
			dataset_path=dataset_path,
			target_dir=target_dir,
			force_rebuild=force_rebuild,
			using_support_only=using_support_only,
			batch_size=batch_size,
			save_outputs=save_outputs,
		)
	else:
		return _build_all_hypergraphs_single(
			dataset_path=dataset_path,
			target_dir=target_dir,
			force_rebuild=force_rebuild,
			using_support_only=using_support_only,
			save_outputs=save_outputs,
		)


def _print_summary(summary: dict) -> None:
	"""格式化打印完成摘要"""
	print("\n" + "="*70)
	print("✅ 超图构建完成！")
	print("="*70)
	
	total = summary.get("total_questions", 0)
	built = summary.get("built", 0)
	skipped = summary.get("skipped", 0)
	failed = summary.get("failed", 0)
	mode = summary.get("mode", "unknown")
	batch_size = summary.get("batch_size", "N/A")
	
	print(f"\n📊 处理统计:")
	print(f"  ├─ 总数:     {total}")
	print(f"  ├─ 成功:     {built} ✅")
	print(f"  ├─ 跳过:     {skipped} ⏭️")
	print(f"  └─ 失败:     {failed} ❌")
	
	print(f"\n⚙️  处理模式:")
	if mode == "gpu_batch":
		print(f"  ├─ 方式:     GPU 批处理 🚀")
		print(f"  └─ batch大小: {batch_size}")
	else:
		print(f"  └─ 方式:     单文本处理 🐢")
	
	print(f"\n📁 输出目录:")
	print(f"  └─ {summary.get('target_dir', 'N/A')}")
	print(f"\n📝 日志文件:")
	if summary.get("save_outputs", True):
		print(f"  ├─ summary.json (已保存)")
		if failed > 0:
			print(f"  └─ failed.json (包含 {failed} 条失败记录)")
	else:
		print(f"  └─ 未保存任何文件（execute-only 模式）")
	
	success_rate = (built / total * 100) if total > 0 else 0
	print(f"\n🎯 成功率: {success_rate:.1f}% ({built}/{total})")
	print("="*70 + "\n")


def main() -> None:
	parser = ArgumentParser(description="Build and store all MuSiQue question hypergraphs.")
	parser.add_argument("--dataset-path", type=str, default=dataset_path)
	parser.add_argument("--target-dir", type=str, default=target_dir)
	parser.add_argument("--force-rebuild", action="store_true")
	parser.add_argument("--using-support-only", action="store_true")
	parser.add_argument(
		"--execute-only",
		action="store_true",
		help="Run the MuSiQue build pipeline without saving any hypergraphs or metadata to disk",
	)
	parser.add_argument(
		"--use-gpu-batch",
		action="store_true",
		help="Enable GPU + batch processing for faster processing (requires CUDA and fastcoref)"
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=4096,
		help="Batch size for GPU batch processing (default: 32, recommended for fastcoref stability)"
	)
	args = parser.parse_args()

	print("\n" + "="*70)
	print("🚀 MuSiQue 超图构建工具")
	print("="*70)
	print(f"📂 数据集: {args.dataset_path}")
	print(f"📁 输出:   {args.target_dir}")
	print(f"🔧 模式:   {'GPU 批处理 🚀' if args.use_gpu_batch else '单文本处理 🐢'}")
	if args.use_gpu_batch:
		print(f"📦 batch大小: {args.batch_size}")
	print("="*70 + "\n")

	summary = build_all_hypergraphs(
		dataset_path=args.dataset_path,
		target_dir=args.target_dir,
		force_rebuild=args.force_rebuild,
		using_support_only=args.using_support_only,
		use_gpu_batch=args.use_gpu_batch,
		batch_size=args.batch_size,
		save_outputs=not args.execute_only,
	)
	
	# 打印格式化的摘要
	_print_summary(summary)


if __name__ == "__main__":
	main()
