import os
import json
import time
from nltk.corpus import wordnet as wn
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from hyper_simulation.llm.chat_completion import get_generate
from tqdm import tqdm

OUTPUT_FILE = "qwen_ontology_mapping.json"
PROGRESS_FILE = "qwen_ontology_progress.json"
BATCH_SIZE = 20  # 每次传给 get_generate 的 prompt 数量 (取决于你的显存大小和 Ollama 并发设置)

VALID_CATEGORIES = [
    "PERSON", "COUNTRY", "LOC", "ORG", "FAC", "NORP", "PRODUCT", 
    "WORK_OF_ART", "LAW", "LANGUAGE", "OCCUPATION", "EVENT", 
    "TEMPORAL", "NUMBER", "CONCEPT", "NOT_ENT"
]

CANDIDATE = [
    "ORGANISM: Living being, such as animal, plant, or microorganism.",
    "FOOD: Edible substance, dish, or cuisine.",
    "MEDICAL: Medical condition, disease, symptom, or treatment.",
    "ANATOMY: Body part, organ, or anatomical structure.",
    "SUBSTANCE: Chemical element, compound, or material.",
    "ASTRO: Astronomical object, such as a star, planet, or galaxy.",
    "AWARD: Prize, honor, or recognition given to a person or organization.",
    "VEHICLE: Means of transportation, such as a car, airplane, or bicycle.",
]

# 针对单条数据的精简 Prompt，极其适合 9B 模型
SINGLE_PROMPT_TEMPLATE = """
You are an expert Ontologist. Classify the following WordNet synset into EXACTLY ONE of these categories:
PERSON: Human being, individual, or specific character.
COUNTRY: A nation with its own government.
LOC: Geographical location, natural region, body of water.
ORG: Organization, institution, company, government body.
FAC: Physical building, facility, structure.
​GPE​​: Geopolitical entity, such as cities, states, provinces (but not countries).
NORP: Nationalities, religious or political groups.
PRODUCT: Physical object, vehicle, device, manufactured good.
WORK_OF_ART: Piece of art, publication, show.
LAW: Legal document, binding agreement.
LANGUAGE: Spoken or written human language.
OCCUPATION: Job, profession, trade.
EVENT: Phenomenon, historical event, sports match.
TEMPORAL: Time period, specific date, unit of time.
NUMBER: Mathematical number, quantity.
CONCEPT: Abstract idea, theoretical concept.
ORGANISM: Living being, such as animal, plant, or microorganism.
FOOD: Edible substance, dish, or cuisine.
MEDICAL: Medical condition, disease, symptom, or treatment.
ANATOMY: Body part, organ, or anatomical structure.
SUBSTANCE: Chemical element, compound, or material.
ASTRO: Astronomical object, such as a star, planet, or galaxy.
AWARD: Prize, honor, or recognition given to a person or organization.
VEHICLE: Means of transportation, such as a car, airplane, or bicycle.
NOT_ENT: Use this if the synset does not fit any of the above 24 categories.

Synset Label: {label}
Meaning & Examples: {text}

Output ONLY the category name from the list above. Do not output any other words or explanations.
"""

# ================= 3. 辅助解析函数 =================
def extract_category(response_text: str) -> str:
    """
    清洗本地模型的输出。即使 Qwen 啰嗦了（比如输出了 "The category is: ORG"），
    也能安全提取出对应的类别。
    """
    response_upper = response_text.strip().upper()
    for cat in VALID_CATEGORIES:
        # 匹配到任何一个合法类别即返回
        if cat in response_upper:
            return cat
    return "NOT_ENT" # 如果模型胡言乱语，默认归为 NOT_ENT


def load_json_file(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def atomic_json_dump(path: str, data) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(temp_path, path)


def save_state(mapping: dict, total_pending: int, done_in_run: int) -> None:
    atomic_json_dump(OUTPUT_FILE, mapping)
    progress_payload = {
        "updated_at": int(time.time()),
        "mapped_total": len(mapping),
        "current_run_total_pending": total_pending,
        "current_run_done": done_in_run,
    }
    atomic_json_dump(PROGRESS_FILE, progress_payload)

# ================= 4. 主控流程 (包含断点续传) =================
def main():
    # 初始化你的本地模型
    print("正在连接本地 Qwen 模型...")
    llm = ChatOllama(model="qwen3.5:9b", top_p=0.95, reasoning=False)

    # 读取已有的进度（断点续传核心逻辑）
    existing_mapping = load_json_file(OUTPUT_FILE, default={})
    if existing_mapping:
        print(f"✅ 找到本地存档，已加载 {len(existing_mapping)} 个已处理标签，准备继续...")
    elif os.path.exists(OUTPUT_FILE):
        print("⚠️ 存档文件不可读，将从头开始。")

    progress = load_json_file(PROGRESS_FILE, default={})
    if progress:
        print(
            "📌 上次记录: "
            f"mapped_total={progress.get('mapped_total', 0)}, "
            f"current_run_done={progress.get('current_run_done', 0)}/{progress.get('current_run_total_pending', 0)}"
        )

    print("正在从 WordNet 提取待处理名词...")
    pending_tasks = []
    
    # 遍历所有名词
    for syn in wn.all_synsets(pos='n'):
        label = syn.name()
        
        # 【关键】：如果这个标签已经在存档里了，就跳过！
        if label in existing_mapping:
            continue
            
        text = syn.definition()
        if syn.examples():
            text += ". " + " ".join(syn.examples())
            
        pending_tasks.append({"label": label, "text": text})

    total_pending = len(pending_tasks)
    if total_pending == 0:
        print("🎉 所有 WordNet 名词都已经处理完毕了！")
        return

    print(f"共有 {total_pending} 个词条需要处理。开始批处理 (Batch Size = {BATCH_SIZE})...")

    done_in_run = 0

    # 分批次处理
    try:
        for i in tqdm(range(0, total_pending, BATCH_SIZE)):
            batch = pending_tasks[i : i + BATCH_SIZE]

            # 1. 为批次中的每个词组装专属 Prompt
            batch_prompts = []
            for item in batch:
                prompt = SINGLE_PROMPT_TEMPLATE.format(label=item['label'], text=item['text'])
                batch_prompts.append(prompt)

            # 2. 调用你的函数进行推理
            try:
                responses = get_generate(batch_prompts, llm)

                # 3. 按条实时落盘，避免中断时丢失整个 batch
                for item, response_text in zip(batch, responses):
                    category = extract_category(response_text)
                    existing_mapping[item['label']] = category
                    done_in_run += 1
                    save_state(existing_mapping, total_pending, done_in_run)

            except Exception as e:
                print(f"❌ 批次 {i} 到 {i+BATCH_SIZE} 推理失败: {e}")
                print("休眠 5 秒后继续下一个批次...")
                time.sleep(5)
                continue

    except KeyboardInterrupt:
        print("\n⛔ 检测到中断，正在保存当前进度...")
        save_state(existing_mapping, total_pending, done_in_run)
        print("✅ 已保存，可下次继续执行。")
        return

    # 结束时再保存一次，确保最终状态一致
    save_state(existing_mapping, total_pending, done_in_run)
    print("✅ 全部处理完成，已保存最终映射与进度文件。")
            
if __name__ == "__main__":
    main()