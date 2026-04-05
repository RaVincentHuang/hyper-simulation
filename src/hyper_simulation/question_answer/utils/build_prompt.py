from hyper_simulation.llm.prompt.hotpot_qa import HOTPOT_QA_BASE
from hyper_simulation.llm.prompt.musique import MUSIQUE_QA_BASE
from hyper_simulation.llm.prompt.multihop import MULTIHOP_QA_BASE
from hyper_simulation.llm.prompt.legalbench_qa import LEGALBENCH_QA_BASE
from hyper_simulation.llm.prompt.legalbench_qa_detailed import QA_CONTRACT_BASE, QA_CONSUMER_BASE, QA_PRIVACY_BASE, QA_RULE_BASE
from hyper_simulation.llm.prompt.legalbench_sara_entailment import LEGALBENCH_SARA_ENTAILMENT_BASE
from hyper_simulation.llm.prompt.legalbench_privacy_policy_entailment import LEGALBENCH_PRIVACY_POLICY_ENTAILMENT_BASE
from hyper_simulation.llm.prompt.legalbench_insurance import LEGALBENCH_INSURANCE_BASE
from hyper_simulation.llm.prompt.legalbench_corporate_lobbying import LEGALBENCH_CORPORATE_LOBBYING_BASE
from hyper_simulation.llm.prompt.legalbench_scalr import LEGALBENCH_SCALR_BASE
from hyper_simulation.llm.prompt.arc import ARC_BASE

def build_prompt(question: str, context_text: str, task: str = "hotpotqa", context_type: str = None) -> str:
    """
    构建用于LLM的prompt，根据不同任务选择相应的模板
    
    Args:
        question: 问题文本
        context_text: 格式化后的context文本
        task: 任务类型 (hotpotqa, musique, multihop, qa/*, legalbench/*)
    
    Returns:
        完整的prompt
    """
    if task == "hotpotqa":
        prompt = HOTPOT_QA_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "musique":
        prompt = MUSIQUE_QA_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "multihop":
        prompt = MULTIHOP_QA_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "legalbench":
        if context_type == "contract":
            prompt = QA_CONTRACT_BASE.format(context_text=context_text, question=question)
        elif context_type == "tos":
            prompt = QA_CONSUMER_BASE.format(context_text=context_text, question=question)
        elif context_type == "privacy_policy":
            prompt = QA_PRIVACY_BASE.format(context_text=context_text, question=question)
        elif context_type == "rules":
            prompt = QA_RULE_BASE.format(context_text=context_text, question=question)
        elif context_type == "legal_sara_entailment":
            prompt = LEGALBENCH_SARA_ENTAILMENT_BASE.format(context_text=context_text, question=question)
        elif context_type == "legal_privacy_policy_entailment":
            prompt = LEGALBENCH_PRIVACY_POLICY_ENTAILMENT_BASE.format(context_text=context_text, question=question)
        elif context_type == "legal_insurance":
            prompt = LEGALBENCH_INSURANCE_BASE.format(context_text=context_text, question=question)
        elif context_type == "legal_corporate_lobbying":
            prompt = LEGALBENCH_CORPORATE_LOBBYING_BASE.format(context_text=context_text, question=question)
        elif context_type == "legal_case":
            prompt = LEGALBENCH_SCALR_BASE.format(context_text=context_text, question=question)
        else:
            # 兜底：用通用 legalbench prompt
            prompt = LEGALBENCH_QA_BASE.format(context_text=context_text, question=question)
    elif task == "qa/contract":
        prompt = QA_CONTRACT_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "qa/consumer":
        prompt = QA_CONSUMER_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "qa/privacy":
        prompt = QA_PRIVACY_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "qa/rule":
        prompt = QA_RULE_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/qa"):
        prompt = LEGALBENCH_QA_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/sara_entailment"):
        prompt = LEGALBENCH_SARA_ENTAILMENT_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/privacy_policy_entailment"):
        prompt = LEGALBENCH_PRIVACY_POLICY_ENTAILMENT_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/insurance"):
        prompt = LEGALBENCH_INSURANCE_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/corporate_lobbying"):
        prompt = LEGALBENCH_CORPORATE_LOBBYING_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/scalr"):
        prompt = LEGALBENCH_SCALR_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "ARC":
        prompt = ARC_BASE.format(
            context_text=context_text,
            question=question
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    return prompt
