"""
SparseCL 矛盾检测基线
基于论文: SparseCL: Sparse Contrastive Learning for Contradiction Detection
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from hyper_simulation.query_instance import QueryInstance

class SparseCLScorer:
    """SparseCL 评分器（单例模式，避免重复加载模型）"""
    _instance = None
    
    def __new__(cls, model_path: str, device: str = "cuda"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_path: str, device: str = "cuda"):
        if self._initialized:
            return
        self.device = device
        self.model_path = model_path
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        """加载模型（只在第一次初始化时执行）"""
        print(f"[SparseCL] 正在加载模型：{self.model_path} 到 {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True).to(self.device)
        self.model.eval()
    
    def _mean_pooling(self, model_output, attention_mask):
        """平均池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(self, text: str):
        """获取单个文本的嵌入向量"""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = self._mean_pooling(outputs, inputs['attention_mask'])
        return F.normalize(embedding, p=2, dim=1)
    
    def _calc_hoyer_sparsity(self, v1: torch.Tensor, v2: torch.Tensor):
        """计算 Hoyer 稀疏度"""
        diff = v1 - v2
        d = diff.shape[1]
        sqrt_d = torch.sqrt(torch.tensor(d, device=self.device))
        l1_norm = torch.norm(diff, p=1, dim=1)
        l2_norm = torch.norm(diff, p=2, dim=1)
        l2_norm = torch.clamp(l2_norm, min=1e-9)
        hoyer = (sqrt_d - (l1_norm / l2_norm)) / (sqrt_d - 1)
        return hoyer
    
    def compute_score(self, text_a: str, text_b: str, alpha: float = 1.0) -> float:
        """计算矛盾得分"""
        emb_a = self.get_embedding(text_a)
        emb_b = self.get_embedding(text_b)
        cosine_score = torch.sum(emb_a * emb_b, dim=1)
        hoyer_score = self._calc_hoyer_sparsity(emb_a, emb_b)
        final_score = cosine_score + alpha * hoyer_score
        return final_score.item()


def query_fixup(
    query_instance: QueryInstance, 
    model_path: str = "/home/vincent/hyper-simulation/models/GTE-SparseCL-msmarco", 
    alpha: float = 1.5
) -> QueryInstance:
    """
    使用 SparseCL 计算矛盾得分，直接注入 fixed_data
    格式: "[Score: 2.34] document text..."
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # 初始化评分器（单例）
    scorer = SparseCLScorer(model_path)
    
    query = query_instance.query
    documents = query_instance.data if query_instance.data else []
    
    if not documents:
        return query_instance
    
    # 计算得分并构建 fixed_data
    fixed_data = []
    for idx, doc in enumerate(documents):
        try:
            score = scorer.compute_score(query, doc, alpha=alpha)
            # ✅ 极简格式：得分 + 原文
            fixed_doc = f"[SparseCL: {score:.2f}] {doc}"
            fixed_data.append(fixed_doc)
        except Exception as e:
            logger.warning(f"[SparseCL] 文档 {idx} 评分失败：{e}")
            fixed_data.append(doc)  # 失败则保留原文档
    
    # 创建修正后的实例
    from copy import deepcopy
    fixed_instance = deepcopy(query_instance)
    fixed_instance.fixed_data = fixed_data
    
    logger.info(f"[SparseCL] 完成：{len(documents)} 个文档已注入得分")
    return fixed_instance