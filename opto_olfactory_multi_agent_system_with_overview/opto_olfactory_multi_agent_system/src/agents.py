from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import SystemConfig
from .models import FusionNet


@dataclass
class AgentResult:
    name: str
    data: Dict[str, Any]


class FeatureExtractionAgent:
    """
    Agent 1：特征提取 Agent。

    职责：
    - 调用双分支 CNN；
    - 对电子鼻时序和光谱图像进行特征抽取；
    - 输出基础分类概率和跨模态特征。
    """

    def __init__(self, cfg: SystemConfig, model_path: str = None, device: str = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FusionNet(cfg).to(self.device)
        self.model.eval()

        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])

    @torch.no_grad()
    def run(self, time_signal: np.ndarray, spectral_image: np.ndarray, env: np.ndarray) -> AgentResult:
        t = torch.tensor(time_signal, dtype=torch.float32).unsqueeze(0).to(self.device)
        img = torch.tensor(spectral_image, dtype=torch.float32).unsqueeze(0).to(self.device)
        e = torch.tensor(env, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits, features = self.model(t, img, e)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        ranked = sorted(
            [(self.cfg.class_names[i], float(probs[i])) for i in range(len(probs))],
            key=lambda x: x[1],
            reverse=True,
        )

        return AgentResult(
            name="FeatureExtractionAgent",
            data={
                "device": self.device,
                "probabilities": {self.cfg.class_names[i]: float(probs[i]) for i in range(len(probs))},
                "ranked_candidates": ranked,
                "time_feature_norm": float(features["time_feature"].norm(dim=1).item()),
                "image_feature_norm": float(features["image_feature"].norm(dim=1).item()),
            },
        )


class DynamicChemicalKnowledgeGraph:
    """
    简化版动态化学知识图谱。

    真实项目中可以替换为：
    - RDF/Neo4j 图数据库；
    - 光谱数据库；
    - 电子鼻响应谱库；
    - GHS 危险性分类；
    - 实验室安全手册；
    - 危险废物处置规范。
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {
            "safe": {
                "prior": 0.45,
                "hazard_level": "normal",
                "hazard": "未发现明显危险挥发物模式。",
                "possible_source": "背景空气、低浓度非特异性扰动",
                "interference": "低风险；可能受湿度、温度漂移影响。",
                "actions": [
                    "继续常规监测。",
                    "保持传感器定期标定。",
                ],
            },
            "methanol": {
                "prior": 0.14,
                "hazard_level": "warning",
                "hazard": "甲醇类挥发物具有易燃性和毒性，吸入或皮肤接触均可能造成健康风险。",
                "possible_source": "有机溶剂瓶、清洗废液、含醇混合废液挥发",
                "interference": "乙醇、异丙醇、丙酮等 VOC 可能产生交叉响应。",
                "actions": [
                    "检查有机溶剂储存柜、废液桶密封状态。",
                    "加强局部排风，避免火源和静电积累。",
                    "佩戴护目镜、防化手套，必要时撤离非相关人员。",
                ],
            },
            "ammonia": {
                "prior": 0.13,
                "hazard_level": "warning",
                "hazard": "氨气具有刺激性和腐蚀性，高浓度时会刺激呼吸道和眼部。",
                "possible_source": "含氨清洗液、碱性废液、氨水试剂瓶挥发",
                "interference": "湿度升高会增强部分金属氧化物传感器响应。",
                "actions": [
                    "检查碱性废液桶和氨水试剂瓶。",
                    "打开通风橱或局部排风装置。",
                    "避免与酸性废液混放，防止二次反应。",
                ],
            },
            "hf_like": {
                "prior": 0.08,
                "hazard_level": "danger",
                "hazard": "疑似含氟酸性挥发物模式。HF 类物质具有强腐蚀性和全身毒性风险。",
                "possible_source": "含氟刻蚀液、BOE、含氟危废桶密封不良",
                "interference": "强酸挥发物和高湿环境可能造成相似响应，需要二次确认。",
                "actions": [
                    "立即远离泄漏源，禁止徒手处理。",
                    "通知实验室安全负责人。",
                    "确认现场是否配备葡萄糖酸钙凝胶和含氟废液专用容器。",
                    "佩戴合适防护装备后再进行现场处置。",
                ],
            },
            "mixed_organic": {
                "prior": 0.20,
                "hazard_level": "warning",
                "hazard": "疑似混合有机挥发物，可能具有易燃、麻醉或刺激性风险。",
                "possible_source": "混合有机废液、光刻胶/显影液残留、清洗剂挥发",
                "interference": "多种 VOC 叠加会造成电子鼻响应非线性。",
                "actions": [
                    "检查有机危废桶是否过满或密封不良。",
                    "降低现场火源风险，保持排风。",
                    "必要时采用 GC-MS、FTIR 或标准气体进行复核。",
                ],
            },
        }

    def get(self, class_name: str) -> Dict[str, Any]:
        return self.nodes[class_name]


class FusionReasoningAgent:
    """
    Agent 2：长链推理与融合决策 Agent。

    这里的“长链推理”是工程系统中的显式推理轨迹：
    - 先读取深度模型概率；
    - 再引入知识图谱先验；
    - 再根据温湿度估计交叉干扰；
    - 最后用简化贝叶斯更新得到后验风险。
    """

    def __init__(self, cfg: SystemConfig, kg: DynamicChemicalKnowledgeGraph):
        self.cfg = cfg
        self.kg = kg

    def _environment_factor(self, class_name: str, env: np.ndarray) -> float:
        temperature, humidity = float(env[0]), float(env[1])

        factor = 1.0

        # 湿度较高时，氨气和 HF-like 酸性挥发物的交叉干扰更强
        if class_name in ["ammonia", "hf_like"] and humidity > 0.65:
            factor *= 1.0 + 0.45 * (humidity - 0.65)

        # 温度较高时，有机挥发物释放增强
        if class_name in ["methanol", "mixed_organic"] and temperature > 0.60:
            factor *= 1.0 + 0.55 * (temperature - 0.60)

        # safe 类别在高温高湿下可信度略微下降
        if class_name == "safe" and (temperature > 0.70 or humidity > 0.75):
            factor *= 0.75

        return float(max(factor, 1e-6))

    def run(self, feature_result: AgentResult, env: np.ndarray) -> AgentResult:
        model_probs: Dict[str, float] = feature_result.data["probabilities"]

        unnormalized = {}
        trace: List[str] = []

        trace.append("步骤1：读取特征提取 Agent 的跨模态分类概率。")
        best_model_label = max(model_probs, key=model_probs.get)
        trace.append(f"步骤2：深度模型初步认为最可能类别为 {best_model_label}，概率为 {model_probs[best_model_label]:.3f}。")
        trace.append("步骤3：引入动态化学知识图谱中的先验概率和危险性信息。")
        trace.append("步骤4：根据当前温湿度估计交叉干扰，对不同类别的似然进行修正。")

        for class_name, prob in model_probs.items():
            node = self.kg.get(class_name)
            prior = float(node["prior"])
            env_factor = self._environment_factor(class_name, env)
            unnormalized[class_name] = prob * prior * env_factor

        total = sum(unnormalized.values()) + 1e-12
        posterior = {k: float(v / total) for k, v in unnormalized.items()}
        ranked = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
        final_label, final_score = ranked[0]

        node = self.kg.get(final_label)
        trace.append(
            f"步骤5：贝叶斯融合后，系统将 {final_label} 作为最终候选，后验风险概率为 {final_score:.3f}。"
        )
        trace.append(
            f"步骤6：知识图谱提示该类别的可能来源为：{node['possible_source']}；主要干扰为：{node['interference']}"
        )

        return AgentResult(
            name="FusionReasoningAgent",
            data={
                "posterior_probabilities": posterior,
                "ranked_posterior": ranked,
                "final_label": final_label,
                "final_score": final_score,
                "reasoning_trace": trace,
                "knowledge_node": node,
            },
        )


class SafetyComplianceAgent:
    """
    Agent 3：安全合规预警 Agent。

    注意：这里给的是示例规则，真实实验室必须替换为本单位正式 SOP、
    危化品管理规范和应急预案。
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    def _alert_level(self, hazard_level: str, score: float) -> str:
        if hazard_level == "danger" or score >= self.cfg.danger_threshold:
            return "DANGER"
        if hazard_level == "warning" or score >= self.cfg.warning_threshold:
            return "WARNING"
        return "NORMAL"

    def run(self, reasoning_result: AgentResult) -> AgentResult:
        data = reasoning_result.data
        label = data["final_label"]
        score = float(data["final_score"])
        node = data["knowledge_node"]
        alert_level = self._alert_level(node["hazard_level"], score)

        if alert_level == "NORMAL":
            summary = "当前未触发危险预警，但建议保持连续监测。"
        elif alert_level == "WARNING":
            summary = "系统检测到潜在危险挥发物模式，建议尽快进行人工复核和源头排查。"
        else:
            summary = "系统检测到高风险挥发物模式，应立即启动实验室应急处置流程。"

        report = {
            "system": "Optoelectronic Fusion Olfactory Data Analysis System",
            "alert_level": alert_level,
            "detected_class": label,
            "risk_score": round(score, 4),
            "hazard_description": node["hazard"],
            "possible_source": node["possible_source"],
            "main_interference": node["interference"],
            "recommended_actions": node["actions"],
            "summary": summary,
            "reasoning_trace": data["reasoning_trace"],
            "posterior_probabilities": data["posterior_probabilities"],
            "disclaimer": "该报告由原型系统生成，仅供科研和教学演示，不能替代正式安全评估和人工应急判断。",
        }

        return AgentResult(name="SafetyComplianceAgent", data=report)


class MultiAgentOlfactorySystem:
    """多 Agent 协同总控。"""

    def __init__(self, cfg: SystemConfig, model_path: str = None):
        self.cfg = cfg
        self.kg = DynamicChemicalKnowledgeGraph()
        self.feature_agent = FeatureExtractionAgent(cfg, model_path=model_path)
        self.reasoning_agent = FusionReasoningAgent(cfg, self.kg)
        self.safety_agent = SafetyComplianceAgent(cfg)

    def analyze(self, time_signal: np.ndarray, spectral_image: np.ndarray, env: np.ndarray) -> Dict[str, Any]:
        feature_result = self.feature_agent.run(time_signal, spectral_image, env)
        reasoning_result = self.reasoning_agent.run(feature_result, env)
        safety_result = self.safety_agent.run(reasoning_result)

        return {
            "feature_agent": feature_result.data,
            "reasoning_agent": reasoning_result.data,
            "safety_report": safety_result.data,
        }
