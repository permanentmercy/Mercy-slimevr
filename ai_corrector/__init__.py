"""
ai_corrector
============
SlimeVR 神经运动学时空残差校正器 (AI Kinematics Residual Corrector).

包含模块:
- corrector: CSTN 因果时空模型、损失函数、GPU 训练与流式 ONNX 导出
- database: 动捕数据集 (AMASS, SMPL-H) 高并发下载与切窗加噪引擎
- checkpoints: 本地分层权重生命周期管理系统 (best, latest, top_k)
- artifacts: 导出的生产 ONNX 模型与评测基准报告
- latex: 技术规范与数学证明文档单一源
"""
from .corrector.model import Corrector, Config

__all__ = ["Corrector", "Config"]
