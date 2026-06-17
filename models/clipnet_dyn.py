"""
clipnet_dyn.py  ―  Dynamic Fake Image Detector
================================================

학습 전략 (DynMM 방식)
-----------------------
매 배치마다 path ∈ {0=artifact, 1=semantic}를 랜덤 샘플링하여
선택된 branch 하나만 실행하고 그 branch의 task loss를 backprop.
Gate network는 artifact feature로 두 branch의 weight를 항상 예측하되,
선택된 path에 해당하는 gate weight에 -log 페널티를 추가.
  → "선택된 expert를 더 잘 선택하도록" gate가 학습됨
  → 각 branch는 독립 expert로 균등하게 학습됨

추론 전략 (Confidence-based Early Exit)
-----------------------------------------
1) Artifact branch 실행 → confidence = |sigmoid(logit) - 0.5| * 2  ∈ [0,1]
2) confidence ≥ conf_threshold  →  artifact 결과만 반환  (semantic 실행 X)
3) confidence <  conf_threshold  →  semantic branch 추가 실행
                                     gate weight로 두 logit을 가중합하여 반환

반환 형식
---------
training=True  : (logit [B,1], gate_reg_loss scalar)
                 train.py 에서  loss = task_loss + lossw * gate_reg_loss
eval=True      : logit [B,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


# ============================================================================
# 유틸
# ============================================================================

def DiffSoftmax(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """Differentiable Softmax with optional straight-through hard gate."""
    y_soft = (logits / tau).softmax(dim)
    if hard:
        index  = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


# ============================================================================
# 공통 attention 모듈 (clipnet.py 그대로)
# ============================================================================

class MultiScaleAttention(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.scale1 = nn.Linear(in_dim, in_dim)
        self.scale2 = nn.Linear(in_dim, in_dim)
        self.scale3 = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scale1(x) + self.scale2(x) + self.scale3(x))


class AFF(nn.Module):
    """Attentive Feature Fusion (MultiScale 포함)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key   = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.bn    = nn.BatchNorm1d(in_dim)
        self.ms    = MultiScaleAttention(in_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1, x2 : [B, num_patches, F]
        q   = self.query(x1)
        k   = self.key(x2)
        v   = self.value(x2)
        attn = torch.softmax(torch.einsum("bik,bjk->bij", q, k), dim=-1)
        out  = torch.einsum("bij,bjk->bik", attn, v)
        out  = out * self.ms(out)          # multiscale channel attention
        return self.gamma * out + x1       # residual


class SelfAttention(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key   = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q    = self.query(x)
        k    = self.key(x)
        v    = self.value(x)
        attn = torch.softmax(torch.einsum("bik,bjk->bij", q, k), dim=-1)
        out  = torch.einsum("bij,bjk->bik", attn, v)
        return self.gamma * out + x


# ============================================================================
# Branch 1 : Artifact Branch
# ============================================================================

class ArtifactBranch(nn.Module):
    """
    local patch 2장 + global patch 1장을 AFF로 융합 → SelfAttention → 분류.
    encode() 는 gate 입력용 feature [B, F]를 반환하고
    classifier 는 별도로 호출할 수 있어서 gate가 feature를 재사용 가능.
    """

    def __init__(self, clip_model, num_features: int, normalize: bool = True):
        super().__init__()
        self.clip_model   = clip_model
        self.num_features = num_features
        self.normalize    = normalize

        self.aff            = AFF(num_features)
        self.self_attention = SelfAttention(num_features)
        self.classifier     = nn.Linear(num_features, 1)

    @torch.no_grad()
    def _clip_encode(self, img: torch.Tensor) -> torch.Tensor:
        """단일 이미지 텐서를 CLIP으로 인코딩 (gradient 없음)."""
        self.clip_model.eval()
        return self.clip_model.encode_image(img, normalize=self.normalize)

    def encode(self, x: list) -> torch.Tensor:
        """
        x   : list[Tensor], len=B, 각 원소 shape [3, C, H, W]
                (index 0,1 = local patch, index 2 = global patch)
        반환 : fused feature [B, F]
        """
        fused = []
        for img in x:
            feats = self._clip_encode(img)                     # [3, F]
            lf1   = feats[0].unsqueeze(0).unsqueeze(0)         # [1,1,F]
            lf2   = feats[1].unsqueeze(0).unsqueeze(0)
            gf    = feats[2].unsqueeze(0).unsqueeze(0)
            f     = self.aff(self.aff(lf1, lf2), gf)           # [1,1,F]
            fused.append(f)

        feat = torch.cat(fused, dim=0)                         # [B,1,F]
        feat = self.self_attention(feat)                       # [B,1,F]
        return feat.squeeze(1)                                 # [B,  F]

    def forward(self, x: list) -> torch.Tensor:
        """반환: logit [B,1]"""
        return self.classifier(self.encode(x))


# ============================================================================
# Branch 2 : Semantic Branch
# ============================================================================

class SemanticBranch(nn.Module):
    """
    CLIP global-patch feature → MLP 분류기.
    Artifact branch보다 더 전역적(semantic)인 위조 패턴을 탐지.
    """

    def __init__(
        self,
        clip_model,
        num_features: int,
        hidden_dim: int  = 512,
        normalize: bool  = True,
    ):
        super().__init__()
        self.clip_model   = clip_model
        self.num_features = num_features
        self.normalize    = normalize

        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    @torch.no_grad()
    def _clip_encode(self, img: torch.Tensor) -> torch.Tensor:
        self.clip_model.eval()
        # index 2 = global patch
        return self.clip_model.encode_image(
            img[2].unsqueeze(0), normalize=self.normalize
        )                                                      # [1, F]

    def encode(self, x: list) -> torch.Tensor:
        """반환: [B, F]"""
        return torch.cat([self._clip_encode(img) for img in x], dim=0)

    def forward(self, x: list) -> torch.Tensor:
        """반환: logit [B,1]"""
        return self.mlp(self.encode(x))


# ============================================================================
# Gating Network
# ============================================================================

class GatingNetwork(nn.Module):
    """
    Artifact feature [B, F] → raw gate logits [B, 2].
    DiffSoftmax를 통해 [w_artifact, w_semantic]으로 변환.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)              # [B, 2]


# ============================================================================
# DynFakeDetector  ―  최종 모델
# ============================================================================

class DynFakeDetector(nn.Module):
    """
    Parameters
    ----------
    pretrained_model_path : str
    temp : float
        DiffSoftmax 온도. 작을수록 한쪽 branch를 강하게 선택.
    hard_gate : bool
        추론 시 straight-through hard gate 사용 여부.
    freeze_clip : bool
        CLIP backbone freeze 여부.
    next_to_last : bool
        CLIP visual projection 제거 여부.
    normalize : bool
    conf_threshold : float
        Early-exit 기준 confidence. 0~1.
        artifact confidence ≥ conf_threshold 이면 semantic branch 생략.
    hidden_dim_gate : int
    hidden_dim_semantic : int

    infer_mode (외부에서 직접 설정)
    --------------------------------
      0  : dynamic (confidence-based early exit)  ← 기본값
      1  : artifact branch 고정
      2  : semantic branch 고정
     -1  : uniform weight (ablation)
    """

    def __init__(
        self,
        pretrained_model_path: str,
        temp: float            = 1.0,
        hard_gate: bool        = False,
        freeze_clip: bool      = True,
        next_to_last: bool     = False,
        normalize: bool        = True,
        conf_threshold: float  = 0.8,
        hidden_dim_gate: int   = 128,
        hidden_dim_semantic: int = 512,
    ):
        super().__init__()

        # ── CLIP backbone ──────────────────────────────────────────────────
        clip_model = open_clip.create_model(
            "ViT-L-14", pretrained=pretrained_model_path
        )
        if next_to_last:
            num_features = clip_model.visual.proj.shape[0]
            clip_model.visual.proj = None
        else:
            num_features = clip_model.visual.output_dim

        # ── 두 branch (CLIP backbone 공유) ────────────────────────────────
        self.artifact_branch = ArtifactBranch(clip_model, num_features, normalize)
        self.semantic_branch = SemanticBranch(
            clip_model, num_features, hidden_dim_semantic, normalize
        )

        # ── Gate ──────────────────────────────────────────────────────────
        self.gate = GatingNetwork(num_features, hidden_dim_gate)

        # ── 하이퍼파라미터 ────────────────────────────────────────────────
        self.temp           = temp
        self.hard_gate      = hard_gate
        self.conf_threshold = conf_threshold
        self.infer_mode     = 0   # 기본값: dynamic early-exit

        # ── 가중치 통계 (affect_dyn.py 호환) ──────────────────────────────
        self.weight_list  = torch.Tensor()
        self.store_weight = False

        # ── CLIP freeze ───────────────────────────────────────────────────
        if freeze_clip:
            for p in clip_model.parameters():
                p.requires_grad = False

    # ── 통계 헬퍼 ─────────────────────────────────────────────────────────

    def reset_weight(self):
        self.weight_list  = torch.Tensor()
        self.store_weight = True

    def weight_stat(self) -> float:
        """semantic branch 평균 사용 비율을 출력하고 반환."""
        tmp = torch.mean(self.weight_list, dim=0)
        print(f"[Gate] artifact={tmp[0]:.4f}  semantic={tmp[1]:.4f}")
        self.store_weight = False
        return tmp[1].item()

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: list):
        """
        학습 시
        --------
        DynMM 방식: path ∈ {0, 1}를 랜덤 샘플링하여 해당 branch만 실행.
        Gate network의 해당 path weight에 -log 페널티를 추가해
        gate가 올바른 branch를 선택하도록 학습.

        반환: (logit [B,1], gate_reg_loss scalar)
          → train.py: loss = task_loss + lossw * gate_reg_loss

        추론 시
        --------
        Confidence-based early exit:
          1) Artifact branch 실행 → confidence = |sigmoid(logit) - 0.5| * 2
          2) 모든 샘플이 conf_threshold 이상 → artifact logit 그대로 반환
          3) 일부 샘플이 미달 → 미달 샘플만 semantic branch 실행 후
             gate weight로 가중합하여 최종 logit 생성

        반환: logit [B,1]
        """
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_infer(x)

    # ── 학습 forward ──────────────────────────────────────────────────────

    def _forward_train(self, x: list):
        """
        DynMM 학습 방식
        ---------------
        Step 1. Artifact branch encode (항상 실행 — gate 입력 필요)
        Step 2. Gate weight 계산
        Step 3. path를 랜덤 샘플링 (0=artifact, 1=semantic)
        Step 4. 선택된 branch의 logit 계산
        Step 5. task_loss용 logit = gate_weight * branch_logit   (soft)
        Step 6. gate_reg_loss = -log(gate_weight[path])          (gate 학습)

        gate_reg_loss 의 의미:
          선택된 path의 gate weight가 클수록(=그 branch를 자신있게 선택)
          페널티가 작아짐. 즉 gate가 "이 샘플은 artifact만으로 충분하다"를
          학습하면 w_artifact → 1이 되어 페널티 → 0.
        """
        # Step 1. Artifact encode (항상 필요)
        artifact_feat = self.artifact_branch.encode(x)              # [B, F]
        pred_artifact = self.artifact_branch.classifier(artifact_feat)  # [B,1]

        # Step 2. Gate weight
        gate_logits = self.gate(artifact_feat)                      # [B, 2]
        weight = DiffSoftmax(
            gate_logits, tau=self.temp, hard=False                  # train: soft
        )                                                            # [B, 2]

        # Step 3. 랜덤 path 샘플링 (배치 단위로 하나 선택)
        path = torch.randint(0, 2, (1,)).item()                     # 0 or 1

        # Step 4. 선택된 branch logit
        if path == 0:
            # artifact branch → 이미 계산됨
            branch_logit = pred_artifact                            # [B,1]
        else:
            # semantic branch → 이때만 실행 (연산 절약)
            branch_logit = self.semantic_branch(x)                  # [B,1]

        # Step 5. 출력 logit: gate weight가 반영된 soft weighted output
        #   → 양쪽 branch가 모두 gate weight에 의해 조율되도록 유지
        #   선택된 branch의 weight만 반영 (다른 branch는 이번 step에서 미실행)
        output = weight[:, path : path + 1] * branch_logit          # [B,1]

        # Step 6. Gate regularization loss
        #   -log(w_path) : 선택된 path의 confidence를 높이도록 gate를 학습
        #   mean over batch
        gate_reg_loss = -torch.log(weight[:, path] + 1e-8).mean()

        return output, gate_reg_loss

    # ── 추론 forward ──────────────────────────────────────────────────────

    def _forward_infer(self, x: list):
        """
        Confidence-based Early Exit
        ----------------------------
        1) Artifact branch 항상 실행
        2) confidence = |sigmoid(logit) - 0.5| * 2  ∈ [0,1]
           confidence가 1에 가까울수록 "확실하게 판단됨"
        3) infer_mode != 0 이면 단일 branch 고정 실행
        4) dynamic 모드:
           - 배치 내 모든 샘플의 confidence ≥ conf_threshold
             → semantic branch 생략, artifact logit 반환
           - 그렇지 않으면 (일부라도 불확실)
             → 전체 배치에 semantic branch 실행 후 gate 가중합
             (샘플별 early exit도 가능하지만 배치 단위가 GPU 효율적)
        """
        # ── infer_mode 분기 ───────────────────────────────────────────────
        if self.infer_mode == 1:
            return self.artifact_branch(x)
        if self.infer_mode == 2:
            return self.semantic_branch(x)

        # ── Step 1. Artifact branch ───────────────────────────────────────
        artifact_feat = self.artifact_branch.encode(x)              # [B, F]
        pred_artifact = self.artifact_branch.classifier(artifact_feat)  # [B,1]

        # ── Step 2. Confidence 계산 ───────────────────────────────────────
        prob       = torch.sigmoid(pred_artifact)                   # [B,1]
        confidence = (prob - 0.5).abs() * 2                        # [B,1], ∈[0,1]

        # 통계 저장 (gate weight 로 기록 — artifact conf를 w_artifact 로 취급)
        if self.store_weight:
            # [w_artifact, w_semantic] 형식 유지
            w_art = confidence.squeeze(1)
            w_sem = 1.0 - w_art
            self.weight_list = torch.cat(
                (self.weight_list,
                 torch.stack([w_art, w_sem], dim=1).detach().cpu()),
                dim=0,
            )

        # ── Step 3. Early exit 판단 ───────────────────────────────────────
        all_confident = (confidence >= self.conf_threshold).all()

        if all_confident:
            # Semantic branch 완전 생략
            return pred_artifact

        # ── Step 4. Semantic branch 실행 (불확실한 샘플 존재) ─────────────
        pred_semantic = self.semantic_branch(x)                     # [B,1]

        # ── Step 5. Gate weight 계산 후 가중합 ───────────────────────────
        if self.infer_mode == -1:
            # ablation: uniform weight
            weight = torch.full(
                (artifact_feat.size(0), 2), 0.5,
                device=artifact_feat.device
            )
        else:
            gate_logits = self.gate(artifact_feat)                  # [B, 2]
            weight = DiffSoftmax(
                gate_logits, tau=self.temp, hard=self.hard_gate
            )                                                        # [B, 2]

        output = weight[:, 0:1] * pred_artifact + weight[:, 1:2] * pred_semantic
        return output                                                # [B,1]

    # ── 유틸 ──────────────────────────────────────────────────────────────

    def freeze_clip(self):
        for p in self.artifact_branch.clip_model.parameters():
            p.requires_grad = False

    def unfreeze_clip(self):
        for p in self.parameters():
            p.requires_grad = True
