import open_clip
import torch
import torch.nn as nn

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft

class MultiScaleAttention(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.scale1 = nn.Linear(in_dim, in_dim)
        self.scale2 = nn.Linear(in_dim, in_dim)
        self.scale3 = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scale1(x) + self.scale2(x) + self.scale3(x))

class AFF(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm1d(in_dim)
        self.ms = MultiScaleAttention(in_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        attn = torch.softmax(torch.einsum("bik,bjk->bij", q, k), dim=-1)
        out = torch.einsum("bij,bjk->bik", attn, v)
        out = out * self.ms(out)
        return self.gamma * out + x1

class SelfAttention(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax(torch.einsum("bik,bjk->bij", q, k), dim=-1)
        out = torch.einsum("bij,bjk->bik", attn, v)
        return self.gamma * out + x

class ArtifactBranch(nn.Module):
    def __init__(self, clip_model, num_features: int, normalize: bool = True):
        super().__init__()
        self.clip_model = clip_model
        self.num_features = num_features
        self.normalize = normalize
        self.aff = AFF(num_features)
        self.self_attention = SelfAttention(num_features)
        self.classifier = nn.Linear(num_features, 1)

    @torch.no_grad()
    def _clip_encode(self, img: torch.Tensor) -> torch.Tensor:
        self.clip_model.eval()
        return self.clip_model.encode_image(img, normalize=self.normalize)

    def encode(self, x: list) -> torch.Tensor:
        fused = []
        for img in x:
            feats = self._clip_encode(img)
            lf1 = feats[0].unsqueeze(0).unsqueeze(0)
            lf2 = feats[1].unsqueeze(0).unsqueeze(0)
            gf = feats[2].unsqueeze(0).unsqueeze(0)
            f = self.aff(self.aff(lf1, lf2), gf)
            fused.append(f)

        feat = torch.cat(fused, dim=0)
        feat = self.self_attention(feat)
        return feat.squeeze(1)

    def forward(self, x: list) -> torch.Tensor:
        return self.classifier(self.encode(x))

class SemanticBranch(nn.Module):
    def __init__(self, clip_model, num_features: int, hidden_dim: int = 512, normalize: bool = True):
        super().__init__()
        self.clip_model = clip_model
        self.num_features = num_features
        self.normalize = normalize
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    @torch.no_grad()
    def _clip_encode(self, img: torch.Tensor) -> torch.Tensor:
        self.clip_model.eval()
        return self.clip_model.encode_image(img[2].unsqueeze(0), normalize=self.normalize)

    def encode(self, x: list) -> torch.Tensor:
        return torch.cat([self._clip_encode(img) for img in x], dim=0)

    def forward(self, x: list) -> torch.Tensor:
        return self.mlp(self.encode(x))

class GatingNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)

class DynFakeDetector(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str,
        temp: float = 1.0,
        hard_gate: bool = False,
        freeze_clip: bool = True,
        next_to_last: bool = False,
        normalize: bool = True,
        conf_threshold: float = 0.8,
        hidden_dim_gate: int = 128,
        hidden_dim_semantic: int = 512,
    ):
        super().__init__()

        clip_model = open_clip.create_model("ViT-L-14", pretrained=pretrained_model_path)
        if next_to_last:
            num_features = clip_model.visual.proj.shape[0]
            clip_model.visual.proj = None
        else:
            num_features = clip_model.visual.output_dim

        self.artifact_branch = ArtifactBranch(clip_model, num_features, normalize)
        self.semantic_branch = SemanticBranch(clip_model, num_features, hidden_dim_semantic, normalize)
        self.gate = GatingNetwork(num_features, hidden_dim_gate)

        self.temp = temp
        self.hard_gate = hard_gate
        self.conf_threshold = conf_threshold
        self.infer_mode = 0
        self.weight_list = torch.Tensor()
        self.store_weight = False

        if freeze_clip:
            for p in clip_model.parameters():
                p.requires_grad = False

    def reset_weight(self):
        self.weight_list = torch.Tensor()
        self.store_weight = True

    def weight_stat(self) -> float:
        tmp = torch.mean(self.weight_list, dim=0)
        print(f"[Gate] artifact={tmp[0]:.4f}  semantic={tmp[1]:.4f}")
        self.store_weight = False
        return tmp[1].item()

    def forward(self, x: list):
        if self.training:
            return self._forward_train(x)
        return self._forward_infer(x)

    def _forward_train(self, x: list):
        artifact_feat = self.artifact_branch.encode(x)
        pred_artifact = self.artifact_branch.classifier(artifact_feat)

        gate_logits = self.gate(artifact_feat)
        weight = DiffSoftmax(gate_logits, tau=self.temp, hard=False)
        path = torch.randint(0, 2, (1,)).item()

        if path == 0:
            branch_logit = pred_artifact
        else:
            branch_logit = self.semantic_branch(x)

        output = weight[:, path:path + 1] * branch_logit
        gate_reg_loss = -torch.log(weight[:, path] + 1e-8).mean()
        return output, gate_reg_loss

    def _forward_infer(self, x: list):
        if self.infer_mode == 1:
            return self.artifact_branch(x)
        if self.infer_mode == 2:
            return self.semantic_branch(x)

        artifact_feat = self.artifact_branch.encode(x)
        pred_artifact = self.artifact_branch.classifier(artifact_feat)

        prob = torch.sigmoid(pred_artifact)
        confidence = (prob - 0.5).abs() * 2

        if self.store_weight:
            w_art = confidence.squeeze(1)
            w_sem = 1.0 - w_art
            self.weight_list = torch.cat(
                (self.weight_list, torch.stack([w_art, w_sem], dim=1).detach().cpu()),
                dim=0,
            )

        if (confidence >= self.conf_threshold).all():
            return pred_artifact

        pred_semantic = self.semantic_branch(x)

        if self.infer_mode == -1:
            weight = torch.full((artifact_feat.size(0), 2), 0.5, device=artifact_feat.device)
        else:
            gate_logits = self.gate(artifact_feat)
            weight = DiffSoftmax(gate_logits, tau=self.temp, hard=self.hard_gate)

        return weight[:, 0:1] * pred_artifact + weight[:, 1:2] * pred_semantic

    def freeze_clip(self):
        for p in self.artifact_branch.clip_model.parameters():
            p.requires_grad = False

    def unfreeze_clip(self):
        for p in self.parameters():
            p.requires_grad = True
