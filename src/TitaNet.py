from pathlib import Path
import numpy as np
import torch, librosa, yaml
from typing import Mapping
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from model_optim import prune_encoder_layers, quantize_model
from utils import _centroid, _l2norm, z_norm_score, calibrated_confidence

class TitaNet:
    def __init__(
        self,
        config_path: str | None = "configs/lightweight_titanet.yaml",
        device: str | None = None,
        use_pruned_model: bool | None = None,
    ):
        self.cfg = self._load_config(config_path)
        model_cfg = self.cfg.get("model", {})
        self.device = device or model_cfg.get("device", "cpu")
        pretrained_name = model_cfg.get("pretrained_name", "titanet_large")
        quant_cfg = dict(model_cfg.get("quantization", {}))

        pruned_model_path = model_cfg.get("pruned_path")
        load_pruned = model_cfg.get("load_pruned", False)
        if use_pruned_model is not None:
            load_pruned = use_pruned_model

        self.model_source = "pretrained"

        if load_pruned and pruned_model_path:
            opt_path = Path(pruned_model_path)
            if not opt_path.exists():
                raise FileNotFoundError(f"Optimized model not found at {opt_path}")
            self.model = EncDecSpeakerLabelModel.restore_from(
                restore_path=str(opt_path),
                map_location=self.device,
            )
            self.model_source = "exported"
        else:
            self.model = EncDecSpeakerLabelModel.from_pretrained(model_name=pretrained_name)

        # Apply lightweight runtime quantization on CPU if enabled.
        self.model = quantize_model(self.model, quant_cfg, device=self.device)

        precision = model_cfg.get("precision", "float32")
        if precision == "float16" and self.device == "cpu":
            # Keep FP32 on CPU if fp16 is requested but unsupported.
            precision = "float32"
        if precision == "float16" and self.device != "cpu":
            self.model = self.model.half()

        self.model.eval()
        self.model.to(self.device)

    def _load_config(self, config_path: str | None) -> dict:
        cfg = {
            "model": {
                "pretrained_name": "titanet_large",
                "device": "cpu",
                "precision": "float32",
                "encoder": {"enabled": True, "prune_amount": 0.0},
                "quantization": {"enabled": False},
                "pruned_path": None,
                "load_pruned": False,
            }
        }
        if not config_path:
            return cfg
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            return cfg
        with cfg_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return self._merge_dicts(cfg, loaded)

    def _merge_dicts(self, base: dict, override: dict) -> dict:
        if not isinstance(base, dict):
            return override
        merged = dict(base)
        for key, value in (override or {}).items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def wav_to_embedding(self, source, target_sr=16000):
        '''
         1) Load mono @16k
         2) Get embedding from TitaNet
         3) L2-normalize
        
        Args:
            source: str or array-like, either a path to waveform file or preloaded audio
            target_sr: int, target sampling rate (default: 16000)
        Returns:
          emb: np.ndarray, shape [D]
        '''
        model = self.model

        # Accept either file paths or already-loaded wave arrays
        if isinstance(source, np.ndarray):
            wav = source.astype(np.float32, copy=False)
        elif isinstance(source, torch.Tensor):
            wav = source.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            wav, _ = librosa.load(source, sr=target_sr, mono=True)
            wav = wav.astype(np.float32, copy=False)

        wav_t = torch.tensor(wav, dtype=torch.float32, device=model.device).unsqueeze(0)  # [1, T]
        wav_len = torch.tensor([wav_t.shape[-1]], dtype=torch.int64, device=model.device)

        with torch.no_grad():
            outputs = model.forward(input_signal=wav_t, input_signal_length=wav_len)
            if isinstance(outputs, tuple):
                logits, emb = outputs
            else:
                logits, emb = outputs, None
            # Handle forward returning either tuple or tensor for embeddings
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            if emb is None:
                # some nemo versions return embeddings in logits position when decoder omits tuples
                if isinstance(logits, (list, tuple)):
                    emb = logits[0]
                else:
                    emb = logits
            emb = emb.squeeze(0).detach().cpu().numpy()

        emb = _l2norm(emb)
        return emb  # shape [D]

    def batch_wav_to_embedding(self, sources, target_sr=16000):
        embeddings = []
        for source in sources:
            emb = self.wav_to_embedding(source, target_sr=target_sr)
            embeddings.append(emb)
        return np.stack(embeddings, axis=0)  # shape [N, D]
    
    def batch_wav_to_centroid(self, sources, target_sr=16000):
        '''
        Args:
            sources: iterable of waveform sources (file paths or np arrays)
            target_sr: int, target sampling rate (default: 16000)
        Returns:
            centroid: np.ndarray, shape [D]
        '''
        embeddings = self.batch_wav_to_embedding(sources, target_sr=target_sr)  # shape [N, D]
        centroid = _centroid(embeddings)
        return centroid  # shape [D]

    def _pairwise_cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> tuple[float, float]:
        """
        Compute cosine similarity between two embeddings and return both the raw [-1, 1]
        cosine and the rescaled [0, 1] value used for gating.
        """
        device = self.device
        eps = 1e-10
        embs1 = torch.as_tensor(emb_a, dtype=torch.float32, device=device).reshape(1, -1)
        embs2 = torch.as_tensor(emb_b, dtype=torch.float32, device=device).reshape(1, -1)
        embs1 = torch.div(embs1, torch.linalg.norm(embs1, dim=1, keepdim=True) + eps)
        embs2 = torch.div(embs2, torch.linalg.norm(embs2, dim=1, keepdim=True) + eps)
        X = embs1.unsqueeze(dim=1)
        Y = embs2.unsqueeze(dim=2)
        numerator = torch.matmul(X, Y).squeeze()
        denom = torch.matmul(X, X.permute(0, 2, 1)).squeeze() * torch.matmul(
            Y.permute(0, 2, 1), Y
        ).squeeze()
        cosine = numerator / torch.sqrt(denom + eps)
        normalized = (cosine + 1.0) * 0.5
        return float(cosine.item()), float(normalized.item())
    
    def recognize(
        self,
        query_wav: str,
        target_sr: int,
        index,
        labels,
        threshold: float = 0.75,
        norm_threshold: float | None = 0.0,
        cohort_size: int | None = 20,
        reference_catalog: Mapping[str, np.ndarray] | None = None,
        pairwise_threshold: float | None = None,
    ):
        if pairwise_threshold is not None and reference_catalog is None:
            raise ValueError(
                "pairwise_threshold specified but reference_catalog is missing; provide enroll centroids to compare."
            )
        emb = self.wav_to_embedding(query_wav, target_sr)
        q = emb.astype("float32").reshape(1, -1)
        k = len(labels) if cohort_size is None else min(len(labels), max(2, cohort_size))
        sims, idxs = index.search(q, k)
        best_score = float(sims[0, 0])
        best_idx = int(idxs[0, 0])

        cohort_scores = sims[0, 1:] if sims.shape[1] > 1 else np.array([], dtype=np.float32)
        norm_score, cohort_mean, cohort_std = z_norm_score(best_score, cohort_scores)
        second_best = float(sims[0, 1]) if sims.shape[1] > 1 else None

        pairwise_similarity = None
        pairwise_similarity_01 = None
        if reference_catalog is not None:
            best_label = labels[best_idx]
            ref_vec = reference_catalog.get(best_label, None)
            if ref_vec is not None:
                pairwise_similarity, pairwise_similarity_01 = self._pairwise_cosine_similarity(
                    emb, ref_vec
                )

        passes_raw = best_score >= threshold
        passes_norm = True if norm_threshold is None else norm_score >= norm_threshold
        passes_pairwise = True
        if pairwise_threshold is not None:
            passes_pairwise = (
                pairwise_similarity_01 is not None and pairwise_similarity_01 >= pairwise_threshold
            )
        accepted = passes_raw and passes_norm and passes_pairwise
        label = labels[best_idx] if accepted else "unknown"
        confidence = (
            calibrated_confidence(
                raw_score=best_score,
                norm_score=norm_score,
                second_best=second_best,
                threshold=threshold,
                norm_threshold=norm_threshold,
                pairwise_similarity=pairwise_similarity_01,
                pairwise_threshold=pairwise_threshold,
            )
        )

        return {
            "label": label,
            "best_match": labels[best_idx],
            "raw_score": best_score,
            "norm_score": norm_score,
            "cohort_mean": cohort_mean,
            "cohort_std": cohort_std,
            "second_best_score": second_best,
            "threshold": threshold,
            "norm_threshold": norm_threshold,
            "pairwise_similarity": pairwise_similarity,
            "pairwise_similarity_01": pairwise_similarity_01,
            "pairwise_threshold": pairwise_threshold,
            "pairwise_passed": passes_pairwise,
            "confidence": confidence,
        }
    
    def get_embedding_dim(self):
        return self.model.encoder_embedding_dimq
    
    def get_model(self):
        return self.model
