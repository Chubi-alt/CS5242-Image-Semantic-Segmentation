from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import SegFormerWithAuxHeads, load_full_model_state

try:  # pragma: no cover
    from transformers import Trainer, TrainingArguments
except Exception:  # pragma: no cover
    Trainer = object  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]


class SegFormerTrainer(Trainer):  # type: ignore[misc]
    def __init__(
        self,
        loss_fn,
        aux_weight: float = 0.4,
        teacher_model: Optional[nn.Module] = None,
        kd_loss_fn=None,
        *args,
        **kwargs,
    ):
        self.loss_fn = loss_fn
        self.aux_weight = aux_weight
        self.teacher_model = teacher_model
        self.kd_loss_fn = kd_loss_fn
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        del kwargs
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        main_loss = self.loss_fn(logits, labels)
        aux_logits = getattr(outputs, "aux_logits", None)
        if aux_logits:
            aux_losses = []
            for aux_logit in aux_logits:
                resized_aux = aux_logit
                if resized_aux.shape[-2:] != labels.shape[-2:]:
                    resized_aux = F.interpolate(resized_aux, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                aux_losses.append(self.loss_fn(resized_aux, labels))
            aux_loss = torch.stack(aux_losses).mean()
            loss = main_loss + float(self.aux_weight) * aux_loss
        else:
            loss = main_loss
        # Knowledge distillation — only during training, skipped at eval/test time
        if (
            self.teacher_model is not None
            and self.kd_loss_fn is not None
            and model.training
        ):
            with torch.no_grad():
                teacher_out = self.teacher_model(
                    **model_inputs, output_hidden_states=True
                )
                teacher_logits = (
                    teacher_out.logits
                    if hasattr(teacher_out, "logits")
                    else teacher_out["logits"]
                )
                if teacher_logits.shape[-2:] != labels.shape[-2:]:
                    teacher_logits = F.interpolate(
                        teacher_logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
            kd_loss = self.kd_loss_fn(
                student_logits=logits,
                teacher_logits=teacher_logits,
                labels=labels,
                student_outputs=outputs,
                teacher_outputs=teacher_out,
            )
            loss = loss + kd_loss
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        # Fully overridden — parent's prediction_step does outputs[1:] which
        # fails on SegFormerModelOutput dataclass.  We reuse self.compute_loss
        # (which already handles both plain and aux-head models) and extract
        # logits from the returned outputs object directly.
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]
        with torch.no_grad():
            _ctx = getattr(self, "compute_loss_context_manager", None)
            if _ctx is not None:
                with _ctx():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        if prediction_loss_only:
            return (loss, None, None)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        return (loss, logits, labels)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        save_dir = Path(output_dir or self.args.output_dir)
        super()._save(str(save_dir), state_dict=state_dict)
        from core.distillation import SegFormerWithAdaptors
        model = self.model_wrapped if getattr(self, "model_wrapped", None) is not None else self.model
        if isinstance(model, (SegFormerWithAuxHeads, SegFormerWithAdaptors)):
            torch.save(model.state_dict(), save_dir / "full_model.pt")

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        try:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        except Exception:
            from core.distillation import SegFormerWithAdaptors
            target_model = model or self.model
            full_model_path = Path(resume_from_checkpoint) / "full_model.pt"
            if isinstance(target_model, (SegFormerWithAuxHeads, SegFormerWithAdaptors)) and full_model_path.exists():
                load_full_model_state(target_model, full_model_path)
                return
            raise


def build_training_args(config, run_context, report_to: str):
    if TrainingArguments is None:
        raise RuntimeError("transformers is required to build TrainingArguments.")

    kwargs = {
        "output_dir": str(Path(run_context.checkpoint_dir)),
        "logging_dir": str(Path(run_context.run_dir) / "logs"),
        "run_name": run_context.run_id,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "save_total_limit": config.save_total_limit,
        "logging_steps": config.logging_steps,
        "load_best_model_at_end": config.load_best_model_at_end,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "dataloader_num_workers": config.dataloader_num_workers,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "report_to": report_to,
        "bf16": bool(config.bf16 and torch.cuda.is_available()),
        "fp16": bool(getattr(config, "fp16", False) and torch.cuda.is_available()),
        "push_to_hub": False,
        "remove_unused_columns": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = config.eval_strategy
    else:
        kwargs["evaluation_strategy"] = config.eval_strategy
    if "label_names" in signature.parameters:
        kwargs["label_names"] = ["labels"]
    kwargs["save_strategy"] = config.save_strategy
    if config.eval_strategy == "steps":
        kwargs["eval_steps"] = config.eval_steps
    if config.save_strategy == "steps":
        kwargs["save_steps"] = config.save_steps
    return TrainingArguments(**kwargs)
