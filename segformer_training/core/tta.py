from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.constants import IGNORE_INDEX
from core.metrics import compute_full_metrics


def tta_predict(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    config,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    original = image.convert("RGB")
    original_width, original_height = original.size
    logit_sum = None
    augmentations = 0

    flips = [False, True] if config.flip else [False]
    with torch.no_grad():
        for scale in config.scales:
            scaled_width = max(1, int(round(original_width * float(scale))))
            scaled_height = max(1, int(round(original_height * float(scale))))
            scaled_image = original.resize((scaled_width, scaled_height), Image.BILINEAR)
            for do_flip in flips:
                augmented = scaled_image.transpose(Image.FLIP_LEFT_RIGHT) if do_flip else scaled_image
                inputs = processor(images=augmented, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                logits = F.interpolate(logits, size=(original_height, original_width), mode="bilinear", align_corners=False)
                if do_flip:
                    logits = torch.flip(logits, dims=[3])
                logit_sum = logits if logit_sum is None else logit_sum + logits
                augmentations += 1
    averaged = logit_sum / float(max(augmentations, 1))
    prediction = averaged.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return prediction


def tta_evaluate(
    model: torch.nn.Module,
    processor,
    test_dataset,
    config,
    device: torch.device,
    step_callback=None,
    num_labels: int = 11,
    ignore_index: int = IGNORE_INDEX,
    compute_confusion_matrix: bool = True,
    prediction_limit: int = 0,
) -> Tuple[Dict[str, Any], List[Tuple[Image.Image, np.ndarray, np.ndarray]]]:
    """Evaluate with TTA in a single pass, optionally collecting prediction triplets.

    Args:
        prediction_limit: 0 = no triplets; -1 = all; N = first N images.

    Returns:
        (metrics_dict, triplets) where each triplet is (PIL.Image, label_array, prediction_array).
    """
    predictions = []
    references = []
    triplets: List[Tuple[Image.Image, np.ndarray, np.ndarray]] = []
    total = len(test_dataset)
    collect_up_to = total if prediction_limit == -1 else prediction_limit

    for index in range(total):
        raw_sample = test_dataset.get_raw_sample(index)
        prediction = tta_predict(
            model=model,
            processor=processor,
            image=raw_sample.image,
            config=config,
            device=device,
        )
        predictions.append(prediction)
        references.append(np.asarray(raw_sample.label, dtype=np.int64))
        if index < collect_up_to:
            triplets.append((raw_sample.image, raw_sample.label, prediction))
        if step_callback is not None:
            step_callback(index + 1, total)

    metrics = compute_full_metrics(
        predictions=np.stack(predictions, axis=0),
        references=np.stack(references, axis=0),
        num_labels=num_labels,
        ignore_index=ignore_index,
        compute_confusion_matrix=compute_confusion_matrix,
    )
    return metrics, triplets
