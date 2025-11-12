#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python run_gsam.py --image ./demo_data/rgb.png --text "a red cola can" --output_dir outputs
import os
import sys
import json
import argparse

import cv2
import torch
import numpy as np
from PIL import Image as Img
import matplotlib.pyplot as plt

# === 本地仓库依赖 ===
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import sam_model_registry, SamPredictor


def load_image_for_dino(image_rgb: np.ndarray):
    """为 GroundingDINO 做预处理（输入 RGB np.uint8）"""
    image_pil = Img.fromarray(image_rgb).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=2000),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image_tensor


def load_gdino(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    ckpt = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    print("[GroundingDINO] load result:", load_res)
    model.eval()
    return model.to(device)


@torch.no_grad()
def grounding_infer(model, image_tensor, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0].cpu()  # (nq, 256)
    boxes = outputs["pred_boxes"][0].cpu()              # (nq, 4)

    # 过滤
    keep = logits.max(dim=1)[0] > box_threshold
    logits_f = logits[keep]
    boxes_f = boxes[keep]

    # 文本短语
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases = []
    for logit in logits_f:
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        phrases.append(phrase + f"({str(logit.max().item())[:4]})")
    return boxes_f, phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_artifacts(output_dir, masks, boxes_abs, phrases):
    os.makedirs(output_dir, exist_ok=True)

    # 语义着色图
    value = 0
    mask_img = torch.zeros(masks.shape[-2:])
    for idx, m in enumerate(masks):
        mask_img[m.cpu().numpy()[0] == True] = value + idx + 1

    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    # 二值 PNG（0/255）
    Img.fromarray((mask_img.numpy() > 0).astype(np.uint8) * 255).save(os.path.join(output_dir, 'mask.png'))

    # 元数据
    meta = [{'value': 0, 'label': 'background'}]
    v = 0
    for label, box in zip(phrases, boxes_abs):
        v += 1
        name, logit = label.split('(')
        logit = logit[:-1]
        meta.append({
            'value': v,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def grounded_sam_infer(
    image_rgb: np.ndarray,
    text_prompt: str,
    output_dir: str = "model_outputs",
    config_file: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounded_checkpoint: str = "groundingdino_swint_ogc.pth",
    sam_version: str = "vit_h",
    sam_checkpoint: str = "sam_vit_h_4b8939.pth",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    os.makedirs(output_dir, exist_ok=True)

    # 1) DINO 预处理与推理
    image_pil, image_tensor = load_image_for_dino(image_rgb)
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    gdino = load_gdino(config_file, grounded_checkpoint, device)
    boxes_filt, phrases = grounding_infer(
        gdino, image_tensor, text_prompt, box_threshold, text_threshold, device=device
    )

    if boxes_filt.numel() == 0:
        print("[Warn] 没有检测到满足阈值的框，输出空 mask。")
        h, w = image_rgb.shape[:2]
        empty = Img.fromarray(np.zeros((h, w), dtype=np.uint8))
        empty.save(os.path.join(output_dir, "mask.png"))
        with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
            json.dump([{'value': 0, 'label': 'background'}], f, indent=2)
        return

    # 2) SAM 分割
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)  # RGB

    # 将中心点宽高盒转为绝对左上右下
    W, H = image_pil.size
    boxes_abs = boxes_filt.clone()
    for i in range(boxes_abs.size(0)):
        boxes_abs[i] = boxes_abs[i] * torch.Tensor([W, H, W, H])
        boxes_abs[i][:2] -= boxes_abs[i][2:] / 2
        boxes_abs[i][2:] += boxes_abs[i][:2]

    boxes_abs_cpu = boxes_abs.cpu()
    transformed = predictor.transform.apply_boxes_torch(boxes_abs_cpu, image_rgb.shape[:2]).to(device)

    with torch.no_grad():
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )

    # 3) 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    for m in masks:
        show_mask(m.cpu().numpy(), plt.gca(), random_color=True)
    for b, lab in zip(boxes_abs_cpu, phrases):
        show_box(b.numpy(), plt.gca(), lab)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "vis.png"), bbox_inches="tight", dpi=300, pad_inches=0.0)

    # 4) 保存产物
    save_mask_artifacts(output_dir, masks, boxes_abs_cpu, phrases)

    # 释放
    del predictor, sam, gdino
    torch.cuda.empty_cache()


def parse_args():
    ap = argparse.ArgumentParser(description="Grounded-SAM: single image -> mask")
    ap.add_argument("--image", type=str, required=True, help="path to an RGB image")
    ap.add_argument("--text", type=str, required=True, help="text prompt, e.g., 'a yellow mustard bottle'")
    ap.add_argument("--output_dir", type=str, default="model_outputs", help="output directory")
    # 可选：阈值与权重/配置路径
    ap.add_argument("--box_threshold", type=float, default=0.3)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--config_file", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--grounded_checkpoint", type=str, default="groundingdino_swint_ogc.pth")
    ap.add_argument("--sam_version", type=str, default="vit_h")
    ap.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # OpenCV 读图是 BGR，这里转 RGB
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    grounded_sam_infer(
        image_rgb=img_rgb,
        text_prompt=args.text,
        output_dir=args.output_dir,
        config_file=args.config_file,
        grounded_checkpoint=args.grounded_checkpoint,
        sam_version=args.sam_version,
        sam_checkpoint=args.sam_checkpoint,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    print(f"[Done] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
