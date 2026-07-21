"""
SAM3 Image Inference

이미지 한 장을 텍스트 프롬프트로 SAM3 추론하고, 결과(마스크/바운딩박스/점수)를 저장함.

실행 예시:
    python day2/day2_4.0.sam3_inference.py \
        --input data/sam3_practice/images/truck.jpg \
        --prompt "window" \
        --output-dir data/sam3_practice_outputs

저장 결과 (--output-dir 하위):
    <stem>_<prompt>_result.png   : 마스크 + bbox 오버레이 시각화
    <stem>_<prompt>_preds.npz    : masks / boxes / scores 원본 배열
    <stem>_<prompt>_meta.json    : 입력/프롬프트/검출 개수 등 메타 정보
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# utils 패키지 접근을 위한 경로 추가 (day2/ 상위 디렉토리)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from day2.utils.vision import get_random_color


# ============================================================================
# 1. 명령행 인자 설정
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 image text-prompt inference")
    parser.add_argument("--input", type=str, default='data/sam3_practice/images/truck.jpg', help="Image path")
    parser.add_argument("--prompt", type=str, default="window", help="Text prompt, preferably in English")
    parser.add_argument("--output-dir", type=str, default="data/sam3_practice_outputs", help="Directory to save results")
    parser.add_argument("--checkpoint", type=str, default="data/checkpoint/sam3/sam3.1_multiplex.pt", help="SAM3 image model checkpoint")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Hide detections below this score")
    parser.add_argument("--alpha", type=float, default=0.3, help="Mask overlay opacity")
    return parser.parse_args()


def autocast_context():
    """CUDA가 있으면 bfloat16 autocast, 없으면 아무것도 하지 않는 컨텍스트."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ============================================================================
# 2. 결과 시각화
# ============================================================================
def draw_predictions(image_rgb, masks, boxes, scores, prompt, score_threshold=0.0, alpha=0.3):
    """마스크와 bbox를 원본 이미지 위에 오버레이한 RGB 이미지를 반환한다.

    가장 점수가 높은 인스턴스는 파란색(BGR 기준 (255, 0, 0))으로 강조한다.
    """
    vis_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    mask_overlay = vis_img.copy()

    # 임계값 이상인 인스턴스만 시각화
    visible = np.flatnonzero(scores >= score_threshold)
    if len(visible) == 0:
        return image_rgb.copy()

    best_idx = int(visible[np.argmax(scores[visible])])
    for i in visible:
        i = int(i)
        color = (255, 0, 0) if i == best_idx else get_random_color()

        # 마스크 오버레이 (masks: (N, 1, H, W))
        mask = masks[i, 0] > 0.5
        mask_overlay[mask] = color

        # bbox + 점수 텍스트
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3 if i == best_idx else 1)
        cv2.putText(vis_img, f"{prompt}: {scores[i]:.2f}", (x1, max(y1 - 10, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    result_bgr = cv2.addWeighted(mask_overlay, alpha, vis_img, 1.0 - alpha, 0)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


# ============================================================================
# 3. 메인 함수
# ============================================================================
def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- 모델 로드 --
    print("[INFO] Loading SAM3 image model...")
    sam3_model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
    )
    processor = Sam3Processor(sam3_model)

    # -- 이미지 로드 --
    image_pil = Image.open(input_path).convert("RGB")
    image_rgb = np.array(image_pil)

    # -- 추론: 이미지 등록 → 텍스트 프롬프트로 검출 --
    print(f"[INFO] Running SAM3 inference on {input_path} with prompt '{args.prompt}'...")
    with torch.inference_mode(), autocast_context():
        inference_state = processor.set_image(image_pil)
        output = processor.set_text_prompt(state=inference_state, prompt=args.prompt)

    # 검출 결과 취득
    masks = output["masks"].float().cpu().numpy()    # (N, 1, H, W)
    boxes = output["boxes"].float().cpu().numpy()    # (N, 4) xyxy
    scores = output["scores"].float().cpu().numpy()  # (N,)
    print(f"[INFO] Found {len(scores)} object(s) for prompt '{args.prompt}'.")

    if len(scores) == 0:
        print("[WARN] No object detected. 다른 텍스트 프롬프트로 다시 시도해 보세요.")
        return

    best_idx = int(np.argmax(scores))
    print(f"[INFO] Best detection: score={scores[best_idx]:.3f}, bbox={boxes[best_idx]}")

    # -- 결과 저장 --
    stem = f"{input_path.stem}_{args.prompt.replace(' ', '_')}"
    result_path = output_dir / f"{stem}_result.png"
    npz_path = output_dir / f"{stem}_preds.npz"
    meta_path = output_dir / f"{stem}_meta.json"

    result_rgb = draw_predictions(image_rgb, masks, boxes, scores,
                                  args.prompt, args.score_threshold, args.alpha)
    cv2.imwrite(str(result_path), cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
    np.savez_compressed(npz_path, masks=masks, boxes=boxes, scores=scores)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": str(input_path),
                "prompt": args.prompt,
                "num_detections": int(len(scores)),
                "score_threshold": args.score_threshold,
                "best_index": best_idx,
                "best_score": float(scores[best_idx]),
                "best_box": boxes[best_idx].tolist(),
                "result_image": str(result_path),
                "prediction_npz": str(npz_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[INFO] Saved visualization → {result_path}")
    print(f"[INFO] Saved raw predictions → {npz_path}")
    print(f"[INFO] Saved metadata → {meta_path}")


if __name__ == "__main__":
    main()
