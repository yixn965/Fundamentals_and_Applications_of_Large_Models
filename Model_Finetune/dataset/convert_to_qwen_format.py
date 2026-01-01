# convert_to_qwen_format.py
# 将 preprocess_mvtec_dataset.py 生成的 output_dataset.jsonl 转为 Qwen3-VL(ms-swift)可直接用于SFT的 train.jsonl
#
# 输入 (每行一个JSON):
#   - image_a_path: 参考正常图（相对或绝对路径）
#   - image_b_path: 待检测图（相对或绝对路径）
#   - label.status: "无异常" / "异常"
#   - label.changes: [{"bbox":[x,y,w,h], ...}, ...]
#
# 输出 (JSONL, 每行一个样本):
# {
#   "images": ["abs_path_ref.png", "abs_path_query.png"],
#   "messages": [
#     {"role":"user","content":"<image><image>...只输出严格JSON..."},
#     {"role":"assistant","content":"{\"status\":\"anomaly\",\"changes\":[{\"bbox\":[x1,y1,x2,y2],\"type\":\"broken_large\"}]}"}
#   ]
# }

# python convert_to_qwen_format.py \
#   --input output_dataset.jsonl \
#   --dataset_root /opt/data/private/gaoj/GaoJing/curriculum/Fundamentals_and_Applications_of_Large_Models/Model_Finetune/dataset \
#   --output_train train.jsonl


import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


def to_abs_path(p: str, dataset_root: str) -> str:
    """相对路径拼到 dataset_root；绝对路径原样返回。"""
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(dataset_root, p))


def bbox_xywh_to_xyxy(b: List[Any]) -> List[int]:
    """[x,y,w,h] -> [x1,y1,x2,y2]"""
    if not isinstance(b, list) or len(b) != 4:
        return [0, 0, 0, 0]
    x, y, w, h = b
    x1 = int(round(float(x)))
    y1 = int(round(float(y)))
    x2 = int(round(float(x) + float(w)))
    y2 = int(round(float(y) + float(h)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def infer_defect_type_from_path(image_b_path: str) -> str:
    """
    尝试从 .../test/<defect_type>/xxx.png 推断 defect_type
    失败则返回 "anomaly"
    """
    p = image_b_path.replace("\\", "/")
    parts = [x for x in p.split("/") if x]
    for i, token in enumerate(parts):
        if token == "test" and i + 1 < len(parts):
            return parts[i + 1]
    # 如果是 good 路径，也兜底一下
    for i, token in enumerate(parts):
        if token == "good":
            return "good"
    return "anomaly"


def infer_category_from_path(path_str: str) -> Optional[str]:
    """
    尝试从相对路径 dataset/<category>/... 推断 category
    你当前预处理脚本常见输出为 dataset/bottle/bottle/...
    """
    p = path_str.replace("\\", "/")
    parts = [x for x in p.split("/") if x]
    if len(parts) >= 2 and parts[0] == "dataset":
        return parts[1]
    return None


def build_prompt(category: Optional[str] = None) -> str:
    schema = '{"status":"normal|anomaly","changes":[{"bbox":[x1,y1,x2,y2],"type":"defect_type"}]}'
    cat_line = f"类别(category)：{category}。\n" if category else ""
    return (
        f"{cat_line}"
        "你是工业视觉质检员。Picture 1 为正常参考图，Picture 2 为待检测图。"
        "请判断待检测图是否存在异常，并仅输出严格JSON（不要输出任何解释/多余文本）。"
        f"JSON固定格式为：{schema}。"
        "若无异常：status=normal 且 changes=[]。"
        "若有异常：status=anomaly，changes 中每个元素包含 bbox=[x1,y1,x2,y2] 与 type=缺陷类型。"
    )


def convert(
    input_file: str,
    output_train: str,
    dataset_root: str,
    val_ratio: float = 0.0,
    output_val: Optional[str] = None,
    seed: int = 42,
) -> Tuple[int, int]:
    random.seed(seed)

    # 读入所有样本（方便可选划分val）
    samples: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] line {idx} JSON decode failed, skip: {e}")

    if not samples:
        raise RuntimeError(f"No samples loaded from {input_file}")

    # 划分 train/val（可选）
    if val_ratio and output_val:
        random.shuffle(samples)
        n_val = max(1, int(len(samples) * val_ratio))
        val_samples = samples[:n_val]
        train_samples = samples[n_val:]
    else:
        train_samples = samples
        val_samples = []

    def write_jsonl(out_path: str, data: List[Dict[str, Any]]) -> int:
        n = 0
        with open(out_path, "w", encoding="utf-8") as wf:
            for s in data:
                image_a = to_abs_path(s["image_a_path"], dataset_root)
                image_b = to_abs_path(s["image_b_path"], dataset_root)

                # 生成严格JSON答案
                status_cn = s.get("label", {}).get("status", "无异常")
                changes_raw = s.get("label", {}).get("changes", []) or []

                if status_cn == "无异常":
                    out_obj = {"status": "normal", "changes": []}
                else:
                    defect_type = infer_defect_type_from_path(s.get("image_b_path", ""))
                    out_changes = []
                    for ch in changes_raw:
                        bbox_xyxy = bbox_xywh_to_xyxy(ch.get("bbox", []))
                        out_changes.append({"bbox": bbox_xyxy, "type": defect_type})
                    out_obj = {"status": "anomaly", "changes": out_changes}

                answer = json.dumps(out_obj, ensure_ascii=False)

                category = infer_category_from_path(s.get("image_b_path", "")) or infer_category_from_path(
                    s.get("image_a_path", "")
                )
                prompt = build_prompt(category)

                record = {
                    "images": [image_a, image_b],
                    "messages": [
                        {"role": "user", "content": "<image><image>" + prompt},
                        {"role": "assistant", "content": answer},
                    ],
                }

                wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
        return n

    n_train = write_jsonl(output_train, train_samples)
    n_val = 0
    if val_samples and output_val:
        n_val = write_jsonl(output_val, val_samples)

    return n_train, n_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output_dataset.jsonl", help="input jsonl from preprocess_mvtec_dataset.py")
    parser.add_argument("--output_train", default="train.jsonl", help="output train jsonl for swift sft")
    parser.add_argument("--dataset_root", default=".", help="root dir to resolve relative image paths")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="optional val split ratio, e.g. 0.02")
    parser.add_argument("--output_val", default="val.jsonl", help="output val jsonl (used only if val_ratio>0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_val = args.output_val if args.val_ratio > 0 else None
    n_train, n_val = convert(
        input_file=args.input,
        output_train=args.output_train,
        dataset_root=args.dataset_root,
        val_ratio=args.val_ratio,
        output_val=output_val,
        seed=args.seed,
    )
    print(f"✅ done. train: {n_train}" + (f", val: {n_val}" if n_val else ""))
    print(f"train file: {os.path.abspath(args.output_train)}")
    if n_val:
        print(f"val file:   {os.path.abspath(args.output_val)}")


if __name__ == "__main__":
    main()
