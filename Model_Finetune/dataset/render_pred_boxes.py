#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import html
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_color(s: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if not s:
        return default
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Color must be 'r,g,b', got: {s}")
    r, g, b = [int(float(x)) for x in parts]
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def load_font(font_path: str, font_size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def apply_path_replace(p: str, replaces: List[Tuple[str, str]]) -> str:
    if not p:
        return p
    for src, dst in replaces:
        if src and (p.startswith(src) or src in p):
            p = p.replace(src, dst, 1)
    return p


def extract_image_paths(obj: Dict[str, Any]) -> List[str]:
    """
    针对你的评估集：obj["images"] 是一个 list，元素可能是 {"path": "..."} 或直接是 str
    """
    paths: List[str] = []

    imgs = obj.get("images")
    if isinstance(imgs, list) and imgs:
        for it in imgs:
            if isinstance(it, str):
                paths.append(it)
            elif isinstance(it, dict):
                p = it.get("path") or it.get("image")
                if isinstance(p, str):
                    paths.append(p)
        if paths:
            return paths

    # 兜底：有些数据集可能用 image/images 字段
    for k in ["image", "img", "image_path", "path"]:
        if isinstance(obj.get(k), str):
            return [obj[k]]

    for k in ["images", "imgs"]:
        v = obj.get(k)
        if isinstance(v, list) and v:
            if isinstance(v[0], str):
                return [v[0]]
            if isinstance(v[0], dict):
                p = v[0].get("path") or v[0].get("image")
                if isinstance(p, str):
                    return [p]

    return []


def extract_gt(obj: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    取最后一条 assistant 的 content 作为 GT json 字符串：
    {"status": "...", "changes":[{"bbox":[...], "description":"..."}]}
    """
    gt_text = ""
    msgs = obj.get("messages", [])
    if isinstance(msgs, list) and msgs:
        last = msgs[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            gt_text = last["content"].strip()

    if not gt_text:
        return "", [], ""

    try:
        gt = json.loads(gt_text)
        status = str(gt.get("status", ""))
        changes = gt.get("changes", [])
        if not isinstance(changes, list):
            changes = []
        return status, changes, gt_text
    except Exception:
        # 如果偶尔不是严格 JSON，就降级：当作无 bbox
        return "", [], gt_text


def scale_boxes_auto(boxes: List[List[float]], w: int, h: int) -> List[List[float]]:
    """
    自动判断 bbox 是 0~1 / 0~1000 / 像素
    """
    if not boxes:
        return boxes
    mx = max(max(b) for b in boxes)

    if mx <= 1.5:  # 0~1
        return [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in boxes]
    if mx <= 1100:  # 0~1000
        return [[b[0] * w / 1000.0, b[1] * h / 1000.0, b[2] * w / 1000.0, b[3] * h / 1000.0] for b in boxes]
    return boxes  # pixel


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


def draw_boxes_on_image(
    img: Image.Image,
    changes: List[Dict[str, Any]],
    coord: str,
    line_width: int,
    box_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
    text_bg: Optional[Tuple[int, int, int]],
    font: ImageFont.ImageFont,
):
    """
    在单张图上画 changes 里的 bbox + description
    coord: auto / 01 / 1000 / pixel
    """
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # 收集 bbox
    raw_boxes = []
    descs = []
    for ch in changes:
        if not isinstance(ch, dict):
            continue
        bbox = ch.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                b = [float(x) for x in bbox]
                raw_boxes.append(b)
                descs.append(str(ch.get("description", "")))
            except Exception:
                continue

    if not raw_boxes:
        return img

    if coord == "01":
        boxes = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in raw_boxes]
    elif coord == "1000":
        boxes = [[b[0] * w / 1000.0, b[1] * h / 1000.0, b[2] * w / 1000.0, b[3] * h / 1000.0] for b in raw_boxes]
    elif coord == "pixel":
        boxes = raw_boxes
    else:
        boxes = scale_boxes_auto(raw_boxes, w, h)

    for i, (b, desc) in enumerate(zip(boxes, descs)):
        x1, y1, x2, y2 = b
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        label = desc if desc else f"change_{i}"
        tx, ty = x1, max(0, y1 - 18)
        if text_bg is not None:
            bb = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle(bb, fill=text_bg)
        draw.text((tx, ty), label, fill=text_color, font=font)

    return img


def concat_images(img1: Image.Image, img2: Image.Image, layout: str, gap: int, bg: Tuple[int, int, int]) -> Image.Image:
    if layout == "v":
        w = max(img1.size[0], img2.size[0])
        h = img1.size[1] + gap + img2.size[1]
        canvas = Image.new("RGB", (w, h), bg)
        canvas.paste(img1, (0, 0))
        canvas.paste(img2, (0, img1.size[1] + gap))
        return canvas
    else:
        w = img1.size[0] + gap + img2.size[0]
        h = max(img1.size[1], img2.size[1])
        canvas = Image.new("RGB", (w, h), bg)
        canvas.paste(img1, (0, 0))
        canvas.paste(img2, (img1.size[0] + gap, 0))
        return canvas


def add_header_text(canvas: Image.Image, text: str, font: ImageFont.ImageFont, text_color: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]):
    if not text:
        return
    draw = ImageDraw.Draw(canvas)
    x, y = 8, 8
    if bg is not None:
        bb = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bb, fill=bg)
    draw.text((x, y), text, fill=text_color, font=font)


def make_html(out_dir: str, items: List[Dict[str, Any]], html_max_width: int, show_gt_json: bool):
    lines = [
        "<html><head><meta charset='utf-8'></head><body>",
        "<h2>Eval Dataset Visualizations</h2>",
    ]
    for it in items:
        img_name = it["img_name"]
        status = it.get("status", "")
        gt_text = it.get("gt_text", "")

        lines.append("<div style='margin:14px 0;'>")
        lines.append(f"<img src='{html.escape(img_name)}' style='max-width:{int(html_max_width)}px;'>")
        lines.append(f"<p><b>{html.escape(img_name)}</b> | status: <b>{html.escape(status)}</b></p>")
        if show_gt_json:
            lines.append("<details><summary>GT JSON</summary>")
            lines.append(f"<pre style='white-space:pre-wrap;max-width:{int(html_max_width)}px;'>{html.escape(gt_text)}</pre>")
            lines.append("</details>")
        lines.append("</div><hr/>")

    lines.append("</body></html>")
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_jsonl", required=True, help="评估集 jsonl（每行一个样本）")
    ap.add_argument("--out_dir", default="vis_eval", help="输出目录")
    ap.add_argument("--limit", type=int, default=200, help="最多处理多少行；<=0 表示不限制")
    ap.add_argument("--start", type=int, default=0, help="从第几行开始（0-based）")

    # 图片路径重映射（很常见：数据里是别的机器的绝对路径）
    ap.add_argument(
        "--path_replace",
        action="append",
        default=[],
        help="路径替换规则，形如 SRC::DST，可重复多次（按顺序应用）",
    )
    ap.add_argument("--skip_missing", action="store_true", help="图片不存在就跳过（默认建议开）")

    # 拼图与画框位置
    ap.add_argument("--layout", choices=["h", "v"], default="h", help="两张图拼接方向：h=横向，v=纵向")
    ap.add_argument("--gap", type=int, default=10, help="两张图之间间隔像素")
    ap.add_argument("--draw_on", choices=["first", "second", "both"], default="second",
                    help="bbox 画在哪张图上：first/second/both（默认 second）")

    # bbox 坐标系
    ap.add_argument("--coord", choices=["auto", "01", "1000", "pixel"], default="auto",
                    help="bbox 坐标系（默认 auto 自动判断）")

    # 样式
    ap.add_argument("--line_width", type=int, default=3)
    ap.add_argument("--box_color", default="255,0,0", help="框颜色 r,g,b")
    ap.add_argument("--text_color", default="0,0,0", help="文字颜色 r,g,b")
    ap.add_argument("--text_bg", default="255,255,255", help="文字背景颜色 r,g,b；留空则不画背景")
    ap.add_argument("--bg_color", default="255,255,255", help="拼图画布背景 r,g,b")
    ap.add_argument("--font_path", default="", help="字体路径（可选）")
    ap.add_argument("--font_size", type=int, default=16)

    # HTML
    ap.add_argument("--html_max_width", type=int, default=1000)
    ap.add_argument("--html_show_gt_json", action="store_true", help="HTML里展开显示GT JSON")

    args = ap.parse_args()

    replaces: List[Tuple[str, str]] = []
    for rule in args.path_replace:
        if "::" in rule:
            src, dst = rule.split("::", 1)
            replaces.append((src, dst))

    box_color = parse_color(args.box_color, (255, 0, 0))
    text_color = parse_color(args.text_color, (0, 0, 0))
    bg_color = parse_color(args.bg_color, (255, 255, 255))
    text_bg = parse_color(args.text_bg, (255, 255, 255)) if args.text_bg.strip() else None
    font = load_font(args.font_path, args.font_size)

    os.makedirs(args.out_dir, exist_ok=True)

    html_items: List[Dict[str, Any]] = []
    processed = 0

    with open(args.dataset_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < args.start:
                continue
            if args.limit > 0 and processed >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            img_paths = extract_image_paths(obj)
            if len(img_paths) < 2:
                # 不够两张就跳过（你这个任务是对比两张图）
                continue

            img1_path = apply_path_replace(img_paths[0], replaces)
            img2_path = apply_path_replace(img_paths[1], replaces)

            if args.skip_missing:
                if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
                    continue

            status, changes, gt_text = extract_gt(obj)

            try:
                img1 = Image.open(img1_path).convert("RGB")
                img2 = Image.open(img2_path).convert("RGB")
            except Exception:
                continue

            # 按 draw_on 画框
            if status != "无异常" and changes:
                if args.draw_on in ("first", "both"):
                    img1 = draw_boxes_on_image(img1, changes, args.coord, args.line_width, box_color, text_color, text_bg, font)
                if args.draw_on in ("second", "both"):
                    img2 = draw_boxes_on_image(img2, changes, args.coord, args.line_width, box_color, text_color, text_bg, font)

            canvas = concat_images(img1, img2, args.layout, args.gap, bg_color)

            header = f"idx={idx} | status={status}"
            add_header_text(canvas, header, font, text_color, text_bg)

            out_name = f"{idx:06d}.jpg"
            out_path = os.path.join(args.out_dir, out_name)
            canvas.save(out_path)

            html_items.append({"img_name": out_name, "status": status, "gt_text": gt_text})
            processed += 1

    make_html(args.out_dir, html_items, args.html_max_width, args.html_show_gt_json)

    print(f"Done. Open: {os.path.join(args.out_dir, 'index.html')}")
    print(f"Saved images: {processed}")


if __name__ == "__main__":
    main()
