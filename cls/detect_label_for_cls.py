#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build dataset crops (visible & partially_visible) from COCO-like JSON + MP4 pairs.

Output structure:
  <out_root>/
    visible/
      train/0 .. train/11
      val/0 .. val/11
    partial/
      train/0 .. train/11
      val/0 .. val/11

Usage example:
  python build_dataset.py --root data_root --out out_dir --player-id 4 --train-ratio 0.8 --dbg 10 --include-invisible-as-zero
"""
import json, cv2, re, random, argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, List, Dict

# ============== Utils ==============
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def clamp(v, lo, hi): return max(lo, min(hi, v))

def get_attr(ann: dict, key: str, *aliases):
    if key in ann and ann[key] not in (None, ""): return ann[key]
    attrs = ann.get("attributes") or {}
    if key in attrs and attrs[key] not in (None, ""): return attrs[key]
    for k in aliases:
        if k in ann and ann[k] not in (None, ""): return ann[k]
        if k in attrs and attrs[k] not in (None, ""): return attrs[k]
    return None

def normalize_visibility(val) -> str:
    """Return 'visible', 'partially_visible', or 'not_visible'."""
    if val is None: return "not_visible"
    if isinstance(val, bool): return "visible" if val else "not_visible"
    s = str(val).strip().lower()
    if s=="" or s in {"0","false","no","none","n/a","na","null","unreadable","unknown","invisible","not_visible"}:
        return "not_visible"
    if "part" in s:
        return "partially_visible"
    if "vis" in s or s in {"1","true","yes","visible"}:
        return "visible"
    return "not_visible"

def jersey_group(jersey_number, number_visible) -> int:
    """Map jersey -> class id: 0,1..10,11."""
    nv = str(number_visible or "").strip().lower()
    invis = {"not_visible","none","invisible","unreadable","unknown","na","n/a","null","false","0"}
    if nv in invis: return 0
    if jersey_number is None: return 0
    s = str(jersey_number).strip()
    m = re.search(r"\d+", s)
    if not m: return 0
    try: n = int(m.group(0))
    except: return 0
    if 1 <= n <= 10: return n
    if n >= 11: return 11
    return 0

def crop_bbox_with_vertical_trim(img, x, y, w, h, top_trim=0.05, bot_trim=0.20):
    H, W = img.shape[:2]
    x0, y0 = max(0,int(round(x))), max(0,int(round(y)))
    x1, y1 = min(W,int(round(x+w))), min(H,int(round(y+h)))
    if x1<=x0 or y1<=y0: return img[0:0,0:0]
    bh = y1-y0
    y0t = y0 + int(round(bh*top_trim))
    y1t = y1 - int(round(bh*bot_trim))
    if y1t<=y0t: return img[0:0,0:0]
    return img[y0t:y1t, x0:x1]

def extract_frame_index_from_filename(fn: str) -> Optional[int]:
    if not fn: return None
    # try patterns (case-insensitive)
    patterns = [
        r"frame[_\-]?0*?(\d+)",
        r"(\d{6,})",
        r"(\d{4,})",
        r"(\d+)"
    ]
    for p in patterns:
        m = re.search(p, fn, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except:
                continue
    return None

def build_name2idx_mapping(img_list: List[dict], dbg: int=0) -> Dict[str,int]:
    name2idx={}
    found_any=False
    for im in img_list:
        fn = im.get("file_name","")
        idx = extract_frame_index_from_filename(fn)
        if idx is not None:
            name2idx[fn]=idx
            found_any=True
    if not found_any:
        if dbg:
            print("[WARN] Không tìm thấy index trong file_name bằng pattern. Dùng fallback enumerate(sorted filenames).")
        sorted_names = sorted([im.get("file_name","") for im in img_list])
        for i,fn in enumerate(sorted_names):
            name2idx[fn]=i
    return name2idx

# ============== Core ==============

def collect_split_clips(root: Path, train_ratio: float, seed: int):
    clips = [p for p in root.iterdir() if p.is_dir()]
    clips.sort()
    random.seed(seed)
    k = max(0, min(len(clips), int(round(len(clips)*train_ratio))))
    train_set = set([p.name for p in random.sample(clips, k)]) if k>0 else set()
    return clips, train_set

def open_pair(clip_dir: Path):
    jsons = list(clip_dir.glob("*.json"))
    mp4s  = list(clip_dir.glob("*.mp4"))
    if not jsons or not mp4s: return None, None
    for j in jsons:
        for v in mp4s:
            if j.stem.replace("_subclip","")==v.stem.replace("_subclip",""):
                return j, v
    return jsons[0], mp4s[0]

def build_one_visibility(clip: Path, split: str, out_split_root: Path,
                         target_vis: str, include_invi_as_zero: bool,
                         pid: int, top_trim: float, bot_trim: float,
                         min_side: int, dbg: int) -> Counter:
    jpath, vpath = open_pair(clip)
    if not jpath or not vpath:
        if dbg: print(f"[WARN] {clip.name}: thiếu json/mp4");
        return Counter()

    data = json.loads(jpath.read_text(encoding="utf-8"))
    # verify categories - optional check
    cat2id = {c.get("name"): c.get("id") for c in data.get("categories", [])}

    img_by_id = {im["id"]: im for im in data.get("images", [])}
    anns_by_img = defaultdict(list)

    tv = target_vis.strip().lower()

    # build mapping file_name -> frame index robustly
    name2idx = build_name2idx_mapping(list(img_by_id.values()), dbg)

    # collect annotations with flexible visibility matching
    for a in data.get("annotations", []):
        if a.get("category_id") != pid:
            continue
        if not a.get("bbox"):
            continue
        # get raw visibility from attributes or top-level
        raw_vis = get_attr(a, "number_visible", "visibility", "visible")
        nv = normalize_visibility(raw_vis)
        add = False
        if tv == "visible":
            if nv == "visible":
                add = True
            elif include_invi_as_zero and nv == "not_visible":
                add = True
        elif tv == "partially_visible":
            if nv == "partially_visible":
                add = True
        if add:
            anns_by_img[a["image_id"]].append((a, nv))

    if not any(anns_by_img.values()):
        if dbg: print(f"[INFO] {clip.name}:{split}:{tv} - không tìm thấy annotation phù hợp.")
        return Counter()

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"[ERR] Không mở được video: {vpath}"); return Counter()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ensure_dir(out_split_root)
    counts = Counter()
    dbg_left = dbg

    for im_id, im in img_by_id.items():
        fn = im.get("file_name")
        if fn not in name2idx:
            if dbg: print(f"DBG skip: {clip.name} file_name '{fn}' không có frame index mapping.")
            continue
        frame_idx = name2idx[fn]
        # try safe frame index within bounds (try frame_idx, then frame_idx-1)
        tried = [frame_idx]
        if frame_count>0 and frame_idx >= frame_count and frame_idx>0:
            tried.append(frame_idx-1)
        frame_read = False
        for t in tried:
            if t < 0: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, frame = cap.read()
            if ok and frame is not None:
                frame_read = True
                break
        if not frame_read:
            if dbg: print(f"DBG không đọc được frame cho {fn} (tried {tried}, frame_count={frame_count}).")
            continue

        for ann, nv in anns_by_img.get(im_id, []):
            bbox = ann.get("bbox")
            if not bbox or len(bbox)!=4: continue
            x,y,w,h = bbox
            # ensure ints
            x,y,w,h = int(round(x)), int(round(y)), int(round(w)), int(round(h))

            jersey = get_attr(ann, "jersey_number", "jersey_num", "shirt_number", "number")
            cls_id = jersey_group(jersey, nv)

            crop = crop_bbox_with_vertical_trim(frame, x, y, w, h, top_trim=top_trim, bot_trim=bot_trim)
            if crop.size == 0:
                if dbg: print(f"DBG crop empty for ann {ann.get('id')} in {fn}")
                continue
            ch, cw = crop.shape[:2]
            if min(ch, cw) < min_side:
                if dbg: print(f"DBG crop too small {ch}x{cw} < min_side({min_side}) for {fn} ann {ann.get('id')}")
                continue

            out_dir = out_split_root / str(cls_id)
            ensure_dir(out_dir)
            ann_id = ann.get("id", 0)
            out_name = f"{clip.name}_{fn.rsplit('.',1)[0]}_{ann_id}.png"
            cv2.imwrite(str(out_dir / out_name), crop)
            counts[cls_id] += 1

            if dbg_left > 0:
                raw_vis = get_attr(ann, "number_visible", "visibility", "visible")
                print(f"DBG [{clip.name}:{split}:{tv}] {fn} vis_norm={nv} raw_vis={raw_vis} jersey={jersey} -> cls={cls_id} size=({ch}x{cw})")
                dbg_left -= 1

    cap.release()
    return counts

def main():
    ap = argparse.ArgumentParser("Build visible & partial datasets with vertical trim")
    ap.add_argument("--root", type=Path, default=Path("data_root"), help="Root chứa các folder clip (mỗi folder: .mp4 + .json)")
    ap.add_argument("--out", type=Path, default=Path("out_dataset"), help="Output root")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--player-name", type=str, default="player")
    ap.add_argument("--player-id", type=int, default=None, help="Dùng category_id trực tiếp (ưu tiên nếu có)")
    ap.add_argument("--top-trim", type=float, default=0.05, help="Phần % cắt phía trên trong bbox (mặc định 0.05)")
    ap.add_argument("--bot-trim", type=float, default=0.20, help="Phần % cắt phía dưới trong bbox (mặc định 0.20)")
    ap.add_argument("--min-side", type=int, default=64, help="Kích thước nhỏ nhất (px) cho crop")
    ap.add_argument("--dbg", type=int, default=0, help="Số dòng debug in ra")
    ap.add_argument("--include-invisible-as-zero", action="store_true",
                    help="Áp dụng cho bộ visible: thêm invisible -> class 0")
    args = ap.parse_args()

    ensure_dir(args.out)
    visible_root = args.out / "visible"
    partial_root = args.out / "partial"
    for r in (visible_root, partial_root):
        ensure_dir(r / "train")
        ensure_dir(r / "val")

    clips, train_set = collect_split_clips(args.root, args.train_ratio, args.seed)
    print(f"Found {len(clips)} clips → train={len(train_set)} val={len(clips)-len(train_set)}")
    if not clips:
        print("[ERR] Không tìm thấy clip."); return

    total_vis_train = Counter(); total_vis_val = Counter()
    total_par_train = Counter(); total_par_val = Counter()

    for clip in clips:
        split = "train" if clip.name in train_set else "val"
        # open json to find category id if player-id not provided
        jpath, vpath = open_pair(clip)
        if not jpath:
            if args.dbg: print(f"[WARN] {clip.name}: no json"); continue
        data = json.loads(jpath.read_text(encoding="utf-8"))
        cat2id = {c.get("name"): c.get("id") for c in data.get("categories", [])}
        pid = args.player_id if args.player_id is not None else cat2id.get(args.player_name)
        if pid is None:
            if args.dbg: print(f"[WARN] {clip.name}: cannot determine player category id (player-name='{args.player_name}'). Skipping clip.")
            continue

        # visible
        cnt_v = build_one_visibility(
            clip, split, visible_root / split, "visible", args.include_invisible_as_zero,
            pid, args.top_trim, args.bot_trim, args.min_side, args.dbg)
        if split == "train": total_vis_train.update(cnt_v)
        else: total_vis_val.update(cnt_v)

        # partial
        cnt_p = build_one_visibility(
            clip, split, partial_root / split, "partially_visible", False,
            pid, args.top_trim, args.bot_trim, args.min_side, args.dbg)
        if split == "train": total_par_train.update(cnt_p)
        else: total_par_val.update(cnt_p)

    print("\n== SUMMARY ==")
    print("VISIBLE  Train:", dict(total_vis_train))
    print("VISIBLE  Val  :", dict(total_vis_val))
    print("PARTIAL  Train:", dict(total_par_train))
    print("PARTIAL  Val  :", dict(total_par_val))
    print(f"\nDone. Output at:\n  {visible_root}\n  {partial_root}\nStructure: <set>/train/0..11, <set>/val/0..11")

if __name__ == "__main__":
    main()
