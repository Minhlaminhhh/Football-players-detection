#!/usr/bin/env python3
import json, cv2, re
from pathlib import Path

ROOT = Path(r"C:\Users\qmphi\Downloads\football_train")     # thư mục gốc chứa nhiều folder con, mỗi folder: video + json
OUT  = Path(r"D:\A.I\output label")      # nơi xuất images/ và labels/
OUT_IMAGES = OUT / "images"
OUT_LABELS = OUT / "labels"

# ==== KHÔNG ĐỔI KÍCH THƯỚC ẢNH ====
def coco_class_map(categories):
    cats_sorted = sorted(categories, key=lambda c: c["id"])
    id2yolo = {c["id"]: i for i, c in enumerate(cats_sorted)}
    names = [c["name"] for c in cats_sorted]
    return id2yolo, names

def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def parse_frame_idx(fname):
    # "frame_000123.PNG" -> 123
    m = re.search(r"frame_(\d+)\.", fname)
    return int(m.group(1)) if m else None

def export_classes(all_names):
    (OUT / "classes.txt").write_text("\n".join(all_names), encoding="utf-8")

def convert_one_folder(folder: Path):
    # tìm 1 json + 1 mp4 cùng "stem"
    jsons = list(folder.glob("*.json"))
    mp4s  = list(folder.glob("*.mp4"))
    if not jsons or not mp4s:
        return False, f"Skip {folder.name}: thiếu json/mp4"

    # ưu tiên cặp cùng stem
    pair = None
    for j in jsons:
        for v in mp4s:
            if j.stem.replace("_subclip","") == v.stem.replace("_subclip",""):
                pair = (j, v); break
        if pair: break
    if not pair:
        pair = (jsons[0], mp4s[0])

    jpath, vpath = pair
    data = load_json(jpath)
    id2yolo, names = coco_class_map(data["categories"])

    # map image_id -> record; file_name -> image_id
    img_by_id = {im["id"]: im for im in data["images"]}
    id_by_name = {im["file_name"]: im["id"] for im in data["images"]}
    anns_by_img = {}
    for a in data.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)

    # kiểm tra kích thước
    # (đa số 3840x1200 hoặc 3840x900 theo JSON)
    W_json = img_by_id[next(iter(img_by_id))]["width"]
    H_json = img_by_id[next(iter(img_by_id))]["height"]

    # xuất khung hình
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return False, f"Không mở được video: {vpath.name}"
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W, H) != (W_json, H_json):
        print(f"[WARN] {folder.name}: Video {W}x{H} != JSON {W_json}x{H_json} -> vẫn tiếp tục, nhưng cần chắc JSON đúng video")

    # tạo thư mục out cho clip này
    clip_name = folder.name
    img_out_dir = OUT_IMAGES / clip_name
    lbl_out_dir = OUT_LABELS / clip_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # duyệt frame & ghi đúng tên như JSON mong đợi (frame_000000.PNG, ...)
    fid = 0
    ok_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        fname = f"frame_{fid:06d}.PNG"
        # chỉ ghi nếu tên frame này có trong JSON
        if fname in id_by_name:
            cv2.imwrite(str(img_out_dir / fname), frame)
            im_id = id_by_name[fname]
            # ghi nhãn YOLO
            lines = []
            for ann in anns_by_img.get(im_id, []):
                cls = id2yolo.get(ann["category_id"])
                x, y, w, h = ann["bbox"]  # COCO xywh
                xc = (x + w/2) / W_json
                yc = (y + h/2) / H_json
                ww = w / W_json
                hh = h / H_json
                # clamp nhẹ để tránh ra ngoài [0,1]
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                ww = min(max(ww, 0.0), 1.0)
                hh = min(max(hh, 0.0), 1.0)
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            (lbl_out_dir / (fname.replace(".PNG", ".txt"))).write_text("\n".join(lines), encoding="utf-8")
            ok_frames += 1
        fid += 1

    cap.release()
    return True, f"OK {folder.name}: {ok_frames} frame khớp JSON"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_classnames = set()
    # quét các folder con
    results = []
    for sub in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        ok, msg = convert_one_folder(sub)
        results.append(msg)
        # thu thập tên lớp (đồng nhất trong tất cả json)
        for jfile in sub.glob("*.json"):
            cats = load_json(jfile)["categories"]
            for c in cats:
                all_classnames.add(c["name"])
    # ghi classes.txt theo thứ tự ổn định
    export_classes(sorted(all_classnames))
    print("\n".join(results))
    print(f"Done. Output: {OUT}")

if __name__ == "__main__":
    main()
