"""Chuyển một bộ dữ liệu YOLO (train/valid/test) sang COCO JSON cho RT-DETR (code gốc).
Mã lớp COCO bắt đầu từ 0 (num_classes = nc, remap_mscoco_category = False) theo khuyến nghị của tác giả RT-DETR.
Thứ tự ảnh giống hệt common.build_coco_gt nên image_id khớp với nhãn thật dùng để chấm điểm.
    python h100/rtdetr/yolo2coco.py <thư mục dữ liệu> <thư mục ra>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common  # noqa: E402

root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
y = common.yaml.safe_load(open(root / 'data.yaml', encoding='utf-8'))
names = list(y['names'].values()) if isinstance(y['names'], dict) else list(y['names'])
for split in ('train', 'valid', 'test'):
    gt, _, items = common.build_coco_gt(root, split, len(names))
    anns, dropped = [], 0
    for a in gt['annotations']:
        x, yy, w, h = a['bbox']
        if w < 1 or h < 1:                    # RT-DETR không nhận hộp suy biến
            dropped += 1
            continue
        anns.append({**a, 'category_id': a['category_id'] - 1})
    coco = {'images': gt['images'], 'annotations': anns,
            'categories': [{'id': i, 'name': n} for i, n in enumerate(names)]}
    json.dump(coco, open(out / f'{split}.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'{split}: {len(coco["images"])} ảnh, {len(anns)} hộp (bỏ {dropped} hộp cạnh < 1 px) → {out / (split + ".json")}')
