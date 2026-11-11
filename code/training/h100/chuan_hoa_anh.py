"""Chuẩn hóa độ phân giải một bộ dữ liệu YOLO (dùng cho VR-TSD) theo ĐÚNG quy tắc của notebook 01b (bộ TDA):
giữ tỉ lệ khung; ảnh có cạnh dài > 1920 thu về 1920 bằng cv2.INTER_AREA, lưu JPEG chất lượng 95; không phóng to;
không sửa nhãn. Sửa TẠI CHỖ trên bản sao dữ liệu ở máy thuê; ghi nhật ký vào <thư mục>/chuan_hoa_log.csv.
    python h100/chuan_hoa_anh.py <thư mục dữ liệu (có train/valid/test)>
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageOps

MAX_LONG, Q = 1920, 95
root = Path(sys.argv[1])
log = root / 'chuan_hoa_log.csv'
if log.exists():
    raise SystemExit(f'Đã chuẩn hóa trước đó ({log}). Không chạy lại để tránh nén ảnh hai lần.')
rows, res, n_rot, n_nolabel, n_empty = [], Counter(), 0, 0, 0
for split in ('train', 'valid', 'test'):
    for p in sorted((root / split / 'images').iterdir()):
        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            continue
        lp = root / split / 'labels' / (p.stem + '.txt')
        if not lp.exists():
            n_nolabel += 1
        elif not lp.read_text(encoding='utf-8').strip():
            n_empty += 1
        with Image.open(p) as im:
            w, h = im.size
            if im.getexif().get(274, 1) != 1:
                n_rot += 1
        res[f'{w}x{h}'] += 1
        s = MAX_LONG / max(w, h)
        if s < 1:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            w1, h1 = int(round(w * s)), int(round(h * s))
            small = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA)
            params = [cv2.IMWRITE_JPEG_QUALITY, Q] if p.suffix.lower() in {'.jpg', '.jpeg'} else []
            assert cv2.imwrite(str(p), small, params), f'Không ghi được {p}'
            rows.append([split, p.name, w, h, w1, h1, round(s, 6)])
with open(log, 'w', newline='', encoding='utf-8') as f:
    csv.writer(f).writerows([['split', 'image', 'w0', 'h0', 'w1', 'h1', 'scale'], *rows])
print(f'Số ảnh: {sum(res.values())} | thu nhỏ: {len(rows)} | ảnh không có file nhãn: {n_nolabel} | file nhãn rỗng: {n_empty} '
      f'| ảnh có cờ xoay EXIF: {n_rot}')
print('Độ phân giải phổ biến (trước):', res.most_common(8))
if n_rot:
    print('⚠️ Có ảnh mang cờ xoay EXIF: báo lại cho Claude trước khi train.')
