"""In tiến độ mọi lượt train (TDA: results/, VR-TSD: results_vrtsd/): epoch đã xong, phút/epoch, giờ còn lại."""
import csv
import json
import re
from pathlib import Path

T = Path(__file__).resolve().parent.parent
TOTAL = {'frcnn_r50fpn': 26}


def rtdetr_epochs(d):
    """Đếm epoch đã xong của RT-DETR từ log.txt của tác giả (mỗi epoch một dòng JSON có 'epoch')."""
    log = d / 'train' / 'log.txt'
    if not log.exists():
        return 0, None
    eps = [json.loads(l).get('epoch') for l in log.read_text(encoding='utf-8').splitlines() if l.strip().startswith('{')]
    cfg = (d / 'cau_hinh_rtdetr.yml')
    m = re.search(r'(\d+)e_coco', cfg.read_text()) if cfg.exists() else None
    return len(eps), int(m.group(1)) if m else None


def rfdetr_epochs(d):
    """Đếm epoch đã xong của RF-DETR từ metrics.csv (mỗi epoch có ít nhất một dòng chỉ số val/...)."""
    m = d / 'train' / 'metrics.csv'
    total = None
    cj = d / 'cau_hinh_rfdetr.json'
    if cj.exists():
        total = json.load(open(cj, encoding='utf-8'))['train_kwargs'].get('epochs')
    if not m.exists():
        return 0, total
    rows = list(csv.DictReader(open(m, encoding='utf-8')))
    eps = {r['epoch'] for r in rows if r.get('epoch') and any(k.startswith('val') and r[k] for k in r)}
    return len(eps), total


for name, R in (('TDA', T / 'results'), ('VR-TSD', T / 'results_vrtsd')):
    if not R.exists():
        continue
    print(f'===== {name} ({R.name}) =====')
    for d in sorted(R.glob('*/seed*')):
        if d.parent.name.startswith('_'):
            continue
        model, seed = d.parent.name, d.name
        if (d / 'metrics.json').exists():
            m = json.load(open(d / 'metrics.json', encoding='utf-8'))
            print(f'{model:14s} {seed:6s} XONG   mAP50-95 test = {m["coco_test"]["mAP50_95"]:.4f}')
            continue
        if model.startswith('rtdetrv2'):
            n, total = rtdetr_epochs(d)
            log = d / 'train' / 'log.txt'
        elif model.startswith('rfdetr'):
            n, total = rfdetr_epochs(d)
            log = d / 'train' / 'metrics.csv'
        else:
            log = d / ('train_log.csv' if model == 'frcnn_r50fpn' else 'train/results.csv')
            n = len(list(csv.reader(open(log, encoding='utf-8')))) - 1 if log.exists() else 0
            total = TOTAL.get(model, 100)
        start = (d / 'env.json').stat().st_mtime if (d / 'env.json').exists() else None
        if not n or not start or not log.exists():
            print(f'{model:14s} {seed:6s} đang khởi động / epoch đầu')
            continue
        per = (log.stat().st_mtime - start) / n
        con = f', còn ~{per * (total - n) / 3600:.1f} giờ' if total else ''
        print(f'{model:14s} {seed:6s} {n}/{total or "?"} epoch, ~{per / 60:.1f} phút/epoch{con}')
