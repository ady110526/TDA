"""
RF-DETR (Robinson et al., ICLR 2026) bằng CODE GỐC của tác giả: gói `rfdetr` của Roboflow
(github.com/roboflow/rf-detr, bản phát hành 1.11.2, giấy phép Apache 2.0), trên bộ TDA hoặc VR-TSD.

Code gốc được dùng NGUYÊN VẸN: kiến trúc, độ phân giải đầu vào mặc định của từng cỡ (Small: 512), tăng cường dữ liệu,
lịch học, EMA, cách chọn checkpoint tốt nhất trên tập valid đều là mặc định của gói. Các tham số được đặt tường minh:
  epochs = 100, lr = 1e-4                 (mặc định của gói, cũng là ví dụ trong tài liệu huấn luyện của tác giả)
  batch_size = 16, grad_accum_steps = 1   (tài liệu tác giả: GPU 40–80 GB → 16 × 1, batch hiệu dụng 16)
  seed = 0/1/2                            (gói mặc định không đặt seed)
Script này chỉ: (1) tạo một "khung nhìn" dữ liệu bằng hard link (không chép ảnh, không đổi file gốc) kèm data.yaml
sạch, (2) gọi model.train() của tác giả, (3) dự đoán test/valid bằng checkpoint tốt nhất trên valid và chấm bằng ĐÚNG
công cụ chung của bài (common.coco_evaluate, pycocotools) để so sánh được với YOLO, Faster R-CNN và RT-DETRv2.

Chạy bằng Python của môi trường riêng (cài ở bước cai_dat, không đụng tới môi trường chung):
    h100/rfdetr/venv/bin/python h100/rfdetr/chay_rfdetr.py --ds tda --data <thư mục TDA> --seed 0 [--smoke]
Bị ngắt: chạy lại đúng lệnh, tự tiếp tục từ last.ckpt.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parents[1]                      # thư mục training/
CONFIGS = {'tda': TRAIN_DIR / 'config.yaml', 'vrtsd': TRAIN_DIR / 'config_vrtsd.yaml'}
SIZES = {'s': 'RFDETRSmall', 'n': 'RFDETRNano', 'm': 'RFDETRMedium'}
HYP = {'epochs': 100, 'batch_size': 16, 'grad_accum_steps': 1, 'lr': 1e-4}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def make_view(src, dst, names):
    """Khung nhìn dữ liệu cho rfdetr: train/valid/test/{images,labels} bằng hard link + data.yaml chỉ có tên lớp và
    đường dẫn tương đối. Hard link = cùng một file trên đĩa → dữ liệu gốc và dấu vân tay không đổi."""
    if (dst / 'data.yaml').exists():
        return
    tmp = dst.with_name(dst.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    for split in ('train', 'valid', 'test'):
        for sub in ('images', 'labels'):
            s, d = src / split / sub, tmp / split / sub
            d.mkdir(parents=True)
            for f in s.iterdir():
                if f.is_file() and not f.name.endswith('.cache'):
                    try:
                        os.link(f, d / f.name)
                    except OSError:
                        shutil.copy2(f, d / f.name)
    import yaml
    yaml.safe_dump({'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
                    'nc': len(names), 'names': list(names)}, open(tmp / 'data.yaml', 'w', encoding='utf-8'),
                   allow_unicode=True, sort_keys=False)
    tmp.rename(dst)


def done_epochs(out):
    """Số epoch đã xong, đọc từ metrics.csv (CSVLogger của PyTorch Lightning trong gói rfdetr)."""
    import csv
    m = out / 'metrics.csv'
    if not m.exists():
        return 0
    rows = list(csv.DictReader(open(m, encoding='utf-8')))
    return len({r['epoch'] for r in rows if r.get('epoch') not in (None, '') and any(k.startswith('val') and r[k] for k in r)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', choices=list(CONFIGS), required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--size', choices=list(SIZES), default=os.environ.get('RFDETR_SIZE', 's'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--smoke', action='store_true', help='1 epoch để kiểm tra cài đặt (kết quả để ở _smoke)')
    args = ap.parse_args()

    os.environ['TDA_CONFIG'] = str(CONFIGS[args.ds])
    sys.path.insert(0, str(TRAIN_DIR))
    import common                                       # noqa: E402  (sau khi đặt TDA_CONFIG)
    cfg = common.load_config()
    ds = common.check_dataset(args.data, cfg, require_fingerprint=not args.smoke)
    env = common.env_info()
    key = f'rfdetr_{args.size}'
    results_dir = (TRAIN_DIR / cfg['run']['results_dir']).resolve()
    base = results_dir / ('_smoke' if args.smoke else '') / key / f'seed{args.seed}'
    if (base / 'metrics.json').exists() and not args.smoke:
        print(f'⏭  {key} seed {args.seed} ({args.ds}) đã xong'); return
    base.mkdir(parents=True, exist_ok=True)
    common.save_json(env, base / 'env.json')

    import rfdetr
    from importlib.metadata import version as _v
    RFV = _v('rfdetr')
    names = list(ds['names'])
    if len(set(names)) != len(names):
        raise SystemExit('❌ Tên lớp bị trùng trong data.yaml → không ánh xạ được lớp dự đoán của RF-DETR theo tên.')
    view = HERE / f'_data_{args.ds}'
    make_view(Path(ds['root']), view, names)

    # 1) huấn luyện bằng model.train() của tác giả, từ trọng số COCO của tác giả
    out = base / 'train'
    done = base / 'train_done.json'
    wdir = HERE / 'weights'; wdir.mkdir(exist_ok=True)
    os.chdir(wdir)                                      # trọng số COCO tải về/đọc lại ở đây
    Model = getattr(rfdetr, SIZES[args.size])
    hyp = dict(HYP, epochs=1) if args.smoke else dict(HYP)
    if not done.exists():
        model = Model()
        last = out / 'last.ckpt'
        kw = dict(dataset_dir=str(view), output_dir=str(out), seed=args.seed, num_workers=args.workers,
                  run_test=False, progress_bar=None, **hyp)
        if last.exists() and not args.smoke:
            kw['resume'] = str(last)
        common.save_json({'rfdetr_version': RFV,
                          'class': SIZES[args.size], 'train_kwargs': kw}, base / 'cau_hinh_rfdetr.json')
        print('model.train(', kw, ')')
        t0 = time.time()
        model.train(**kw)
        common.save_json({'train_seconds_last_session': round(time.time() - t0, 1)}, done)

    # 2) dự đoán test/valid bằng checkpoint tốt nhất trên valid, chấm bằng công cụ chung
    best = next((out / n for n in ('checkpoint_best_total.pth', 'checkpoint_best_ema.pth',
                                   'checkpoint_best_regular.pth') if (out / n).exists()), None)
    assert best, f'Không thấy checkpoint_best_*.pth trong {out}'
    import numpy as np
    import torch
    from PIL import Image
    model = Model.from_checkpoint(str(best))
    cls_names = list(model.class_names)
    idx = {n: i for i, n in enumerate(names)}
    miss = [n for n in cls_names if n not in idx]
    if miss:
        raise SystemExit(f'❌ Tên lớp của mô hình không có trong data.yaml: {miss[:5]}')
    y = cfg['yolo']

    def predict(split, timing=False):
        gt, path_to_id, items = common.build_coco_gt(ds['root'], split, ds['nc'])
        preds, ms = [], []
        for p, _ in items:
            im = Image.open(p).convert('RGB')
            if timing:
                torch.cuda.synchronize(); t = time.perf_counter()
            det = model.predict(im, threshold=y['eval_conf'])
            if timing:
                torch.cuda.synchronize(); ms.append((time.perf_counter() - t) * 1000)
            iid = path_to_id[str(Path(p).resolve())]
            if det.confidence is None or len(det) == 0:
                continue
            order = np.argsort(-det.confidence)[:y['max_det']]
            for j in order:
                x0, y0, x1, y1 = det.xyxy[j].tolist()
                preds.append({'image_id': iid, 'category_id': idx[cls_names[int(det.class_id[j])]] + 1,
                              'bbox': [x0, y0, x1 - x0, y1 - y0], 'score': float(det.confidence[j])})
        return gt, preds, ms

    gt, preds, ms = predict('test', timing=True)
    common.save_json(preds, base / 'test_predictions_coco.json')
    coco_res, per_class = common.coco_evaluate(gt, preds, ds['names'])
    _, vpreds, _ = predict('valid')
    common.save_json(vpreds, base / 'valid_predictions_coco.json')
    ctx = {'cfg': cfg, 'env': env, 'ds': ds, 'machine': {'id': 'h100'}, 'results_dir': results_dir}
    tr = json.load(open(done, encoding='utf-8'))
    pre = sorted(wdir.glob('*.pth'))
    common.finish_run(base, ctx, key, args.seed, coco_res, per_class,
                      {'code_goc': 'roboflow/rf-detr (pip rfdetr)', 'rfdetr_version': RFV,
                       'model_class': SIZES[args.size], 'resolution': int(model.model_config.resolution),
                       **{f'hyp_{k}': v for k, v in hyp.items()},
                       'pretrained': ';'.join(f'{p.name}:{sha256(p)[:16]}' for p in pre),
                       'checkpoint_dung': best.name, 'epochs_xong': done_epochs(out),
                       'ms_moi_anh_trung_vi': round(float(np.median(ms[50:] if len(ms) > 60 else ms)), 2),
                       'train_seconds_last_session': tr['train_seconds_last_session'], 'n_test_images': len(gt['images'])})


if __name__ == '__main__':
    main()
