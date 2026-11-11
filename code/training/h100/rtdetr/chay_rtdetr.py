"""
RT-DETRv2 bằng CODE GỐC của tác giả (github.com/lyuwenyu/RT-DETR, thư mục rtdetrv2_pytorch) trên bộ TDA hoặc VR-TSD.

Code gốc được dùng NGUYÊN VẸN: cấu hình mô hình, lịch huấn luyện, tăng cường dữ liệu, cách chọn checkpoint đều của
tác giả. Script này chỉ: (1) chuyển dữ liệu YOLO sang COCO, (2) viết một file cấu hình nhỏ `__include__` cấu hình gốc
và chỉ thay đường dẫn dữ liệu + số lớp, (3) gọi tools/train.py của tác giả, (4) dự đoán tập test/valid và chấm điểm
bằng ĐÚNG công cụ chung của bài (common.coco_evaluate, pycocotools) để so sánh được với YOLO và Faster R-CNN.

    python h100/rtdetr/chay_rtdetr.py --ds tda   --data <thư mục TDA>    --seed 0 [--smoke]
    python h100/rtdetr/chay_rtdetr.py --ds vrtsd --data <thư mục VR-TSD> --seed 0
Bị ngắt: chạy lại đúng lệnh, tự tiếp tục từ last.pth.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parents[1]                      # thư mục training/
REPO = HERE / 'RT-DETR' / 'rtdetrv2_pytorch'
CONFIGS = {'tda': TRAIN_DIR / 'config.yaml', 'vrtsd': TRAIN_DIR / 'config_vrtsd.yaml'}


def find_ckpt_url(model_cfg):
    """Lấy link trọng số COCO của đúng cấu hình từ README của tác giả (không ghi cứng link)."""
    txt = (REPO / 'README.md').read_text(encoding='utf-8')
    toks = model_cfg.replace('_coco', '').split('_')[1:]           # ví dụ ['r18vd', '120e']
    urls = re.findall(r'https://[^\s)\]"\']+\.pth', txt)
    hit = [u for u in urls if all(t in u for t in toks)]
    if not hit:
        raise SystemExit(f'❌ Không tìm thấy link trọng số cho {model_cfg} trong README. Các link có: {urls}')
    return hit[0]


def latest(path, pats):
    for pat in pats:
        c = sorted(path.glob(pat), key=lambda p: p.stat().st_mtime)
        if c:
            return c[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', choices=list(CONFIGS), required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model-cfg', default=os.environ.get('RTDETR_CFG', 'rtdetrv2_r18vd_120e_coco'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--smoke', action='store_true', help='1 epoch để kiểm tra cài đặt (kết quả để ở _smoke)')
    args = ap.parse_args()

    os.environ['TDA_CONFIG'] = str(CONFIGS[args.ds])
    sys.path.insert(0, str(TRAIN_DIR))
    import common                                       # noqa: E402  (sau khi đặt TDA_CONFIG)
    cfg = common.load_config()
    ds = common.check_dataset(args.data, cfg, require_fingerprint=not args.smoke)
    env = common.env_info()
    key = 'rtdetrv2_' + args.model_cfg.split('_')[1]   # ví dụ rtdetrv2_r18vd
    results_dir = (TRAIN_DIR / cfg['run']['results_dir']).resolve()
    base = results_dir / ('_smoke' if args.smoke else '') / key / f'seed{args.seed}'
    if (base / 'metrics.json').exists() and not args.smoke:
        print(f'⏭  {key} seed {args.seed} ({args.ds}) đã xong'); return
    base.mkdir(parents=True, exist_ok=True)
    common.save_json(env, base / 'env.json')
    assert REPO.exists(), f'Chưa có code gốc ở {REPO}. Chạy: bash h100/chay_h100.sh cai_rtdetr'

    # 1) dữ liệu COCO
    coco = HERE / f'coco_{args.ds}'
    if not (coco / 'test.json').exists():
        subprocess.run([sys.executable, str(HERE / 'yolo2coco.py'), ds['root'], str(coco)], check=True)

    # 2) trọng số COCO của tác giả
    url = find_ckpt_url(args.model_cfg)
    wdir = HERE / 'weights'; wdir.mkdir(exist_ok=True)
    pre = wdir / url.split('/')[-1]
    if not pre.exists():
        print('Tải', url); urllib.request.urlretrieve(url, pre)

    # 3) cấu hình: include cấu hình gốc, chỉ thay dữ liệu, số lớp, thư mục ra
    root = Path(ds['root'])
    ycfg = base / 'cau_hinh_rtdetr.yml'
    lines = [f"__include__: ['{(REPO / 'configs' / 'rtdetrv2' / (args.model_cfg + '.yml')).as_posix()}']",
             f"output_dir: {(base / 'train').as_posix()}",
             f"num_classes: {ds['nc']}", "remap_mscoco_category: False",
             "train_dataloader:", "  dataset:", f"    img_folder: {(root / 'train' / 'images').as_posix()}",
             f"    ann_file: {(coco / 'train.json').as_posix()}", f"  num_workers: {args.workers}",
             "val_dataloader:", "  dataset:", f"    img_folder: {(root / 'valid' / 'images').as_posix()}",
             f"    ann_file: {(coco / 'valid.json').as_posix()}", f"  num_workers: {args.workers}"]
    if args.smoke:
        lines.append('epoches: 1')
    ycfg.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # 4) huấn luyện bằng tools/train.py của tác giả
    done = base / 'train_done.json'
    if not done.exists():
        last = latest(base / 'train', ['last.pth', 'checkpoint*.pth'])
        port = 29500 + int(hashlib.md5(str(base).encode()).hexdigest(), 16) % 2000
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=1', f'--master_port={port}',
               'tools/train.py', '-c', str(ycfg), '--use-amp', f'--seed={args.seed}']
        cmd += ['-r', str(last)] if (last and not args.smoke) else ['-t', str(pre)]
        print('Lệnh:', ' '.join(cmd))
        t0 = time.time()
        with open(base / 'train_stdout.log', 'a', encoding='utf-8') as lf:
            r = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise SystemExit(f'❌ tools/train.py lỗi (mã {r.returncode}). Xem {base / "train_stdout.log"}')
        common.save_json({'train_seconds_last_session': round(time.time() - t0, 1)}, done)

    # 5) dự đoán test/valid bằng mô hình tốt nhất trên valid, chấm bằng công cụ chung
    best = latest(base / 'train', ['best*.pth', 'last.pth', 'checkpoint*.pth'])
    assert best, f'Không thấy checkpoint trong {base / "train"}'
    import torch
    import torchvision.transforms as T
    from PIL import Image
    sys.path.insert(0, str(REPO))
    from src.core import YAMLConfig                     # code gốc của tác giả
    yc = YAMLConfig(str(ycfg), resume=str(best))
    ck = torch.load(best, map_location='cpu', weights_only=False)
    state = ck['ema']['module'] if 'ema' in ck else ck['model']
    yc.model.load_state_dict(state)
    model, post = yc.model.deploy().cuda().eval(), yc.postprocessor.deploy()
    tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])   # giống references/deploy của tác giả
    y = cfg['yolo']

    def predict(split, timing=False):
        gt, path_to_id, items = common.build_coco_gt(ds['root'], split, ds['nc'])
        preds, ms = [], []
        with torch.inference_mode():
            for p, _ in items:
                im = Image.open(p).convert('RGB')
                if timing:
                    torch.cuda.synchronize(); t = time.perf_counter()
                x = tf(im)[None].cuda()
                size = torch.tensor([[im.width, im.height]], device='cuda')
                labels, boxes, scores = post(model(x), size)
                labels, boxes, scores = labels[0].cpu().numpy(), boxes[0].float().cpu().numpy(), scores[0].float().cpu().numpy()
                if timing:
                    torch.cuda.synchronize(); ms.append((time.perf_counter() - t) * 1000)
                iid = path_to_id[str(Path(p).resolve())]
                keep = scores >= y['eval_conf']
                order = scores[keep].argsort()[::-1][:y['max_det']]
                for (x0, y0, x1, y1), c, s in zip(boxes[keep][order], labels[keep][order], scores[keep][order]):
                    preds.append({'image_id': iid, 'category_id': int(c) + 1,
                                  'bbox': [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], 'score': float(s)})
        return gt, preds, ms

    gt, preds, ms = predict('test', timing=True)
    common.save_json(preds, base / 'test_predictions_coco.json')
    coco_res, per_class = common.coco_evaluate(gt, preds, ds['names'])
    _, vpreds, _ = predict('valid')
    common.save_json(vpreds, base / 'valid_predictions_coco.json')
    import numpy as np
    ctx = {'cfg': cfg, 'env': env, 'ds': ds, 'machine': {'id': 'h100'}, 'results_dir': results_dir}
    tr = json.load(open(done, encoding='utf-8'))
    common.finish_run(base, ctx, key, args.seed, coco_res, per_class,
                      {'code_goc': 'lyuwenyu/RT-DETR rtdetrv2_pytorch', 'model_cfg': args.model_cfg,
                       'pretrained': url, 'checkpoint_dung': best.name,
                       'ms_moi_anh_trung_vi_fp32': round(float(np.median(ms[50:] if len(ms) > 60 else ms)), 2),
                       'train_seconds_last_session': tr['train_seconds_last_session'], 'n_test_images': len(gt['images'])})


if __name__ == '__main__':
    main()
