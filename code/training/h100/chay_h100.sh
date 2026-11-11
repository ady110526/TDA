#!/usr/bin/env bash
# Benchmark trên máy thuê 1 × H100: 7 mô hình + RT-DETRv2-S + RF-DETR-S (code gốc) × 3 seed, trên 2 bộ dữ liệu (TDA, VR-TSD).
# Chạy trong thư mục training/ (cạnh common.py). Thứ tự dùng:
#   bash h100/chay_h100.sh cai_dat
#   bash h100/chay_h100.sh du_lieu tda   "<link Drive của split_data_by_box.zip | đường dẫn zip>"
#   bash h100/chay_h100.sh du_lieu vrtsd "<link tải VR-TSD (Roboflow/Drive) | đường dẫn zip>"
#   bash h100/chay_h100.sh chuan_hoa_vrtsd
#   bash h100/chay_h100.sh ghi_dau_van_tay tda    ;  bash h100/chay_h100.sh ghi_dau_van_tay vrtsd
#   bash h100/chay_h100.sh thu_rtdetr ;  bash h100/chay_h100.sh thu_rfdetr
#   bash h100/chay_h100.sh chay 3
#   bash h100/chay_h100.sh tien_do               (xem bất cứ lúc nào)
#   bash h100/chay_h100.sh sau_train ; bash h100/chay_h100.sh dong_goi
set -euo pipefail
cd "$(dirname "$0")/.."                       # về thư mục training/
export DATA_TDA="${DATA_TDA:-$HOME/tda_data/tda}"
export DATA_VRTSD="${DATA_VRTSD:-$HOME/tda_data/vrtsd}"
LOG_DIR=logs_h100
mkdir -p "$LOG_DIR" ket_qua_xong

data_of() { [ "$1" = tda ] && echo "$DATA_TDA" || echo "$DATA_VRTSD"; }
cfg_of()  { [ "$1" = tda ] && echo config.yaml || echo config_vrtsd.yaml; }

case "${1:-}" in
cai_dat)
  nvidia-smi; echo "CPU: $(nproc) lõi | RAM: $(free -g | awk '/Mem/{print $2}') GB | Ổ đĩa:"; df -h . | tail -1; python3 --version
  python3 -m pip install --upgrade pip
  python3 -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  python3 -m pip install -r requirements.txt gdown
  python3 -c "from ultralytics import YOLO; [YOLO(w) for w in ['yolov8n.pt','yolov8s.pt','yolo11n.pt','yolo11s.pt','yolo26n.pt','yolo26s.pt']]"
  python3 -c "import torchvision; torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')"
  # Code gốc RT-DETR của tác giả. KHÔNG cài requirements.txt của họ (tránh đổi phiên bản torch chung); chỉ cài phần còn thiếu.
  [ -d h100/rtdetr/RT-DETR ] || git clone https://github.com/lyuwenyu/RT-DETR.git h100/rtdetr/RT-DETR
  git -C h100/rtdetr/RT-DETR rev-parse HEAD | tee h100/rtdetr/rtdetr_commit.txt
  python3 -m pip install faster-coco-eval tensorboard scipy pyyaml
  # RF-DETR (gói rfdetr của tác giả, bản 1.11.2) trong môi trường RIÊNG dùng chung torch của máy:
  # ghim các gói chung (torch, numpy, pycocotools, ...) để cài rfdetr không làm đổi phiên bản của các mô hình khác.
  python3 -m pip freeze | grep -iE '^(torch|torchvision|numpy|pycocotools|pillow|opencv-python|ultralytics|scipy|pyyaml)==' > h100/rfdetr/rang_buoc.txt
  [ -x h100/rfdetr/venv/bin/python ] || python3 -m venv --system-site-packages h100/rfdetr/venv
  h100/rfdetr/venv/bin/python -m pip install -c h100/rfdetr/rang_buoc.txt "rfdetr[train]==1.11.2"
  h100/rfdetr/venv/bin/python -m pip freeze > h100/rfdetr/pip_freeze_rfdetr.txt
  mkdir -p h100/rfdetr/weights && (cd h100/rfdetr/weights && ../venv/bin/python -c "from rfdetr import RFDETRSmall; RFDETRSmall(); print('✅ trọng số RF-DETR-S đã tải')")
  python3 -c "import torch, numpy; print('Môi trường chung vẫn là torch', torch.__version__, '| numpy', numpy.__version__)"
  echo "✅ Cài đặt xong" ;;
du_lieu)
  DS="${2:?tda hoặc vrtsd}"; SRC="${3:?link hoặc đường dẫn file zip}"; DST=$(data_of "$DS")
  mkdir -p "$(dirname "$DST")"; TMP="$(dirname "$DST")/${DS}_tai_ve.zip"
  if [ -f "$SRC" ]; then cp "$SRC" "$TMP"
  elif [[ "$SRC" == *drive.google.com* ]]; then python3 -m gdown --fuzzy "$SRC" -O "$TMP"
  else curl -L "$SRC" -o "$TMP"; fi
  rm -rf "$DST" "$DST.tmp" && mkdir -p "$DST.tmp" && unzip -q "$TMP" -d "$DST.tmp"
  Y=$(find "$DST.tmp" -name data.yaml | head -1); mv "$(dirname "$Y")" "$DST"; rm -rf "$DST.tmp"
  echo "✅ $DS ở $DST"; for s in train valid test; do echo "  $s: $(ls "$DST/$s/images" | wc -l) ảnh"; done ;;
chuan_hoa_vrtsd)
  python3 h100/chuan_hoa_anh.py "$DATA_VRTSD" ;;
ghi_dau_van_tay)
  DS="${2:?tda hoặc vrtsd}"; D=$(data_of "$DS"); C=$(cfg_of "$DS")
  FP=$(TDA_CONFIG=$C python3 -c "import common;print(common.dataset_fingerprint('$D'))")
  sed -i "s/^  fingerprint: \".*\"/  fingerprint: \"$FP\"/" "$C"
  TDA_CONFIG=$C python3 train_h100.py --data "$D" --check ;;
thu_rtdetr)
  python3 h100/rtdetr/chay_rtdetr.py --ds tda --data "$DATA_TDA" --seed 0 --smoke 2>&1 | tee "$LOG_DIR/thu_rtdetr.log"
  echo "✅ Nếu dòng cuối là '✅ rtdetrv2_... seed 0: COCO test mAP50-95 = ...' thì RT-DETR đã chạy được." ;;
thu_rfdetr)
  h100/rfdetr/venv/bin/python h100/rfdetr/chay_rfdetr.py --ds tda --data "$DATA_TDA" --seed 0 --smoke 2>&1 | tee "$LOG_DIR/thu_rfdetr.log"
  echo "✅ Nếu dòng cuối là '✅ rfdetr_s seed 0: COCO test mAP50-95 = ...' thì RF-DETR đã chạy được." ;;
chay)
  N="${2:-3}"; CORES=$(nproc); W=$(( CORES / N - 1 )); [ "$W" -gt 8 ] && W=8; [ "$W" -lt 2 ] && W=2
  echo "Chạy song song $N lượt, mỗi lượt $W tiến trình đọc dữ liệu (máy có $CORES lõi)"
  TDA_CONFIG=config.yaml python3 h100/tao_cache.py "$DATA_TDA"
  TDA_CONFIG=config_vrtsd.yaml python3 h100/tao_cache.py "$DATA_VRTSD"
  # Thứ tự: seed 0 (TDA rồi VR-TSD, mọi mô hình) → seed 1 → seed 2. Hết tiền giữa chừng vẫn có đủ mô hình ở seed 0.
  : > "$LOG_DIR/hang_doi.txt"
  for s in 0 1 2; do for d in tda vrtsd; do
    for m in frcnn_r50fpn rtdetr rfdetr yolo26s yolo11s yolov8s yolo26n yolo11n yolov8n; do echo "$d $m $s" >> "$LOG_DIR/hang_doi.txt"; done
  done; done
  nohup bash -c "cat '$LOG_DIR/hang_doi.txt' | xargs -P $N -L 1 bash h100/mot_luot.sh $W" > "$LOG_DIR/hang_doi_chay.log" 2>&1 &
  echo "✅ Đã bắt đầu $(wc -l < "$LOG_DIR/hang_doi.txt") lượt (chạy nền). Xem tiến độ: bash h100/chay_h100.sh tien_do" ;;
tien_do)
  python3 h100/tien_do.py; echo; tail -n 20 "$LOG_DIR/da_xong.txt" 2>/dev/null || true
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv; df -h . | tail -1 ;;
sau_train)
  for DS in tda vrtsd; do
    D=$(data_of "$DS"); C=$(cfg_of "$DS"); mkdir -p "gt_$DS"
    TDA_CONFIG=$C python3 export_gt.py --data "$D" --split test  && mv test_gt_coco.json  "gt_$DS/"
    TDA_CONFIG=$C python3 export_gt.py --data "$D" --split valid && mv valid_gt_coco.json "gt_$DS/"
    TDA_CONFIG=$C python3 predict_valid.py --data "$D"; rm -f valid_gt_coco.json
  done
  python3 benchmark_fps.py --data "$DATA_TDA" --results results
  python3 common.py aggregate results && mv aggregate aggregate_tda
  python3 common.py aggregate results_vrtsd && mv aggregate aggregate_vrtsd ;;
dong_goi)
  OUT=ket_qua_h100_$(date +%Y%m%d_%H%M).zip
  zip -qr "$OUT" results results_vrtsd aggregate_* gt_* fps* logs_h100 config.yaml config_vrtsd.yaml h100/rtdetr/rtdetr_commit.txt \
      h100/rfdetr/pip_freeze_rfdetr.txt -x "*/last.pt" "*/last.pth" "*/checkpoint[0-9]*.pth" "*/*.ckpt" "*/checkpoint_best_regular.pth" "*/checkpoint_best_ema.pth" "*/_cache_tmp/*" || true
  ls -lh "$OUT"; echo "✅ Tải $OUT về máy (chuột phải → Download) TRƯỚC KHI dừng máy thuê." ;;
*) sed -n 2,13p "$0" ;;
esac
