#!/usr/bin/env bash
# Chạy MỘT lượt trong hàng đợi:  bash h100/mot_luot.sh <workers> <tda|vrtsd> <mô hình> <seed>
# Xong lượt nào thì nén ngay kết quả lượt đó vào ket_qua_xong/ (tải về dần được, không phải chờ hết).
W="$1"; DS="$2"; M="$3"; S="$4"
cd "$(dirname "$0")/.."
LOG="logs_h100/${DS}_${M}_seed${S}.log"
if [ "$DS" = tda ]; then DATA="$DATA_TDA"; export TDA_CONFIG=config.yaml; RES=results
else DATA="$DATA_VRTSD"; export TDA_CONFIG=config_vrtsd.yaml; RES=results_vrtsd; fi
if [ "$M" = rtdetr ]; then
  python3 h100/rtdetr/chay_rtdetr.py --ds "$DS" --data "$DATA" --seed "$S" --workers "$W" > "$LOG" 2>&1
elif [ "$M" = rfdetr ]; then
  h100/rfdetr/venv/bin/python h100/rfdetr/chay_rfdetr.py --ds "$DS" --data "$DATA" --seed "$S" --workers "$W" > "$LOG" 2>&1
else
  python3 train_h100.py --data "$DATA" --only "$M" --seed "$S" --workers "$W" > "$LOG" 2>&1
fi
CODE=$?
echo "$(date +%F_%T) xong $DS $M seed $S (mã thoát $CODE)" >> logs_h100/da_xong.txt
case "$M" in rtdetr) PAT="rtdetrv2_*" ;; rfdetr) PAT="rfdetr_*" ;; *) PAT="$M" ;; esac
RUN=$(ls -d "$RES"/$PAT/seed"$S" 2>/dev/null | head -1)
if [ "$CODE" = 0 ] && [ -n "$RUN" ]; then
  zip -qr "ket_qua_xong/${DS}_${M}_seed${S}.zip" "$RUN" "$LOG" -x "*/last.pt" "*/last.pth" "*/checkpoint[0-9]*.pth" "*/*.ckpt" "*/checkpoint_best_regular.pth" "*/checkpoint_best_ema.pth"
fi
