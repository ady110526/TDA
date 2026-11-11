# Train trên máy thuê 1 × H100 80 GB: 2 bộ dữ liệu × (7 mô hình + RT-DETRv2-S + RF-DETR-S) × 3 seed

| | TDA (bộ của nhóm) | VR-TSD (đối chứng) |
|---|---|---|
| Dữ liệu | `output_split_fit_tau10/split_data_by_box.zip` (export Roboflow "Fit (black edges) in" 1280×1280, chia theo nhóm dHash τ = 10, phân tầng theo số hộp) | Bản export Roboflow v1 (8.078 ảnh, 58 lớp), **giữ nguyên phép chia gốc** train/valid/test của bản export |
| Xử lý ảnh | Roboflow "Fit with border" 1280×1280 | giữ nguyên bản export (640×640); bước `chuan_hoa_vrtsd` chỉ thu nhỏ ảnh có cạnh > 1920 nên với VR-TSD không đổi ảnh nào, chạy để có biên bản kiểm tra |
| Cấu hình | `config.yaml` → kết quả ở `results/` | `config_vrtsd.yaml` (giống hệt, chỉ khác tên dữ liệu) → `results_vrtsd/` |

**Mô hình:** YOLOv8/YOLO11/YOLO26 (n, s), Faster R-CNN R50-FPN (cấu hình như trong bài), và **RT-DETRv2-S** chạy bằng **code gốc của tác giả** (github.com/lyuwenyu/RT-DETR, `rtdetrv2_pytorch`, cấu hình `rtdetrv2_r18vd_120e_coco`, trọng số COCO của tác giả, lịch huấn luyện gốc 120 epoch), và **RF-DETR-S** (Robinson et al., ICLR 2026) chạy bằng **code gốc của tác giả** (gói `rfdetr` bản 1.11.2 của Roboflow, Apache 2.0, trọng số COCO của tác giả; giữ mặc định của tác giả: độ phân giải 512, 100 epoch, lr 1e-4; batch 16 theo hướng dẫn của tác giả cho GPU 40–80 GB). RF-DETR được cài trong **môi trường riêng** (`h100/rfdetr/venv`) để không đổi phiên bản của các mô hình khác. Mọi mô hình được chấm bằng cùng công cụ pycocotools của bài.
**Lưu kết quả:** xong lượt nào, kết quả lượt đó được nén ngay vào thư mục `ket_qua_xong/` để tải về dần.

## 0. Trước khi thuê máy
1. Drive: chia sẻ `output_split_fit_tau10/split_data_by_box.zip` ở chế độ "Bất kỳ ai có đường liên kết", sao chép link.
2. VR-TSD: trên Roboflow, project VR-TSD → Versions → v2 → **Download Dataset** → định dạng **YOLOv8** → chọn **"Show download code"** → tab **Terminal** → sao chép **đường link** trong lệnh `curl -L "https://app.roboflow.com/ds/....?key=...."` (chỉ phần trong ngoặc kép).
3. Nếu FPT cho thêm **Persistent disk**, chọn khoảng 50 GB (dừng máy giữa chừng không mất kết quả).

## 1. Cài đặt (khoảng 10 phút)
JupyterLab → kéo thả `code_train_h100.zip` → File → New → Terminal:
```bash
unzip -q code_train_h100.zip && cd training
bash h100/chay_h100.sh cai_dat
```
**Chụp phần đầu (GPU, CPU, RAM, ổ đĩa) gửi Claude.**

## 2. Dữ liệu
```bash
bash h100/chay_h100.sh du_lieu tda   "<link Drive bước 0.1>"
bash h100/chay_h100.sh du_lieu vrtsd "<link Roboflow bước 0.2>"
bash h100/chay_h100.sh chuan_hoa_vrtsd
bash h100/chay_h100.sh ghi_dau_van_tay tda
bash h100/chay_h100.sh ghi_dau_van_tay vrtsd
```
Số ảnh train/valid/test của TDA phải khớp `split_summary.json` của notebook 03 bản mới. **Chụp màn hình hai lệnh `ghi_dau_van_tay` gửi Claude** (mã cấu hình và dấu vân tay dữ liệu ghi vào bài).

## 3. Chạy thử RT-DETR và RF-DETR (mỗi cái 1 epoch, khoảng 5–15 phút)
```bash
bash h100/chay_h100.sh thu_rtdetr
bash h100/chay_h100.sh thu_rfdetr
```
Code của hai mô hình này **chưa được chạy thử trên GPU thật** (Claude không có GPU); bước này để bắt lỗi trước khi chạy thật. Lỗi thì gửi Claude `logs_h100/thu_rtdetr.log` (+ `results/_smoke/rtdetrv2_r18vd/seed0/train_stdout.log`) hoặc `logs_h100/thu_rfdetr.log`.

## 4. Chạy thật
```bash
bash h100/chay_h100.sh chay 3
```
54 lượt, chạy nền 3 lượt cùng lúc, thứ tự: **seed 0** (TDA rồi VR-TSD, đủ 9 mô hình) → seed 1 → seed 2. Hết tiền giữa chừng vẫn có đủ mô hình ở seed 0.
Bị ngắt: chạy lại đúng lệnh này; lượt đã xong được bỏ qua, lượt dở chạy tiếp từ checkpoint.

## 5. Theo dõi
```bash
bash h100/chay_h100.sh tien_do
```
**Sau khoảng 30–45 phút, gửi Claude kết quả lệnh này** để tính tổng số giờ và tiền thật.
Tải dần các file trong `ket_qua_xong/` về máy (chuột phải → Download).

## 6. Khi xong
```bash
bash h100/chay_h100.sh sau_train
bash h100/chay_h100.sh dong_goi
```
Tải `ket_qua_h100_....zip` về, kiểm tra mở được, **rồi mới dừng máy thuê** (không có persistent disk thì dừng máy là mất hết).
