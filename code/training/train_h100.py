"""
MÁY THUÊ – 1 × NVIDIA H100 80 GB (Linux). Chạy cả 7 mô hình × 3 seed trên CÙNG một GPU.

Dùng chung common.py và config.yaml với các file train khác. Cách chạy xem training/h100/HUONG_DAN_H100.md:
    python train_h100.py --data <thư mục split_data_by_box> --check
    python train_h100.py --data <thư mục split_data_by_box> --only yolov8n --seed 0 --workers 8
"""
import common

MACHINE = {
    'id': 'h100',
    'label': 'Máy thuê – NVIDIA H100 80 GB',
    'gpu_hint': 'H100',          # tên GPU phải chứa chuỗi này
    'models': ['yolov8n', 'yolo11n', 'yolo26n', 'yolov8s', 'yolo11s', 'yolo26s', 'frcnn_r50fpn'],
}

if __name__ == '__main__':
    common.main(MACHINE)
