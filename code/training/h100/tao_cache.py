"""Tạo sẵn file labels.cache của Ultralytics cho train/valid/test TRƯỚC khi chạy song song nhiều lượt train,
để các tiến trình không cùng lúc ghi một file cache. Không train, không đổi dữ liệu (dấu vân tay bỏ qua *.cache).
    TDA_CONFIG=<config> python h100/tao_cache.py <thư mục dữ liệu>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

data = sys.argv[1]
cfg = common.load_config()
ds = common.check_dataset(data, cfg, require_fingerprint=True)
out = Path(__file__).resolve().parent / '_cache_tmp' / Path(ds['root']).name
yaml_path = common.resolved_data_yaml(ds, out)

from ultralytics.cfg import DEFAULT_CFG  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

d = check_det_dataset(yaml_path)
for split in ('train', 'val', 'test'):
    YOLODataset(img_path=d[split], data=d, task='detect', augment=False, hyp=DEFAULT_CFG, prefix=f'{split}: ')
caches = sorted(Path(ds['root']).rglob('*.cache'))
print('✅ Đã tạo cache:', [str(c.relative_to(ds['root'])) for c in caches])
