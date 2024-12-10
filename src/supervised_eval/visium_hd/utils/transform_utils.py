import sys
import importlib
import select_utils
importlib.reload(select_utils)

from select_utils import adata_transform
def transform_with_cell_types(sdata, cell_types=None):
    if cell_types is None:
        cell_types = [0,1,2,3]  # 或者直接从数据源中获取默认的 cell_types 列表
    return adata_transform(sdata, cell_types)