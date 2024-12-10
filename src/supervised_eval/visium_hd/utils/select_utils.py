import torch
from typing import Tuple
import torch.nn.functional as F
from spatialdata import SpatialData

CELL_TYPES = [0, 1, 2, 3]

# def adata_transform(sdata: SpatialData, cell_types) -> Tuple[torch.Tensor, torch.Tensor]:
#     if 'Mayo_VisiumHD_789_full_image' in sdata:
#         tile = sdata['Mayo_VisiumHD_789_full_image'].data.compute()
#         tile = torch.tensor(tile, dtype=torch.float32)
#     else:
#         raise KeyError("The image 'Mayo_VisiumHD_789_full_image' is not present in sdata.")
    
#     if "square_016um" in sdata and 'Cluster' in sdata["square_016um"].obs.columns:
#         expected_category = sdata["square_016um"].obs['Cluster'].values[0]
#         if expected_category in cell_types:
#             expected_category = cell_types.index(expected_category)
#             cell_type = F.one_hot(
#                 torch.tensor(expected_category),
#                 num_classes=len(cell_types)
#             ).type(torch.float32)
#         else:
#             raise ValueError(f"Expected category '{expected_category}' not found in cell_types.")
#     else:
#         raise KeyError("The table 'square_016um' or column 'Cluster' is not present in sdata.")

#     return tile, cell_type

from torchvision import transforms

def adata_transform(sdata: SpatialData, cell_types) -> Tuple[torch.Tensor, torch.Tensor]:
    if 'Mayo_VisiumHD_789_full_image' in sdata:
        # 提取图像数据
        tile = sdata['Mayo_VisiumHD_789_full_image'].data.compute()
        tile = torch.tensor(tile, dtype=torch.float32)

        # 添加预处理：Resize 和 Normalize
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整到模型需要的尺寸
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 常见的ImageNet标准
        ])
        tile = preprocess(tile)
    else:
        raise KeyError("The image 'Mayo_VisiumHD_789_full_image' is not present in sdata.")
    
    if "square_016um" in sdata and 'Cluster' in sdata["square_016um"].obs.columns:
        expected_category = sdata["square_016um"].obs['Cluster'].values[0]
        if expected_category in cell_types:
            expected_category = cell_types.index(expected_category)
            cell_type = F.one_hot(
                torch.tensor(expected_category),
                num_classes=len(cell_types)
            ).type(torch.float32)
        else:
            raise ValueError(f"Expected category '{expected_category}' not found in cell_types.")
    else:
        raise KeyError("The table 'square_016um' or column 'Cluster' is not present in sdata.")

    return tile, cell_type
