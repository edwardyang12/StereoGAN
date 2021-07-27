from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .custom_dataset import CustomDataset
from .custom_test import CustomDatasetTest

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "custom_dataset_full": CustomDataset,
    "custom_dataset_test": CustomDatasetTest,

}
