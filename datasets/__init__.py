from .custom_dataset import CustomDataset
from .custom_test import CustomDatasetTest
from .messy_dataset import MESSYDataset

__datasets__ = {
    "custom_dataset_full": CustomDataset,
    "custom_dataset_test": CustomDatasetTest,
    "messy_table": MESSYDataset
}
