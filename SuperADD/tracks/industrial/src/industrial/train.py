# Anaconda 环境下没有 uv workspace, 需要手动把 industrial 包的两个目录
# (tracks/industrial/src 与 utils) 加入 sys.path, Python 才能把两个目录下的
# industrial/ 子目录合并为同一个 namespace package
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent              # .../tracks/industrial/src/industrial
_PROJECT_ROOT = _HERE.parents[3]                      # 项目根目录
sys.path.insert(0, str(_HERE.parents[0]))             # tracks/industrial/src
sys.path.insert(0, str(_PROJECT_ROOT / "utils"))      # utils

from industrial.dataset import MVTecAD2Dataset
from industrial.model import SuperADD
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from industrial.paths import get_root_config, get_dataset_path, get_model_path


def main() -> None:

    config = get_root_config()

    dataset_path = get_dataset_path('mvtec_ad_2')

    for category in config['categories']:
        print(f'processing category {category}')

        # reproducible seed
        np.random.seed(42)

        train_data = MVTecAD2Dataset(dataset_path, category, 'train', transform=transforms.ToTensor())

        # access subsampled fraction of train data
        sample_indices = list(range(0, len(train_data), config['train_fraction']))
        train_images = [train_data[i].image for i in tqdm(sample_indices, desc=f'loading {category} train images', file=sys.stdout)]

        # create model
        model = SuperADD(backbone=config['backbone'],
                         layers=config['layers'],
                         resize_factor=config['patch_size'] / 1024,
                         patch_size=config['patch_size'],
                         patch_overlap=config['patch_overlap'],
                         max_database_size=config['max_database_size'],
                         threshold_fraction=config['threshold_fraction'],
                         subsampling_iterations=config['subsampling_iterations'],
                         threshold_percentile=config['threshold_percentile'],
                         threshold_factor=config['threshold_factor'],
                         evaluation_downscale=config['evaluation_downscale'],
                         closing_radius=config['closing_radius'],
                         closing_angles=config['closing_angles'],
                         closing_lower_threshold=config['closing_lower_threshold'],
                         binary_erosion=config['binary_erosion'],
                         brightness_augmentation=config['brightness_augmentation'],
                         device='cuda')

        # train model on anomaly free images
        model.train(train_images)

        # store model to disk
        model_path = get_model_path(category)
        model.to_disk(model_path)

if __name__ == "__main__":
    main()
