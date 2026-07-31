"""MVTec dataset solvers for anomaly detection."""

import argparse
import json
from pathlib import Path


class MVTecSolver:
    """Generate meta.json for a multi-class MVTec-like dataset."""

    CLSNAMES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]
    IMG_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}

    def __init__(self, root='data/mvtec', class_names=None):
        self.root = Path(root)
        self.meta_path = self.root / 'meta.json'
        self.class_names = class_names

    @classmethod
    def _is_image(cls, path):
        return path.is_file() and path.suffix.lower() in cls.IMG_EXTENSIONS

    def _relative(self, path):
        return path.relative_to(self.root).as_posix()

    def _discover_classes(self):
        if not self.root.exists():
            raise FileNotFoundError(f'Dataset root not found: {self.root}')

        class_names = sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / 'train').is_dir() and (path / 'test').is_dir()
        )
        if not class_names:
            raise FileNotFoundError(
                f'No classes found under {self.root}. Each class directory must contain train/ and test/.'
            )
        return class_names

    def _find_mask(self, class_dir, defect_name, image_path):
        mask_dir = class_dir / 'ground_truth' / defect_name
        if not mask_dir.is_dir():
            raise FileNotFoundError(f'Mask directory not found: {mask_dir}')

        masks = [path for path in mask_dir.iterdir() if self._is_image(path)]
        expected_stems = {image_path.stem, f'{image_path.stem}_mask'}
        matches = [path for path in masks if path.stem in expected_stems]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f'Multiple masks found for {image_path}: {matches}')
        raise FileNotFoundError(
            f'No mask found for {image_path}. Expected a mask named '
            f'{image_path.stem}.* or {image_path.stem}_mask.* in {mask_dir}.'
        )

    def _collect_phase(self, class_name, phase):
        class_dir = self.root / class_name
        phase_dir = class_dir / phase
        if not phase_dir.is_dir():
            raise FileNotFoundError(f'Directory not found: {phase_dir}')

        class_info = []
        for defect_dir in sorted(path for path in phase_dir.iterdir() if path.is_dir()):
            defect_name = defect_dir.name
            is_abnormal = defect_name != 'good'
            images = sorted(path for path in defect_dir.iterdir() if self._is_image(path))
            for image_path in images:
                mask_path = self._find_mask(class_dir, defect_name, image_path) if is_abnormal else None
                class_info.append(dict(
                    img_path=self._relative(image_path),
                    mask_path=self._relative(mask_path) if mask_path else '',
                    cls_name=class_name,
                    specie_name=defect_name,
                    anomaly=1 if is_abnormal else 0,
                ))
        return class_info

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        class_names = self.class_names or self._discover_classes()
        for cls_name in class_names:
            for phase in ['train', 'test']:
                cls_info = self._collect_phase(cls_name, phase)
                info[phase][cls_name] = cls_info
                if phase == 'test':
                    anomaly_samples += sum(item['anomaly'] == 1 for item in cls_info)
                    normal_samples += sum(item['anomaly'] == 0 for item in cls_info)
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('meta_path', self.meta_path)
        print('classes', len(class_names), class_names)
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)


class SingleClassMVTecSolver:
    """Generate meta.json for a single-class MVTec-like directory.

    Expected layout:
        root/
            train/good/*
            test/good/*
            test/<defect_name>/*
            ground_truth/<defect_name>/*
    """

    IMG_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}

    def __init__(self, root='data', class_name='foreign_object'):
        self.root = Path(root)
        self.class_name = class_name
        self.meta_path = self.root / 'meta.json'

    @classmethod
    def _is_image(cls, path):
        return path.is_file() and path.suffix.lower() in cls.IMG_EXTENSIONS

    def _relative(self, path):
        return path.relative_to(self.root).as_posix()

    def _find_mask(self, defect_name, image_path):
        mask_dir = self.root / 'ground_truth' / defect_name
        if not mask_dir.exists():
            raise FileNotFoundError(f'Mask directory not found: {mask_dir}')

        masks = sorted(path for path in mask_dir.iterdir() if self._is_image(path))
        stem_matches = [path for path in masks if path.stem == image_path.stem]
        if stem_matches:
            return stem_matches[0]

        if len(masks) == 1:
            return masks[0]

        raise FileNotFoundError(
            f'No mask with stem "{image_path.stem}" found in {mask_dir}. '
            'Please keep mask filenames aligned with test anomaly images.'
        )

    def _collect_phase(self, phase):
        phase_dir = self.root / phase
        if not phase_dir.exists():
            raise FileNotFoundError(f'Directory not found: {phase_dir}')

        cls_info = []
        for specie_dir in sorted(path for path in phase_dir.iterdir() if path.is_dir()):
            specie_name = specie_dir.name
            is_abnormal = specie_name != 'good'
            images = sorted(path for path in specie_dir.iterdir() if self._is_image(path))
            for img_path in images:
                mask_path = self._find_mask(specie_name, img_path) if is_abnormal else ''
                cls_info.append(dict(
                    img_path=self._relative(img_path),
                    mask_path=self._relative(mask_path) if is_abnormal else '',
                    cls_name=self.class_name,
                    specie_name=specie_name,
                    anomaly=1 if is_abnormal else 0,
                ))
        return cls_info

    def run(self):
        info = dict(train={}, test={})
        info['train'][self.class_name] = self._collect_phase('train')
        info['test'][self.class_name] = self._collect_phase('test')

        normal_samples = sum(item['anomaly'] == 0 for item in info['test'][self.class_name])
        anomaly_samples = sum(item['anomaly'] == 1 for item in info['test'][self.class_name])
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('meta_path', self.meta_path)
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate MVTec meta.json')
    parser.add_argument('--root', type=str, default='data/mvtec', help='dataset root')
    parser.add_argument('--single_class', action='store_true', help='root is a single MVTec-like class')
    parser.add_argument('--class_name', type=str, default='foreign_object', help='class name for --single_class')
    parser.add_argument('--classes', nargs='+', help='optional class names; default: auto-discover all classes')
    args = parser.parse_args()

    if args.single_class:
        runner = SingleClassMVTecSolver(root=args.root, class_name=args.class_name)
    else:
        runner = MVTecSolver(root=args.root, class_names=args.classes)
    runner.run()
