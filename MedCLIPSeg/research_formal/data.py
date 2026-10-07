"""Native BGR224 preprocessing; paired reflections happen in the training loss."""
from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator


def make_dataset(cfg, split, records):
    transform = RandomGenerator([224, 224]) if split == 'Train' else ValGenerator([224, 224])
    return DatasetSegmentation(getattr(cfg.DATASET, split.upper()+'_PATH'), cfg.DATASET.NAME,
                               records, transform, image_size=cfg.DATASET.SIZE)
