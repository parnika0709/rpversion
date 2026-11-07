import numpy as np
from collections import Counter
from braintumor.config import config
from braintumor.data.dataset import BratsDataset
from braintumor.data.slicing import SlicingDataset
from braintumor.data.splits import discover_and_split
from braintumor.data.preprocessing import train_transforms
from braintumor.models.attention_unet_2d import MVP_MODALITIES

train_cases, _, _ = discover_and_split(
    config.data_root,
    split_file=config.split_file,
)

modalities = list(MVP_MODALITIES)

train_ds = BratsDataset(
    train_cases,
    modalities,
    transform=train_transforms(modalities),
)

slice_ds = SlicingDataset(
    train_ds,
    training=True,
    samples_per_volume=8,
    foreground_probability=0.7,
)

counts = Counter()

print("Inspecting training slices...")

for i in range(len(slice_ds)):
    labels = slice_ds[i]["label"].numpy()
    values, frequencies = np.unique(labels, return_counts=True)

    for value, frequency in zip(values, frequencies):
        counts[int(value)] += int(frequency)

total = sum(counts.values())

print("\nTRAINING PIXEL DISTRIBUTION")
for label in range(4):
    count = counts[label]
    percentage = 100 * count / total if total else 0
    print(f"Class {label}: {count:,} pixels ({percentage:.4f}%)")

print(f"\nTotal pixels: {total:,}")
