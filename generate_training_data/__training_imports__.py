import numpy as np
import argparse
import yaml

from tqdm import tqdm
from typing import Optional
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset, random_split, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


PALETTE      = sns.color_palette("tab10")
GW_BLUE      = "#1f77b4"   # matches typical GW-paper colour for data
SURROGATE_RED = "#d62728"  # surrogate / model prediction
TRUTH_GREEN  = "#2ca02c"
STAGE_COLORS = {1: "#4e79a7", 2: "#f28e2b", 3: "#59a14f"}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "font.size":        12,
        "axes.labelsize":   13,
        "axes.titlesize":   14,
        "legend.fontsize":  11,
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
        "lines.linewidth":  2.0,
    })
