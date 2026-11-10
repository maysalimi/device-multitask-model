
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
"""
StudentLife App Usage Analysis
------------------------------
Predicts RUNNING_TASKS_numRunning from app-usage features.

Expected project layout:
    data/app_usage/running_app_u00.csv

Run from the project root:
    python device.MTP.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "app_usage" / "running_app_u00.csv"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"

TARGET_COLUMN = "RUNNING_TASKS_numRunning"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 3

# Identifiers and raw activity labels are excluded, as in the original
# analysis. Columns are dropped only if they exist in the CSV.
COLUMNS_TO_DROP = [
    "id",
    "device",
    "timestamp",
    "RUNNING_TASKS_baseActivity_mClass",
    "RUNNING_TASKS_baseActivity_mPackage",
    "RUNNING_TASKS_topActivity_mClass",
    "RUNNING_TASKS_topActivity_mPackage",
]


# ---------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and create time-derived features."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the StudentLife dataset and place "
            "running_app_u00.csv in data/app_usage/."
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Derive time features before dropping the original timestamp.
    if "timestamp" in df.columns:
        timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = timestamp.dt.hour
        df["weekday"] = timestamp.dt.weekday

    df = df.drop(
        columns=[column for column in COLUMNS_TO_DROP if column in df.columns]
    )

    # A row without a target cannot be used for supervised learning.
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    return df


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing that is fitted only on training folds."""
