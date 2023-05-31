import json, os, sys

"""
Feature Documentation of Numerai v4.1 dataset

Dimensionality Reduction based on correlation patterns for features 

We observe groups of highly correalated features (>0.85) consistenctly in each era at regular postitions in the data. 
A good hypothesis is by creating features from different lookbacks 

"""


## This is the path of the json file when copied in the docker image
filepath = "/pythor/app/numerai/numerai-v4.1_features.json"
with open(filepath, "r") as f:
    NUMERAI_SUNSHINE_JSON = json.load(f)

NUMERAI_SUNSHINE_TARGETS_ALL = NUMERAI_SUNSHINE_JSON["targets"][1:]
NUMERAI_SUNSHINE_TARGETS = NUMERAI_SUNSHINE_TARGETS_ALL

if False:
    NUMERAI_SUNSHINE_TARGETS = [
        "target_cyrus_v4_20",
        "target_xerxes_v4_20",
        "target_sam_v4_20",
        "target_caroline_v4_20",
        "target_ralph_v4_20",
        "target_nomi_v4_20",
        "target_victor_v4_20",
        "target_tyler_v4_20",
        "target_waldo_v4_20",
        "target_jerome_v4_20",
    ]


NUMERAI_SUNSHINE_FEATURES_V2 = NUMERAI_SUNSHINE_JSON["feature_sets"][
    "v2_equivalent_features"
]
NUMERAI_SUNSHINE_FEATURES_V4 = list(NUMERAI_SUNSHINE_JSON["feature_stats"].keys())

## Smart Beta Factors
SMART_BETAS = NUMERAI_SUNSHINE_FEATURES_V4[1040:1181]

## Structured Features
v4_features_lookback = dict()
for i in range(0, 5):
    v4_features_lookback[i] = NUMERAI_SUNSHINE_FEATURES_V4[208 * i : 208 * (i + 1)]
sunshine_features_lookback = dict()
for i in range(0, 5):
    sunshine_features_lookback[i] = NUMERAI_SUNSHINE_FEATURES_V4[1181 + i :: 5]
NUMERAI_SUNSHINE_FEATURES_LOOKBACK = dict()
for i in range(0, 5):
    NUMERAI_SUNSHINE_FEATURES_LOOKBACK[i] = (
        SMART_BETAS + v4_features_lookback[i] + sunshine_features_lookback[i]
    )


"""
Feature Documentation of Numerai v4.2 dataset (Rain)


"""
