# -*- coding: utf-8 -*-
"""
finalproject_baseballmodel.py
UTF-8 safe version: utility functions for evaluating pitches.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# Pitch configuration: for each pitch type, specify feature weights
# and whether higher values are better (`ascending=True`) or worse.
PITCH_CONFIG = {
    'FF': {
        'release_speed':     {'weight': 0.4, 'ascending': True},
        'release_spin_rate': {'weight': 0.2, 'ascending': True},
        'pfx_z':             {'weight': 0.4, 'ascending': True},
        'pfx_x_abs':         {'weight': 0.0, 'ascending': True}
    },
    'SI': {
        'release_speed':     {'weight': 0.3, 'ascending': True},
        'release_spin_rate': {'weight': 0.0, 'ascending': True},
        'pfx_z':             {'weight': 0.3, 'ascending': False},
        'pfx_x_abs':         {'weight': 0.4, 'ascending': True}
    },
    'FC': {
        'release_speed':     {'weight': 0.4, 'ascending': True},
        'release_spin_rate': {'weight': 0.2, 'ascending': True},
        'pfx_z':             {'weight': 0.1, 'ascending': True},
        'pfx_x_abs':         {'weight': 0.3, 'ascending': True}
    },
    'SL': {
        'release_speed':     {'weight': 0.2, 'ascending': True},
        'release_spin_rate': {'weight': 0.3, 'ascending': True},
        'pfx_z':             {'weight': 0.1, 'ascending': False},
        'pfx_x_abs':         {'weight': 0.4, 'ascending': True}
    },
    'ST': {
        'release_speed':     {'weight': 0.1, 'ascending': True},
        'release_spin_rate': {'weight': 0.3, 'ascending': True},
        'pfx_z':             {'weight': 0.0, 'ascending': True},
        'pfx_x_abs':         {'weight': 0.6, 'ascending': True}
    },
    'CU': {
        'release_speed':     {'weight': 0.1, 'ascending': True},
        'release_spin_rate': {'weight': 0.4, 'ascending': True},
        'pfx_z':             {'weight': 0.5, 'ascending': False},
        'pfx_x_abs':         {'weight': 0.0, 'ascending': True}
    },
    'CH': {
        'release_speed':     {'weight': 0.1, 'ascending': False},
        'release_spin_rate': {'weight': 0.1, 'ascending': False},
        'pfx_z':             {'weight': 0.4, 'ascending': False},
        'pfx_x_abs':         {'weight': 0.4, 'ascending': True}
    },
    'FS': {
        'release_speed':     {'weight': 0.2, 'ascending': True},
        'release_spin_rate': {'weight': 0.3, 'ascending': False},
        'pfx_z':             {'weight': 0.5, 'ascending': False},
        'pfx_x_abs':         {'weight': 0.0, 'ascending': True}
    }
}


def calculate_pitch_pr(df, config_matrix):
    """Calculate pitch quality score per row based on config_matrix."""
    df = df.copy()
    df['pitch_quality_score'] = 0.0
    pitch_types = df['pitch_type'].unique()

    for p_type in pitch_types:
        if p_type not in config_matrix:
            continue
        mask = df['pitch_type'] == p_type
        score = 0.0
        for feature, settings in config_matrix[p_type].items():
            if settings['weight'] > 0:
                pr = df.loc[mask, feature].rank(pct=True, ascending=settings['ascending'])
                score += pr * settings['weight']
        df.loc[mask, 'pitch_quality_score'] = score * 100

    return df


def evaluate_new_pitch(new_pitch, baseline_df, config_matrix):
    """Evaluate a single new pitch against baseline_df using config_matrix.

    new_pitch: dict with keys 'pitch_type', 'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z'
    Returns a numeric score (0-100) or an error message string.
    """
    p_type = new_pitch.get('pitch_type')
    if p_type not in config_matrix:
        return f"Unknown pitch type: {p_type}"

    ref_data = baseline_df[baseline_df['pitch_type'] == p_type]
    if len(ref_data) == 0:
        return f"No reference data for pitch type: {p_type}"

    # ensure pfx_x_abs present
    new_pitch = new_pitch.copy()
    new_pitch['pfx_x_abs'] = abs(new_pitch.get('pfx_x', 0.0))

    score = 0.0
    for feature, settings in config_matrix[p_type].items():
        if settings['weight'] > 0:
            raw_pr = stats.percentileofscore(ref_data[feature], new_pitch[feature], kind='weak')
            final_pr = raw_pr if settings['ascending'] else (100.0 - raw_pr)
            score += (final_pr / 100.0) * settings['weight']

    return round(score * 100, 2)


if __name__ == "__main__":
    # main script: load data, compute scores, and allow simple CLI evaluation
    print("Loading data...")
    df_raw = pd.read_csv('statcast_bat_tracking_2024_2025.csv')
    target_pitches = list(PITCH_CONFIG.keys())

    core_columns = [
        'pitch_type', 'release_speed', 'release_spin_rate',
        'pfx_x', 'pfx_z', 'spin_axis', 'release_pos_x', 'release_pos_z',
        'plate_x', 'plate_z'
    ]

    df_clean = df_raw[df_raw['pitch_type'].isin(target_pitches)][core_columns].dropna().copy()
    df_clean['pfx_x_abs'] = df_clean['pfx_x'].abs()
    print(f"Loaded {len(df_clean)} rows")

    df_scored = calculate_pitch_pr(df_clean, PITCH_CONFIG)
    print("Scored baseline data")

    # simple CLI loop for interactive evaluation
    while True:
        print('\n' + '='*40)
        p_type = input("Pitch type (FF, ST, FS, SI or q to quit): ").upper()
        if p_type == 'Q':
            break
        try:
            speed = float(input("release_speed (mph): "))
            spin = float(input("release_spin_rate (rpm): "))
            pfx_z = float(input("pfx_z: "))
            pfx_x = float(input("pfx_x: "))
        except ValueError:
            print("Invalid input, try again")
            continue

        user_pitch = {
            'pitch_type': p_type,
            'release_speed': speed,
            'release_spin_rate': spin,
            'pfx_x': pfx_x,
            'pfx_z': pfx_z
        }
        try:
            score = evaluate_new_pitch(user_pitch, df_clean, PITCH_CONFIG)
            print(f"Score for {p_type}: {score}")
        except Exception as e:
            print(f"Error evaluating pitch: {e}")
