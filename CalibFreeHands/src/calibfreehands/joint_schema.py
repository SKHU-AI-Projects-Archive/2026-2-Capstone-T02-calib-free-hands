"""Canonical hand joint names shared by data exports and visualizations."""

JOINT_NAMES = ("wrist", *(f"joint_{index}" for index in range(1, 21)))
