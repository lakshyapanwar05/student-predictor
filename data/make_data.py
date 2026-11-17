import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 500
study_hours = rng.uniform(0, 10, n).round(1)
attendance = rng.uniform(40, 100, n).round(0)
previous_marks = rng.normal(60, 15, n).clip(0, 100).round(0)

score = 0.35 * study_hours + 0.04 * attendance + 0.05 * previous_marks + rng.normal(0, 0.8, n)
passed = (score > np.percentile(score, 35)).astype(int)

pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_marks": previous_marks,
    "passed": passed,
}).to_csv("data/students.csv", index=False)
