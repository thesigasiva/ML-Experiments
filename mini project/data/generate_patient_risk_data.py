import os
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).with_name('patient_risk_dataset.csv')

rng = np.random.default_rng(42)

rows = []
for _ in range(800):
    age = int(rng.integers(18, 85))
    gender = rng.choice(['Male', 'Female', 'Other'])
    bmi = float(rng.normal(27.5, 5.5))
    heart_rate = float(rng.normal(78, 12))
    sys_bp = float(rng.normal(128, 16))
    diastolic_bp = float(rng.normal(82, 10))
    oxygen_saturation = float(rng.normal(96.5, 3.2))
    temperature = float(rng.normal(98.4, 1.1))
    hemoglobin = float(rng.normal(12.8, 2.0))
    sodium = float(rng.normal(138, 4.0))
    potassium = float(rng.normal(4.1, 0.6))
    creatinine = float(rng.normal(1.0, 0.5))
    wbc = float(rng.normal(8.3, 2.2))
    previous_admissions = int(rng.poisson(1.2))
    icu_stays = int(rng.poisson(0.4))
    chronic_condition_count = int(rng.integers(0, 6))
    diabetes = int(rng.choice([0, 1], p=[0.72, 0.28]))
    hypertension = int(rng.choice([0, 1], p=[0.68, 0.32]))
    copd = int(rng.choice([0, 1], p=[0.83, 0.17]))
    smoking = int(rng.choice([0, 1], p=[0.75, 0.25]))
    major_surgery = int(rng.choice([0, 1], p=[0.82, 0.18]))
    admission_severity = rng.choice(['Low', 'Moderate', 'High'], p=[0.35, 0.45, 0.20])

    risk_score = (
        0.12 * max(age - 50, 0) / 40
        + 0.08 * max(bmi - 25, 0) / 20
        + 0.12 * max(heart_rate - 80, 0) / 30
        + 0.08 * max(sys_bp - 120, 0) / 40
        + 0.10 * max(100 - oxygen_saturation, 0) / 15
    )

    icu_score = (
        0.12 * max(age - 60, 0) / 30
        + 0.10 * max(heart_rate - 90, 0) / 30
        + 0.18 * max(100 - oxygen_saturation, 0) / 15
        + 0.12 * max(sys_bp - 140, 0) / 40
    )

    readmission_risk = 1 if risk_score > 0.17 else 0
    icu_emergency_risk = 1 if icu_score > 0.13 else 0

    rows.append({
        'age': age,
        'gender': gender,
        'bmi': round(bmi, 2),
        'heart_rate': round(heart_rate, 2),
        'sys_bp': round(sys_bp, 2),
        'diastolic_bp': round(diastolic_bp, 2),
        'oxygen_saturation': round(oxygen_saturation, 2),
        'temperature': round(temperature, 2),
        'hemoglobin': round(hemoglobin, 2),
        'sodium': round(sodium, 2),
        'potassium': round(potassium, 2),
        'creatinine': round(creatinine, 2),
        'wbc': round(wbc, 2),
        'previous_admissions': previous_admissions,
        'icu_stays': icu_stays,
        'chronic_condition_count': chronic_condition_count,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'copd': copd,
        'smoking': smoking,
        'major_surgery': major_surgery,
        'admission_severity': admission_severity,
        'readmission_risk': readmission_risk,
        'icu_emergency_risk': icu_emergency_risk,
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)
print(f'Saved synthetic patient risk dataset to {OUTPUT_PATH}')
