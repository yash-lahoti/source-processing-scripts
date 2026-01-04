"""Script to generate sample Excel files for testing with 50 patients."""
import pandas as pd
import numpy as np
import uuid
from pathlib import Path
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate 50 unique patients
NUM_PATIENTS = 50

# Severity options
SEVERITY_OPTIONS = ["mild", "moderate", "severe"]
SEVERITY_WEIGHTS = [0.2, 0.6, 0.2]  # More moderate cases

# ICD codes
ICD_CODES = {
    "H40.1192": "Primary open-angle glaucoma, unspecified eye, moderate stage",
    "H40.1193": "Primary open-angle glaucoma, unspecified eye, severe stage",
    "H40.1190": "Primary open-angle glaucoma, unspecified eye, mild stage",
    "H40.1191": "Primary open-angle glaucoma, unspecified eye, mild stage",
    "H40.1194": "Primary open-angle glaucoma, unspecified eye, severe stage",
}

# Generate patient UIDs and basic info
patient_uids = [str(uuid.uuid4()) for _ in range(NUM_PATIENTS)]
client_ids = np.random.choice([1, 20, 30, 40], size=NUM_PATIENTS, p=[0.4, 0.3, 0.2, 0.1])
site_ids = np.random.choice([218223, 218224, 218225, 218226], size=NUM_PATIENTS)

# Generate encrypted MRNs (simplified)
def generate_mrn():
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    return ''.join(random.choice(chars) for _ in range(40)) + "="

pat_mrns = [generate_mrn() for _ in range(NUM_PATIENTS)]

# Generate first diagnosis dates (2010-2020)
start_date = datetime(2010, 1, 1)
end_date = datetime(2020, 12, 31)
date_range = (end_date - start_date).days
first_diagnosis_dates = [
    (start_date + timedelta(days=random.randint(0, date_range))).strftime("%m/%d/%Y")
    for _ in range(NUM_PATIENTS)
]

# Cohort data
cohort_data = {
    "client_id": client_ids.tolist(),
    "site_id": site_ids.tolist(),
    "pat_mrn": pat_mrns,
    "patient_uid": patient_uids,
    "first_POAG_diagnosis": first_diagnosis_dates
}

# Generate IOP data with varied measurement counts
iop_records = []
for i, uid in enumerate(patient_uids):
    # Varied measurement counts: most have 2-4, some have 1, few have 5+
    num_measurements = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.1, 0.3, 0.3, 0.2, 0.05, 0.05])
    
    base_date = datetime.strptime(first_diagnosis_dates[i], "%m/%d/%Y")
    
    for j in range(num_measurements):
        # Dates after first diagnosis
        days_after = random.randint(0, 365 * 3)  # Up to 3 years after diagnosis
        contact_date = (base_date + timedelta(days=days_after)).strftime("%m/%d/%Y")
        
        # Generate encounter ID
        enc_id = generate_mrn()
        
        # Realistic IOP values: 10-30 mmHg
        os_iop = np.random.normal(18, 4)
        os_iop = max(10, min(30, int(os_iop)))
        
        od_iop = np.random.normal(20, 4)
        od_iop = max(10, min(30, int(od_iop)))
        
        iop_records.append({
            "client_id": client_ids[i],
            "site_id": site_ids[i],
            "pat_mrn": pat_mrns[i],
            "patient_uid": uid,
            "first_POAG_diagnosis": first_diagnosis_dates[i],
            "PAT_ENC_CSN_ID": enc_id,
            "CONTACT_DATE": contact_date,
            "os_iop": os_iop,
            "od_iop": od_iop
        })

iop_data = pd.DataFrame(iop_records)

# Generate diagnosis data
diagnosis_records = []
for i, uid in enumerate(patient_uids):
    # Most patients have 1-2 diagnoses
    num_diagnoses = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
    
    base_date = datetime.strptime(first_diagnosis_dates[i], "%m/%d/%Y")
    
    for j in range(num_diagnoses):
        # Diagnosis dates around first diagnosis
        days_offset = random.randint(-30, 365)
        dx_date = (base_date + timedelta(days=days_offset)).strftime("%m/%d/%Y")
        
        # Select severity
        severity = np.random.choice(SEVERITY_OPTIONS, p=SEVERITY_WEIGHTS)
        
        # Select ICD code based on severity
        if severity == "mild":
            code = "H40.1190"
        elif severity == "moderate":
            code = np.random.choice(["H40.1192", "H40.1191"], p=[0.8, 0.2])
        else:
            code = np.random.choice(["H40.1193", "H40.1194"], p=[0.7, 0.3])
        
        short_desc = ICD_CODES.get(code, ICD_CODES["H40.1192"])
        
        diagnosis_records.append({
            "pat_mrn": pat_mrns[i],
            "POAG_dx_date": dx_date,
            "patient_uid": uid,
            "client_id": client_ids[i],
            "source_severity": severity,
            "code": code,
            "short_desc": short_desc
        })

diagnosis_data = pd.DataFrame(diagnosis_records)

# Generate encounter/vital signs data
enc_records = []
for i, uid in enumerate(patient_uids):
    # Varied encounter counts: 2-4 on average
    num_encounters = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.1, 0.25, 0.3, 0.2, 0.1, 0.05])
    
    base_date = datetime.strptime(first_diagnosis_dates[i], "%m/%d/%Y")
    
    for j in range(num_encounters):
        # Encounter dates after first diagnosis
        days_after = random.randint(0, 365 * 2)  # Up to 2 years
        contact_date = (base_date + timedelta(days=days_after)).strftime("%m/%d/%Y")
        
        # Generate encounter ID
        enc_id = generate_mrn()
        
        # Realistic vital signs
        # BP: systolic 100-160, diastolic 60-100
        bp_systolic = np.random.normal(130, 15)
        bp_systolic = max(100, min(160, int(bp_systolic)))
        
        bp_diastolic = np.random.normal(80, 10)
        bp_diastolic = max(60, min(100, int(bp_diastolic)))
        
        # Pulse: 50-100 bpm
        pulse = np.random.normal(72, 10)
        pulse = max(50, min(100, int(pulse)))
        
        enc_records.append({
            "patient_uid": uid,
            "pat_enc_csn_id": enc_id,
            "contact_date": contact_date,
            "pulse": pulse,
            "bp_systolic": bp_systolic,
            "bp_diastolic": bp_diastolic
        })

enc_data = pd.DataFrame(enc_records)

# Create sample_data directory
sample_dir = Path("sample_data")
sample_dir.mkdir(exist_ok=True)

# Write Excel files
pd.DataFrame(cohort_data).to_excel(sample_dir / "cohort.xlsx", index=False)
iop_data.to_excel(sample_dir / "iop.xlsx", index=False)
diagnosis_data.to_excel(sample_dir / "diagnosis.xlsx", index=False)
enc_data.to_excel(sample_dir / "enc.xlsx", index=False)

print(f"Sample data files created in sample_data/")
print(f"  - Cohort: {NUM_PATIENTS} patients")
print(f"  - IOP: {len(iop_records)} measurements")
print(f"  - Diagnosis: {len(diagnosis_records)} records")
print(f"  - Encounters: {len(enc_records)} records")
