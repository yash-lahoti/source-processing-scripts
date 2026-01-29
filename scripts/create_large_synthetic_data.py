"""Script to generate large synthetic aggregated dataset (10K patients) for testing."""
import pandas as pd
import numpy as np
import json
import uuid
from pathlib import Path
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Generate 10,000 patients
NUM_PATIENTS = 10000

# Severity options matching the image
SEVERITY_OPTIONS = ["mild", "moderate", "severe", "unspecified", "indeterminate"]

# ICD codes
ICD_CODES = {
    "H40.1190": "Primary open-angle glaucoma, unspecified eye, mild stage",
    "H40.1191": "Primary open-angle glaucoma, unspecified eye, mild stage",
    "H40.1192": "Primary open-angle glaucoma, unspecified eye, moderate stage",
    "H40.1193": "Primary open-angle glaucoma, unspecified eye, severe stage",
    "H40.1194": "Primary open-angle glaucoma, unspecified eye, severe stage",
    "H40.10X": "Primary open-angle glaucoma, unspecified",
}

# Generate patient UIDs (shortened format like in image: ab3e9f17-)
def generate_short_uid():
    """Generate shortened UUID format like ab3e9f17-"""
    full_uuid = str(uuid.uuid4())
    # Take first 8 characters and add hyphen
    return full_uuid[:8] + "-"

print(f"Generating {NUM_PATIENTS} patients...")
print("=" * 60)

# Generate patient UIDs
print("Generating patient UIDs...")
patient_uids = [generate_short_uid() for _ in tqdm(range(NUM_PATIENTS), desc="UIDs", unit="patient")]

# Pre-generate measurement counts for all patients (vectorized)
print("Generating measurement counts...")
num_measurements_list = np.random.choice([1, 2, 3, 4, 5, 6], size=NUM_PATIENTS, p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05])

# Pre-generate severity counts
num_severities_list = np.random.choice([1, 2, 3], size=NUM_PATIENTS, p=[0.5, 0.3, 0.2])

# Generate test data
print("Generating patient data...")
test_data = []

for i in tqdm(range(NUM_PATIENTS), desc="Patients", unit="patient"):
    uid = patient_uids[i]
    num_measurements = num_measurements_list[i]
    
    # Blood pressure - realistic ranges (vectorized generation)
    bp_diastolic_raw = np.random.normal(75, 10, num_measurements)
    bp_diastolic_list = np.clip(bp_diastolic_raw, 50, 100).astype(int).tolist()
    
    bp_systolic_raw = np.random.normal(130, 15, num_measurements)
    bp_systolic_list = np.clip(bp_systolic_raw, 100, 180).astype(int).tolist()
    
    # IOP measurements - realistic ranges (10-30 mmHg)
    od_iop_raw = np.random.normal(20, 4, num_measurements)
    od_iop_list = np.clip(od_iop_raw, 10, 30).astype(int).tolist()
    
    os_iop_raw = np.random.normal(18, 4, num_measurements)
    os_iop_list = np.clip(os_iop_raw, 10, 30).astype(int).tolist()
    
    # Pulse - realistic ranges (50-100 bpm)
    pulse_raw = np.random.normal(72, 10, num_measurements)
    pulse_list = np.clip(pulse_raw, 50, 100).astype(int).tolist()
    
    # Severity - match distribution from image
    severity_list = []
    code_list = []
    short_desc_list = []
    
    num_severities = num_severities_list[i]
    for _ in range(num_severities):
        severity = np.random.choice(SEVERITY_OPTIONS, p=[0.2, 0.3, 0.2, 0.2, 0.1])
        severity_list.append(severity)
        
        # Select code based on severity
        if severity == "mild":
            code = np.random.choice(["H40.1190", "H40.1191"], p=[0.7, 0.3])
        elif severity == "moderate":
            code = np.random.choice(["H40.1192", "H40.1191"], p=[0.8, 0.2])
        elif severity == "severe":
            code = np.random.choice(["H40.1193", "H40.1194"], p=[0.7, 0.3])
        else:
            code = "H40.10X"
        
        code_list.append(code)
        short_desc_list.append(ICD_CODES.get(code, ICD_CODES["H40.1192"]))
    
    # Format as JSON string arrays (matching the CSV format)
    row = {
        'patient_uid': uid,
        'bp_diastolic': json.dumps(bp_diastolic_list),
        'bp_systolic': json.dumps(bp_systolic_list),
        'code': json.dumps(code_list),
        'od_iop': json.dumps(od_iop_list),
        'os_iop': json.dumps(os_iop_list),
        'pulse': json.dumps(pulse_list),
        'short_desc': json.dumps(short_desc_list),
        'source_severity': json.dumps(severity_list)
    }
    
    test_data.append(row)

print("\nCreating DataFrame...")
# Create DataFrame
df = pd.DataFrame(test_data)

# Create test_data directory
test_dir = Path("test_data")
test_dir.mkdir(exist_ok=True)

# Save as CSV
output_file = test_dir / "aggregated_10k_patients.csv"
print(f"\nSaving to {output_file}...")
df.to_csv(output_file, index=False)

print(f"\n✓ Dataset created: {output_file}")
print(f"  - Patients: {len(df):,}")
print(f"  - Columns: {len(df.columns)}")
print(f"  - Format: JSON string arrays")
print(f"  - File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
print(f"\nSample data (first 3 patients):")
print(df.head(3).to_string())
print(f"\n✓ Saved to: {output_file}")



