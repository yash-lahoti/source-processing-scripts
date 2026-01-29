"""Script to generate test aggregated dataset matching the format from the image."""
import pandas as pd
import numpy as np
import json
import uuid
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Generate 15 patients matching the image format
NUM_PATIENTS = 15

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

patient_uids = [generate_short_uid() for _ in range(NUM_PATIENTS)]

# Generate test data
test_data = []

for i, uid in enumerate(patient_uids):
    # Generate varied measurement counts (1-4 measurements per patient)
    num_measurements = np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2])
    
    # Blood pressure - realistic ranges
    bp_diastolic_list = []
    bp_systolic_list = []
    for _ in range(num_measurements):
        bp_diastolic_list.append(int(np.random.normal(75, 10)))
        bp_diastolic_list[-1] = max(50, min(100, bp_diastolic_list[-1]))
        bp_systolic_list.append(int(np.random.normal(130, 15)))
        bp_systolic_list[-1] = max(100, min(180, bp_systolic_list[-1]))
    
    # IOP measurements - realistic ranges (10-30 mmHg)
    od_iop_list = []
    os_iop_list = []
    for _ in range(num_measurements):
        od_iop_list.append(int(np.random.normal(20, 4)))
        od_iop_list[-1] = max(10, min(30, od_iop_list[-1]))
        os_iop_list.append(int(np.random.normal(18, 4)))
        os_iop_list[-1] = max(10, min(30, os_iop_list[-1]))
    
    # Pulse - realistic ranges (50-100 bpm)
    pulse_list = []
    for _ in range(num_measurements):
        pulse_val = int(np.random.normal(72, 10))
        pulse_list.append(max(50, min(100, pulse_val)))
    
    # Severity - match distribution from image
    severity_list = []
    code_list = []
    short_desc_list = []
    
    # Some patients have multiple severity entries
    num_severities = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
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

# Create DataFrame
df = pd.DataFrame(test_data)

# Create test_data directory
test_dir = Path("test_data")
test_dir.mkdir(exist_ok=True)

# Save as CSV
output_file = test_dir / "aggregated_test_patients.csv"
df.to_csv(output_file, index=False)

print(f"✓ Test dataset created: {output_file}")
print(f"  - Patients: {len(df)}")
print(f"  - Columns: {len(df.columns)}")
print(f"  - Format: JSON string arrays")
print(f"\nSample data:")
print(df.head(3).to_string())
print(f"\n✓ Saved to: {output_file}")



