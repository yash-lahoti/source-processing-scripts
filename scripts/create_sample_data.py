"""Script to generate sample Excel files for testing."""
import pandas as pd
from pathlib import Path

# Sample data
cohort_data = {
    "client_id": [1, 1, 20],
    "site_id": [218223, 218223, 218224],
    "pat_mrn": ["+++mWO1THf5/6ZwZUuTeJr5Yh5LiZii9HfLvQsG+WwA=", "a1Socasd/tzITdgt1g1J01E8InQp4DqjMkSWdq3FtbU=", "ddvvtiarzx5xs6MfJYi9F3WuDELs8JxNWahmY4EUGDE="],
    "patient_uid": ["ab3e9f17-75d5-4d7d-9867-ab476d64c6ac", "e61e649d-1078-4d9d-b5b6-cd609e8c1b11", "ff2d2be2-abac-47c6-a984-d322ec1215d5"],
    "first_POAG_diagnosis": ["11/1/2013", "8/3/2019", "9/30/2016"]
}

iop_data = {
    "client_id": [1, 1, 1],
    "site_id": [218223, 218223, 218223],
    "pat_mrn": ["+++mWO1THf5/6ZwZUuTeJr5Yh5LiZii9HfLvQsG+WwA=", "+++mWO1THf5/6ZwZUuTeJr5Yh5LiZii9HfLvQsG+WwA=", "a1Socasd/tzITdgt1g1J01E8InQp4DqjMkSWdq3FtbU="],
    "patient_uid": ["ab3e9f17-75d5-4d7d-9867-ab476d64c6ac", "ab3e9f17-75d5-4d7d-9867-ab476d64c6ac", "e61e649d-1078-4d9d-b5b6-cd609e8c1b11"],
    "first_POAG_diagnosis": ["11/1/2013", "11/1/2013", "8/3/2019"],
    "PAT_ENC_CSN_ID": ["7/XgB7dD/RHuQbnx3+/wUpE19JFvFS1YaB6rxRdhVDA=", "8/YhC8eE/SIvRcoy4+/xVqF20KGwGT2ZbC7sySeiWEB=", "9/ZiD9fF/TJwSdpz5+/yWrG31LHxHU3acD8tzTfjXFC="],
    "CONTACT_DATE": ["7/25/2013", "8/15/2013", "9/10/2019"],
    "os_iop": [18, 19, 20],
    "od_iop": [21, 22, 23]
}

diagnosis_data = {
    "pat_mrn": ["ddvvtiarzx5xs6MfJYi9F3WuDELs8JxNWahmY4EUGDE=", "a1Socasd/tzITdgt1g1J01E8InQp4DqjMkSWdq3FtbU=", "a1Socasd/tzITdgt1g1J01E8InQp4DqjMkSWdq3FtbU=", "CBcvguQI8jH5RnguL8Pev11E0Iou+Q9B7/NTrTV1jYY="],
    "POAG_dx_date": ["9/30/2016", "8/3/2019", "1/12/2019", "12/27/2018"],
    "patient_uid": ["ff2d2be2-abac-47c6-a984-d322ec1215d5", "e61e649d-1078-4d9d-b5b6-cd609e8c1b11", "e61e649d-1078-4d9d-b5b6-cd609e8c1b11", "304e9655-ee3c-45e2-a49d-9ec8--dc882be8"],
    "client_id": [20, 1, 1, 1],
    "source_severity": ["moderate", "moderate", "severe", "moderate"],
    "code": ["H40.1192", "H40.1192", "H40.1193", "H40.1192"],
    "short_desc": ["Primary open-angle glaucoma, unspecified eye, moderate stage", "Primary open-angle glaucoma, unspecified eye, moderate stage", "Primary open-angle glaucoma, unspecified eye, severe stage", "Primary open-angle glaucoma, unspecified eye, moderate stage"]
}

enc_data = {
    "patient_uid": ["ab3e9f17-75d5-4d7d-9867-ab476d64c6ac", "ab3e9f17-75d5-4d7d-9867-ab476d64c6ac", "e61e649d-1078-4d9d-b5b6-cd609e8c1b11"],
    "pat_enc_csn_id": ["17EKvJ6dA7/DNOhGWOfWM65tfUt1ziFpwAYVt2S1SLE=", "18FLwK7eB8/EOPhHXPgXN76ugVu2ajGqxBZWu3T2TMF=", "19GMxL8fC9/FPQiIYQhYO87vhWv3bkHryCZXv4U3UNG="],
    "contact_date": ["7/16/2017", "8/20/2017", "9/15/2019"],
    "pulse": [67, 72, 68],
    "bp_systolic": [139, 142, 135],
    "bp_diastolic": [82, 85, 80]
}

# Create sample_data directory
sample_dir = Path("sample_data")
sample_dir.mkdir(exist_ok=True)

# Write Excel files
pd.DataFrame(cohort_data).to_excel(sample_dir / "cohort.xlsx", index=False)
pd.DataFrame(iop_data).to_excel(sample_dir / "iop.xlsx", index=False)
pd.DataFrame(diagnosis_data).to_excel(sample_dir / "diagnosis.xlsx", index=False)
pd.DataFrame(enc_data).to_excel(sample_dir / "enc.xlsx", index=False)

print("Sample data files created in sample_data/")

