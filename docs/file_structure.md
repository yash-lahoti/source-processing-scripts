# Excel Files Structure

This document outlines the Excel files and their column structures based on the data sections identified.

## File Structure

### 1. cohort.xlsx
**Description:** Patient cohort information with first POAG diagnosis dates

**Columns:**
- `client_id` (Integer)
- `site_id` (Integer)
- `pat_mrn` (String - Encrypted/Hashed MRN)
- `patient_uid` (String - UUID)
- `first_POAG_diagnosis` (Date - Format: M/D/YYYY)

---

### 2. iop.xlsx
**Description:** Intraocular pressure (IOP) measurements for patients

**Columns:**
- `client_id` (Integer)
- `site_id` (Integer)
- `pat_mrn` (String - Encrypted/Hashed MRN)
- `patient_uid` (String - UUID)
- `first_POAG_diagnosis` (Date - Format: M/D/YYYY)
- `PAT_ENC_CSN_ID` (String - Encrypted/Hashed Encounter ID)
- `CONTACT_DATE` (Date - Format: M/D/YYYY)
- `os_iop` (Integer - Left eye intraocular pressure)
- `od_iop` (Integer - Right eye intraocular pressure)

---

### 3. diagnosis.xlsx
**Description:** POAG diagnosis records with severity and ICD codes

**Columns:**
- `pat_mrn` (String - Encrypted/Hashed MRN)
- `POAG_dx_date` (Date - Format: M/D/YYYY)
- `patient_uid` (String - UUID)
- `client_id` (Integer)
- `source_severity` (String - e.g., "moderate")
- `code` (String - ICD code, e.g., "H40.1192")
- `short_desc` (String - Diagnosis description)

---

### 4. enc.xlsx
**Description:** Patient encounter records with vital signs

**Columns:**
- `patient_uid` (String - UUID)
- `pat_enc_csn_id` (String - Encrypted/Hashed Encounter ID)
- `contact_date` (Date - Format: M/D/YYYY)
- `pulse` (Integer - Heart rate)
- `bp_systolic` (Integer - Systolic blood pressure)
- `bp_diastolic` (Integer - Diastolic blood pressure)

---

## Notes

- All dates appear to be in M/D/YYYY format
- Patient identifiers (`pat_mrn`, `PAT_ENC_CSN_ID`, `pat_enc_csn_id`) appear to be encrypted or hashed values
- `patient_uid` fields contain UUIDs for linking records across files
- The data relates to Primary Open-Angle Glaucoma (POAG) patient records

