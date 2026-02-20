# Dataset Research Findings

## 1. Longitudinal EHR (Structured Features)
- **Primary Source:** MIMIC-IV (Medical Information Mart for Intensive Care)
- **Vendors:** Epic Systems, Cerner (Oracle Health)
- **Key Dataset:** Epic Cosmos (aggregate dataset)
- **Citation (MIMIC-IV):** Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T. J., Sun, B., Lodi, S., McCague, N., Marco, M., Steven, H., & Celi, L. A. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10(1), 1. https://doi.org/10.1038/s41597-022-01899-x

## 2. Time-Series Vitals (LSTM Trend Learning)
- **Sources:** Laboratory Information Systems (LIS), Patient Monitoring Systems (e.g., Philips IntelliVue, GE Carescape).
- **Public Proxy:** MIMIC-IV ICU module (contains high-frequency vitals).
- **Note:** LIS typically provides discrete lab results, while Monitoring Systems provide continuous waveforms/vitals.

## 3. Clinical Embeddings (NLP Risk Scoring)
- **Primary Model:** Bio_ClinicalBERT
- **Source:** Hugging Face (emilyalsentzer/Bio_ClinicalBERT)
- **Citation:** Alsentzer, E., Murphy, J. R., Boag, W., Weng, W. H., Jindi, D., Johnson, A. E. W., & McDermott, M. B. A. (2019). Publicly Available Clinical BERT Embeddings. *Proceedings of the 2nd Clinical Natural Language Processing Workshop*, 72–78. https://doi.org/10.18653/v1/W19-1909

## 4. Culture Results (Training Labels)
- **Source:** Laboratory Information Systems (LIS) - Ground Truth.
- **Context:** Microbiology data in EHRs (like the `microbiologyevents` table in MIMIC) serves as the gold standard for infection/culture labels.

## 5. Validation Set (MLOps Deployment Gate)
- **Source:** Partner Hospital (External Validation).
- **Context:** Represents a "Hold-out" or "External" validation set of 1,000+ records used for model performance verification before clinical deployment.
