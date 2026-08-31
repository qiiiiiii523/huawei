# Final subject split report

## Decision

`metadata/subject_split.csv` is a byte-for-byte copy of the user-confirmed `metadata/subject_split_proposed.csv` (SHA-256: `3f19f510f8247d9a3e7c48038eed013718a3d7e9404a812e802d48ec345c0563`). The proposed file was retained. No new randomization was performed.

The split uses only normalized `subject_id` and fixed random seed **42**. Every record and every device for one subject is assigned to the same split. The synchronized `split` column in the manifest and both pair manifests is derived only from this subject mapping.

## Split summary

| Split | Subjects | Manifest records |
|---|---:|---:|
| train | 88 | 573 |
| validation | 22 | 155 |

No `subject_id` conflict was found: each of the 110 subjects maps to exactly one of `train` or `validation`.

## Usable paired samples

A usable paired sample requires `pair_status=paired` and usable input and target quality in the existing pair manifests.

| Split | watch ECG -> d12 | machine d6 -> d12 | body-scale d6 -> d12 |
|---|---:|---:|---:|
| train | 83 | 80 | 29 |
| validation | 21 | 20 | 5 |
| total | 104 | 100 | 34 |

## Scope and source-of-truth rule

`subject_split.csv` is now the unique split authority for all subsequent tasks. The proposed file remains as the retained confirmation source and is identical to the formal file. This step did not modify raw data, delete records, alter `pair_status`, cut windows, preprocess signals, or train a model.
