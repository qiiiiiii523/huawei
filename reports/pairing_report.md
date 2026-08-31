# Task pairing report

Pairing uses exact normalized subject_id (HNU) for cross-device matching. No nearest-time, file-order, or guessed matching is used. d6 and d12 do not need identical timestamps.

## Task 1: watch single-lead ECG -> d12

|pair_status|count|
|---|---:|
|paired|117|
|review|4|
|unmatched|9|

The watch ECG groupid and externalid are retained. Four records have non-HNU/unreliable subject IDs and require manual ID confirmation; nine records with HNU IDs (seven distinct HNU subjects) have no d12 target.

## Task 2: d6 / body-scale d6 -> d12

|input_type|paired|review|unmatched|
|---|---:|---:|---:|
|ecg_machine_d6|100|0|0|
|body_scale_d6|34|0|0|

## Manual review

The 13 unresolved Task 1 cases are listed in metadata/id_mapping_review.csv. Only pair_status=paired is a confirmed pair. review and unmatched records must not be used as confirmed training pairs. No train/validation split was created.

Outputs: metadata/pair_manifest_task1.csv, metadata/pair_manifest_task2.csv, metadata/id_mapping_review.csv.
