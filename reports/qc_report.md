# ECG Basic QC Report

- Scan root: Data
- Manifest records: 728
- .downloading files: 0

## Status counts

|status|count|
|---|---:|
|usable|616|
|review|111|
|reject|1|

## Checks performed

Parseability; incomplete downloads; duration; timestamp monotonicity; obvious gaps; required leads/channels; all-zero or near-constant; truncation risk; d12 12 leads; d6 six limb leads; scale header sampling rate/unit/channels; NaN/Inf.

Duration triage: <30 sec = review (structural parse/lead failures = reject); 30-90 sec = review; >90 sec = usable candidate. A single large ECG peak is not a rejection criterion. V1-V6 are not required for d6.

## Reason counts

|reason|count|
|---|---:|
|obvious time gaps: 1|55|
|duration <30 sec|32|
|duration 30-90 sec|8|
|timestamps not strictly increasing|6|
|obvious time gaps: 1; duration <30 sec|4|
|obvious time gaps: 1; duration 30-90 sec|4|
|signal ZIP cannot be parsed; duration <30 sec; no numeric samples; sampling rate unavailable|1|
|timestamps not strictly increasing; duration <30 sec|1|
|timestamps not strictly increasing; obvious time gaps: 1|1|

Full records are in reject_records.csv and review_records.csv; normalized status is in raw_record_manifest.csv.
