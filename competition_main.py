"""This component-only branch deliberately has no full-ECG competition entry."""

if __name__ == "__main__":
    raise SystemExit(
        "B5-fast exports the frozen fast component only. "
        "Use: python fast_main.py --help. "
        "Full ECG submission requires a future explicit slow component; "
        "legacy slow CNN/context inference is disabled at this entry."
    )
