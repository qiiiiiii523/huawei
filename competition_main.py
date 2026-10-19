"""Windows submission entry. Paths are based on executable location, not cwd."""
from __future__ import annotations

import multiprocessing
import sys


if __name__=='__main__':
    multiprocessing.freeze_support()
    try:
        from baselines.B5_final.submission import main
        main()
    except Exception as exc:
        from pathlib import Path
        import traceback
        root=Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
        details=traceback.format_exc()
        try: (root/'submission_error.log').write_text(details,encoding='utf-8')
        except OSError: pass
        print('Submission failed: '+str(exc)+'\nSee submission_error.log. Existing results were preserved.',file=sys.stderr)
        sys.exit(1)
