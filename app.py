import time
import threading
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse

from ingest_pipeline import main as run_ingest


app = FastAPI(title="Wind Bilance Ingest Worker")

_ingest_lock = threading.Lock()
_is_running: bool = False
_last_run_ts: Optional[float] = None
_last_error: Optional[str] = None


def _run_ingest_safe():
    global _is_running, _last_run_ts, _last_error

    with _ingest_lock:
        if _is_running:
            print("[worker] Ingest già in esecuzione, non ne avvio un altro.")
            return
        _is_running = True

    try:
        print("[worker] >>> Avvio ingest")
        start = time.time()
        run_ingest()
        elapsed = time.time() - start
        print(f"[worker] <<< Ingest completato in {elapsed:.1f}s")
        _last_run_ts = time.time()
        _last_error = None
    except Exception as e:
        print("[worker] ERRORE in ingest:", repr(e))
        _last_error = repr(e)
    finally:
        with _ingest_lock:
            _is_running = False


@app.get("/")
def root():
    return {"status": "ok", "message": "Wind Bilance ingest worker"}


@app.get("/status")
def status():
    return {
        "running": _is_running,
        "last_run_ts": _last_run_ts,
        "last_error": _last_error,
    }


@app.post("/ingest")
def ingest(background_tasks: BackgroundTasks):
    """
    Avvia un ingest in background, se non ce n'è già uno in corso.
    """
    with _ingest_lock:
        if _is_running:
            # Non è un errore: semplicemente c'è già un ingest in corso
            return JSONResponse(
                status_code=200,
                content={"status": "already_running", "detail": "Ingest già in esecuzione"},
            )

        # Avvio ingest in background
        background_tasks.add_task(_run_ingest_safe)
        return {"status": "started", "detail": "Ingest avviato"}
