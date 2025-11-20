import os
import time
import threading
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ingest_pipeline import main as run_ingest


app = FastAPI(title="Wind Bilance Ingest Worker")

# Flag + lock per evitare ingest sovrapposti
_ingest_lock = threading.Lock()
_is_running = False


def _run_ingest_safe():
    """
    Esegue run_ingest() evitando che ci siano più ingest in parallelo.
    """
    global _is_running

    with _ingest_lock:
        if _is_running:
            print("[worker] Ingest già in esecuzione, salto.")
            return
        _is_running = True

    try:
        print("[worker] >>> Avvio ingest manuale/background")
        start = time.time()
        run_ingest()
        elapsed = time.time() - start
        print(f"[worker] <<< Ingest completato in {elapsed:.1f}s")
    except Exception as e:
        print("[worker] ERRORE in ingest:", repr(e))
    finally:
        with _ingest_lock:
            _is_running = False


def _background_loop():
    """
    Loop che esegue ingest periodico ogni N secondi.
    """
    interval_str = os.getenv("INGEST_INTERVAL_SECONDS", "600")  # default 10 minuti
    try:
        interval = int(interval_str)
    except ValueError:
        interval = 600

    print(f"[worker] Loop ingest attivo – intervallo = {interval} secondi")

    while True:
        _run_ingest_safe()
        print(f"[worker] Sleep per {interval} secondi")
        time.sleep(interval)


@app.get("/")
def root():
    return {"status": "ok", "message": "Wind Bilance ingest worker"}


@app.on_event("startup")
def on_startup():
    """
    Avviato all'avvio del processo: parte il thread di background.
    """
    t = threading.Thread(target=_background_loop, daemon=True)
    t.start()
    print("[worker] Thread di background avviato.")


@app.get("/healthz")
def health():
    """
    Endpoint di health check (utile per Render).
    """
    return {"status": "ok"}


@app.post("/ingest")
def ingest_manual():
    """
    Forza un ingest manuale:
    - se non c'è ingest in corso → parte subito in un thread separato
    - se c'è ingest in corso → ritorna 409 (conflitto)
    """
    global _is_running

    with _ingest_lock:
        if _is_running:
            return JSONResponse(
                status_code=409,
                content={"status": "busy", "detail": "Ingest già in esecuzione"},
            )

        # Lancia ingest in un thread separato
        t = threading.Thread(target=_run_ingest_safe, daemon=True)
        t.start()

    return {"status": "started", "detail": "Ingest avviato"}


# opzionale: endpoint GET /ingest per comodità da browser
@app.get("/ingest")
def ingest_manual_get():
    return ingest_manual()
