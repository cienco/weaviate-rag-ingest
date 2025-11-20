# Weaviate Ingest Worker

Worker per ingest multimodale (Vertex + Document AI) su Weaviate Cloud.

## Variabili d'ambiente richieste

- `WEAVIATE_URL` – URL del cluster Weaviate Cloud
- `WEAVIATE_API_KEY` – API key del cluster Weaviate
- `GCP_PROJECT_ID` – ID del progetto GCP
- `VERTEX_LOCATION` – (opzionale) location Vertex, default `us-central1`
- `DOCAI_PROJECT_ID` – (opzionale) progetto GCP per Document AI, default `GCP_PROJECT_ID`
- `DOCAI_LOCATION` – location Document AI (es. `eu`), default `eu`
- `DOCAI_PROCESSOR_ID` – ID del processor Document AI (ultima parte del nome)
- `GCP_SERVICE_ACCOUNT_JSON` – contenuto JSON della service account con permessi Vertex + Document AI
- `SOURCE_BASE_DIR` – percorso della cartella locale con i file da indicizzare (default `/data/wind_bilance_files`)

## Comandi

Installazione dipendenze:

```bash
pip install -r requirements.txt
