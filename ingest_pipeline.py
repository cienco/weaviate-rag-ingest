import os
import io
import base64
from datetime import datetime, timezone
from typing import List, Dict, Any

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import (
    Property,
    DataType,
    Configure,
    Multi2VecField,
)
from weaviate.classes.query import Filter

from pdf2image import convert_from_bytes
import fitz  # pymupdf
import docx  # python-docx
import pandas as pd

from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.cloud import documentai_v1 as documentai

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# =============================================================================
# CONFIGURAZIONE DA ENV VAR
# =============================================================================

# Weaviate
WCS_URL = os.getenv("WEAVIATE_URL")
WCS_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not WCS_URL or not WCS_API_KEY:
    raise RuntimeError("WEAVIATE_URL e WEAVIATE_API_KEY devono essere settate nelle env vars.")

# GCP / Vertex / Document AI
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
if not GCP_PROJECT_ID:
    raise RuntimeError("GCP_PROJECT_ID deve essere settata nelle env vars.")

VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
DOCAI_PROJECT_ID = os.getenv("DOCAI_PROJECT_ID", GCP_PROJECT_ID)
DOCAI_LOCATION = os.getenv("DOCAI_LOCATION", "eu")
DOCAI_PROCESSOR_ID = os.getenv("DOCAI_PROCESSOR_ID")

if not DOCAI_PROCESSOR_ID:
    raise RuntimeError("DOCAI_PROCESSOR_ID deve essere settata nelle env vars.")

# Service Account JSON (contenuto completo della chiave) passato via env
GCP_SA_JSON = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
if not GCP_SA_JSON:
    raise RuntimeError("GCP_SERVICE_ACCOUNT_JSON deve contenere il JSON della service account.")

SA_PATH = "/tmp/gcp-sa.json"
with open(SA_PATH, "w") as f:
    f.write(GCP_SA_JSON)

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

base_creds = service_account.Credentials.from_service_account_file(
    SA_PATH,
    scopes=SCOPES,
)
base_creds = base_creds.with_scopes(SCOPES)
base_creds.refresh(Request())
vertex_token = base_creds.token

# Sorgente file: cartella locale (potrà essere montata o popolata da un altro sistema)
SOURCE_BASE_DIR = os.getenv("SOURCE_BASE_DIR", "/data/wind_bilance_files")

# Tipi di file
INDEXABLE_TYPES = {"pdf", "docx", "txt", "png", "tif", "xls"}
IGNORED_TYPES = {"zip", "sql", "doc", "msg"}

# Limite di caratteri per il campo text del multimodal embedding
MAX_TEXT_CHARS = 900


# =============================================================================
# GOOGLE DRIVE CONFIG
# =============================================================================

# ID della cartella root su Google Drive da cui leggere TUTTI i file Wind Bilance
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")
if not GDRIVE_FOLDER_ID:
    raise RuntimeError("GDRIVE_FOLDER_ID non è settata: serve l'ID della cartella su Google Drive.")

_drive_service = None


def get_drive_service():
    """
    Client Google Drive v3 riusando le stesse credenziali (base_creds).
    """
    global _drive_service
    if _drive_service is None:
        _drive_service = build("drive", "v3", credentials=base_creds)
    return _drive_service


# =============================================================================
# HELPER GENERALI
# =============================================================================

def normalize_ext(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext.startswith("."):
        ext = ext[1:]
    return ext


def parse_iso(dt_str: str | None):
    if not dt_str:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def page_img_to_bytes(page_img) -> bytes:
    buf = io.BytesIO()
    page_img.save(buf, format="PNG")
    return buf.getvalue()


def chunk_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> List[str]:
    text = text or ""
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


# =============================================================================
# CONNESSIONE WEAVIATE + SCHEMA
# =============================================================================

def get_weaviate_client() -> weaviate.WeaviateClient:
    """
    Connessione a Weaviate Cloud usando SOLO l'API key del cluster.
    Niente header Authorization custom (quello lo gestisce il client per WCS).
    """
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WCS_URL,
        auth_credentials=Auth.api_key(WCS_API_KEY),
    )
    return client


def create_schema_if_needed(client: weaviate.WeaviateClient):
    """
    Crea le collection FileIndexStatus e WindChunk se non esistono.
    """
    existing = set(client.collections.list_all())

    # FileIndexStatus (non vettoriale)
    if "FileIndexStatus" not in existing:
        client.collections.create(
            name="FileIndexStatus",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="sourceId",     data_type=DataType.TEXT),
                Property(name="name",         data_type=DataType.TEXT),
                Property(name="path",         data_type=DataType.TEXT),
                Property(name="url",          data_type=DataType.TEXT),
                Property(name="fileType",     data_type=DataType.TEXT),
                Property(name="lastModified", data_type=DataType.DATE),
                Property(name="indexedAt",    data_type=DataType.DATE),
                Property(name="isDeleted",    data_type=DataType.BOOL),
                Property(name="note",         data_type=DataType.TEXT),
            ],
        )

    # WindChunk (multimodale)
    if "WindChunk" not in existing:
        client.collections.create(
            name="WindChunk",
            vectorizer_config=Configure.Vectorizer.multi2vec_google(
                project_id=GCP_PROJECT_ID,
                location=VERTEX_LOCATION,
                model_id="multimodalembedding@001",
                image_fields=[Multi2VecField(name="image_b64", weight=0.4)],
                text_fields=[Multi2VecField(name="text",      weight=0.6)],
            ),
            properties=[
                Property(name="sourceId",  data_type=DataType.TEXT),
                Property(name="fileName",  data_type=DataType.TEXT),
                Property(name="fileType",  data_type=DataType.TEXT),
                Property(name="pageIndex", data_type=DataType.INT),
                Property(name="chunkIndex", data_type=DataType.INT),
                Property(name="sheetName", data_type=DataType.TEXT),
                Property(name="text",      data_type=DataType.TEXT),
                Property(name="image_b64", data_type=DataType.BLOB),
                Property(name="url",       data_type=DataType.TEXT),
            ],
        )


# =============================================================================
# SORGENTE FILE (filesystem locale)
# =============================================================================

class SourceFile:
    def __init__(self, id: str, name: str, path: str, url: str, last_modified: str):
        self.id = id
        self.name = name
        self.path = path
        self.url = url
        self.last_modified = last_modified


def list_source_files() -> List[SourceFile]:
    """
    Legge tutti i file da una cartella di Google Drive (e sotto-cartelle),
    usando GDRIVE_FOLDER_ID come root.
    """
    service = get_drive_service()
    root_id = GDRIVE_FOLDER_ID
    files: List[SourceFile] = []

    # BFS sulle cartelle di Drive: (prefix_path, folder_id)
    queue: List[tuple[str, str]] = [("", root_id)]

    while queue:
        path_prefix, folder_id = queue.pop(0)

        page_token = None
        while True:
            resp = service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)",
                pageToken=page_token,
            ).execute()

            for item in resp.get("files", []):
                mime = item.get("mimeType")
                fid = item["id"]
                name = item["name"]

                # Sottocartella → metti in coda
                if mime == "application/vnd.google-apps.folder":
                    new_prefix = f"{path_prefix}{name}/"
                    queue.append((new_prefix, fid))
                    continue

                # File "normale"
                rel_path = f"{path_prefix}{name}"
                last_modified = item.get("modifiedTime")  # ISO 8601 già ok per Weaviate
                url = item.get("webViewLink", "")

                files.append(
                    SourceFile(
                        id=fid,                # sourceId = fileId di Drive
                        name=name,
                        path=rel_path,         # path logico es: "subdir/file.pdf"
                        url=url,
                        last_modified=last_modified,
                    )
                )

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    print(f"[source] Trovati {len(files)} file in Google Drive (root={root_id})")
    return files


def download_source_file(file_meta: Dict[str, Any]) -> bytes:
    """
    Scarica il file da Google Drive usando sourceId come fileId.
    """
    service = get_drive_service()
    file_id = file_meta["sourceId"]

    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        # Se vuoi puoi loggare: print("Download %d%%" % int(status.progress() * 100))

    return buf.getvalue()


# =============================================================================
# DOCUMENT AI (OCR)
# =============================================================================

_docai_client = None


def get_docai_client():
    global _docai_client
    if _docai_client is not None:
        return _docai_client
    _docai_client = documentai.DocumentProcessorServiceClient(credentials=base_creds)
    return _docai_client


def run_ocr_on_image_bytes(image_bytes: bytes) -> str:
    client = get_docai_client()

    processor_name = client.processor_path(
        DOCAI_PROJECT_ID,
        DOCAI_LOCATION,
        DOCAI_PROCESSOR_ID,
    )

    raw_document = documentai.RawDocument(
        content=image_bytes,
        mime_type="image/png",
    )

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document,
    )

    result = client.process_document(request=request)
    doc = result.document

    text = doc.text or ""
    text = text.replace("\r\n", "\n").strip()
    return text


# =============================================================================
# FILEINDEXSTATUS: SYNC, QUERY, UPDATE
# =============================================================================

def sync_source_to_fileindex(client: weaviate.WeaviateClient):
    """
    Sincronizza SourceFile -> FileIndexStatus.
    """
    coll = client.collections.get("FileIndexStatus")

    existing: Dict[str, Any] = {}
    res = coll.query.fetch_objects(limit=1000)
    for obj in res.objects:
        sid = obj.properties.get("sourceId")
        if sid:
            existing[sid] = obj

    src_files = list_source_files()
    seen_ids = set()

    for sf in src_files:
        source_id = sf.id
        seen_ids.add(source_id)

        file_type = normalize_ext(sf.name)

        props = {
            "sourceId":     source_id,
            "name":         sf.name,
            "path":         sf.path,
            "url":          sf.url,
            "fileType":     file_type,
            "lastModified": sf.last_modified,
        }

        if file_type in IGNORED_TYPES:
            props["note"] = f"ignored: {file_type}"
        else:
            props["note"] = ""

        if source_id in existing:
            coll.data.update(id=existing[source_id].uuid, properties=props)
        else:
            coll.data.insert(properties=props)

    for sid, obj in existing.items():
        if sid not in seen_ids:
            coll.data.update(id=obj.uuid, properties={"isDeleted": True})


def list_files_to_ingest(client: weaviate.WeaviateClient) -> List[Dict[str, Any]]:
    coll = client.collections.get("FileIndexStatus")

    where_filter = Filter.all_of([
        Filter.by_property("isDeleted").equal(False),
    ])

    res = coll.query.fetch_objects(limit=1000, filters=where_filter)
    files: List[Dict[str, Any]] = []

    for obj in res.objects:
        props = obj.properties
        file_type = (props.get("fileType") or "").lower()

        if file_type not in INDEXABLE_TYPES:
            continue

        last_mod = parse_iso(props.get("lastModified"))
        indexed_at = parse_iso(props.get("indexedAt"))

        if indexed_at is None:
            files.append(props)
            continue

        if last_mod and last_mod > indexed_at:
            files.append(props)

    return files


def mark_file_indexed(client: weaviate.WeaviateClient, source_id: str):
    coll = client.collections.get("FileIndexStatus")
    res = coll.query.fetch_objects(
        filters=Filter.by_property("sourceId").equal(source_id),
        limit=1,
    )
    if not res.objects:
        return
    obj = res.objects[0]
    coll.data.update(
        id=obj.uuid,
        properties={"indexedAt": now_iso_utc()}
    )


# =============================================================================
# GESTIONE WINDCHUNK
# =============================================================================

def delete_windchunks_for_file(client: weaviate.WeaviateClient, source_id: str):
    coll = client.collections.get("WindChunk")
    where = Filter.by_property("sourceId").equal(source_id)
    coll.data.delete_many(where=where)


def extract_native_text_by_page(pdf_bytes: bytes) -> List[str]:
    texts = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        texts.append(page.get_text("text"))
    return texts


# =============================================================================
# INGEST PER TIPO DI FILE
# =============================================================================

def ingest_pdf(client: weaviate.WeaviateClient, file_meta: Dict[str, Any]):
    source_id = file_meta["sourceId"]
    file_name = file_meta["name"]
    file_url = file_meta["url"]

    print(f"[ingest_pdf] Inizio ingest {file_name} ({source_id})")

    pdf_bytes = download_source_file(file_meta)

    pages = convert_from_bytes(pdf_bytes, dpi=200)
    native_text_pages = extract_native_text_by_page(pdf_bytes)

    coll = client.collections.get("WindChunk")

    for page_index, page_img in enumerate(pages, start=1):
        img_bytes = page_img_to_bytes(page_img)
        image_b64 = base64.b64encode(img_bytes).decode("utf-8")

        ocr_text = run_ocr_on_image_bytes(img_bytes)

        native_text = ""
        if page_index - 1 < len(native_text_pages):
            native_text = native_text_pages[page_index - 1] or ""

        combined_text = (native_text + "\n" + ocr_text).strip()
        chunks = chunk_text(combined_text) if combined_text else [""]

        for idx, chunk in enumerate(chunks):
            props = {
                "sourceId":   source_id,
                "fileName":   file_name,
                "fileType":   "pdf",
                "pageIndex":  page_index,
                "chunkIndex": idx,
                "sheetName":  "",
                "text":       chunk,
                "image_b64":  image_b64 if idx == 0 else None,
                "url":        file_url,
            }
            coll.data.insert(properties=props)

    print(f"[ingest_pdf] Fine ingest {file_name}: {len(pages)} pagine indicizzate")


def ingest_docx(client: weaviate.WeaviateClient, file_meta: Dict[str, Any]):
    source_id = file_meta["sourceId"]
    file_name = file_meta["name"]
    file_url = file_meta["url"]

    print(f"[ingest_docx] Inizio ingest {file_name} ({source_id})")

    file_bytes = download_source_file(file_meta)
    doc_stream = io.BytesIO(file_bytes)
    document = docx.Document(doc_stream)

    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)

    chunks = chunk_text(full_text)
    coll = client.collections.get("WindChunk")

    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        props = {
            "sourceId":   source_id,
            "fileName":   file_name,
            "fileType":   "docx",
            "pageIndex":  0,
            "chunkIndex": idx,
            "sheetName":  "",
            "text":       chunk,
            "image_b64":  None,
            "url":        file_url,
        }
        coll.data.insert(properties=props)

    print(f"[ingest_docx] Fine ingest {file_name}: {len(chunks)} chunk")


def ingest_txt(client: weaviate.WeaviateClient, file_meta: Dict[str, Any]):
    source_id = file_meta["sourceId"]
    file_name = file_meta["name"]
    file_url = file_meta["url"]

    print(f"[ingest_txt] Inizio ingest {file_name} ({source_id})")

    file_bytes = download_source_file(file_meta)

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="ignore")

    chunks = chunk_text(text)
    coll = client.collections.get("WindChunk")

    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        props = {
            "sourceId":   source_id,
            "fileName":   file_name,
            "fileType":   "txt",
            "pageIndex":  0,
            "chunkIndex": idx,
            "sheetName":  "",
            "text":       chunk,
            "image_b64":  None,
            "url":        file_url,
        }
        coll.data.insert(properties=props)

    print(f"[ingest_txt] Fine ingest {file_name}: {len(chunks)} chunk")


def ingest_image(client: weaviate.WeaviateClient, file_meta: Dict[str, Any]):
    source_id = file_meta["sourceId"]
    file_name = file_meta["name"]
    file_url = file_meta["url"]
    file_type = file_meta.get("fileType", "").lower()

    print(f"[ingest_image] Inizio ingest {file_name} ({source_id})")

    img_bytes = download_source_file(file_meta)
    image_b64 = base64.b64encode(img_bytes).decode("utf-8")

    ocr_text = run_ocr_on_image_bytes(img_bytes)

    coll = client.collections.get("WindChunk")

    props = {
        "sourceId":   source_id,
        "fileName":   file_name,
        "fileType":   file_type,
        "pageIndex":  1,
        "chunkIndex": 0,
        "sheetName":  "",
        "text":       ocr_text,
        "image_b64":  image_b64,
        "url":        file_url,
    }
    coll.data.insert(properties=props)

    print(f"[ingest_image] Fine ingest {file_name}")


def ingest_xls(client: weaviate.WeaviateClient, file_meta: Dict[str, Any]):
    source_id = file_meta["sourceId"]
    file_name = file_meta["name"]
    file_url = file_meta["url"]

    print(f"[ingest_xls] Inizio ingest {file_name} ({source_id})")

    file_bytes = download_source_file(file_meta)
    xls_stream = io.BytesIO(file_bytes)

    sheets = pd.read_excel(xls_stream, sheet_name=None)

    coll = client.collections.get("WindChunk")

    chunk_counter = 0

    for sheet_name, df in sheets.items():
        text_repr = df.to_csv(index=False, sep=";", line_terminator="\n")
        text = f"Sheet: {sheet_name}\n{text_repr}"

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            props = {
                "sourceId":   source_id,
                "fileName":   file_name,
                "fileType":   "xls",
                "pageIndex":  0,
                "chunkIndex": chunk_counter,
                "sheetName":  sheet_name,
                "text":       chunk,
                "image_b64":  None,
                "url":        file_url,
            }
            coll.data.insert(properties=props)
            chunk_counter += 1

    print(f"[ingest_xls] Fine ingest {file_name}: {chunk_counter} chunk da {len(sheets)} sheet")


# =============================================================================
# DISPATCH
# =============================================================================

def ingest_single_file(client: weaviate.WeaviateClient, file_meta: Dict[str, Any]):
    source_id = file_meta["sourceId"]
    file_type = (file_meta.get("fileType") or "").lower()

    print(f"[ingest] File: {file_meta['name']} ({source_id}), type={file_type}")

    delete_windchunks_for_file(client, source_id)

    if file_type == "pdf":
        ingest_pdf(client, file_meta)
    elif file_type == "docx":
        ingest_docx(client, file_meta)
    elif file_type == "txt":
        ingest_txt(client, file_meta)
    elif file_type in {"png", "tif"}:
        ingest_image(client, file_meta)
    elif file_type == "xls":
        ingest_xls(client, file_meta)
    else:
        print(f"[ingest] Tipo non gestito (ignorato): {file_type}")
        return

    mark_file_indexed(client, source_id)


# =============================================================================
# MAIN
# =============================================================================

def main():
    client = get_weaviate_client()
    create_schema_if_needed(client)

    sync_source_to_fileindex(client)
    files = list_files_to_ingest(client)
    print(f"[main] File da ingest: {len(files)}")

    for fm in files:
        try:
            ingest_single_file(client, fm)
        except Exception as e:
            print(f"[ERROR] Ingest fallito per {fm.get('name')} ({fm.get('sourceId')}): {e}")

    client.close()
    print("[main] Ingest completato.")
# placeholder ingest_pipeline.py - use content from chat
