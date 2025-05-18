#!/usr/bin/env python3
"""
Lightweight CLI for the PDF-Q&A API.

Usage:
    $ python chatbot.py /path/to/document.pdf
"""

import argparse
import datetime
import json
import queue
import sys
import threading
import uuid
import requests
from pathlib import Path

# --- config ------------------------------------------------------------
BASE_URL   = "http://localhost:8000"
EVENT_PATH = "/events/stream"
UPLOAD_URL = "/documents/upload"
CHAT_URL   = "/chat"
TIMEOUT    = 15          # seconds to wait for ready / indexed
# -----------------------------------------------------------------------


def sse_reader(url: str, out_q: queue.Queue):
    """
    Simple blocking SSE client.  Writes dicts {"event": str, "data": str}
    into out_q until the HTTP connection closes.
    """
    with requests.get(url, stream=True) as resp:
        event, data = None, []
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            if raw.startswith("event:"):
                event = raw[6:].strip()
            elif raw.startswith("data:"):
                data.append(raw[5:].strip())
            elif raw == "":            # empty line → dispatch
                if event:
                    out_q.put({"event": event, "data": "\n".join(data)})
                event, data = None, []

def _render(chunk: str) -> str:
    """Turn literal \\n into real newlines."""
    return chunk.replace("\\n", "\n")

def main():
    parser = argparse.ArgumentParser(description="CLI for PDF Q&A backend")
    parser.add_argument("pdf", type=Path, help="PDF to ingest")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"file not found: {args.pdf}")

    client_id = str(uuid.uuid4())
    headers   = {"X-Client-Id": client_id}

    # 1) start SSE listener thread --------------------------------------
    q = queue.Queue()
    stream_url = f"{BASE_URL}{EVENT_PATH}?client_id={client_id}"
    t = threading.Thread(target=sse_reader, args=(stream_url, q), daemon=True)
    t.start()

    # 2) upload PDF ------------------------------------------------------
    with args.pdf.open("rb") as f:
        files = {"file": (args.pdf.name, f, "application/pdf")}
        r = requests.post(f"{BASE_URL}{UPLOAD_URL}", headers=headers, files=files)
        r.raise_for_status()
    print(f"Ingested {args.pdf.name}. Waiting for indexing…")

    # 3) wait for ready & indexed events --------------------------------
    doc_id = None
    indexed = False
    while not indexed:
        try:
            evt = q.get(timeout=TIMEOUT)
        except queue.Empty:
            sys.exit("backend timed-out while processing the document")

        if evt["event"] == "ready":
            doc_id = eval(evt["data"])["document_id"]   # small & safe
        elif evt["event"] == "indexed":
            indexed = True

    print("Indexed. Please ask your questions.")

    # 4) interactive chat loop ------------------------------------------
    try:
        while True:
            user_q = input(">>> ").strip()
            if user_q.lower() == "exit":
                break

            payload = {"query": user_q, "document_id": doc_id}
            r = requests.post(f"{BASE_URL}{CHAT_URL}", headers=headers, json=payload)
            r.raise_for_status()

            # collect answer tokens
            while True:
                evt = q.get(timeout=TIMEOUT)
                if evt["event"] == "chat":
                    resp = json.loads(evt["data"])
                    if (resp["type"] == 'chunk'):
                        sys.stdout.write(_render(resp["content"]))
                        sys.stdout.flush()
                    elif (resp["type"] == 'final'):
                        sys.stdout.write("\n\nRelevant pages from source: " + str(resp["content"]["source_pages"]))
                        sys.stdout.flush()
                        break
                elif evt["event"] == "error":
                    print(f"\n[server error] {evt['data']}")
                    break
                elif evt["event"] == "ping":
                    continue
                else:
                    # "indexed" or trailing meta means answer finished
                    print()
                    break

            print("\n")

    except KeyboardInterrupt:
        pass

    print("\nbye!")


if __name__ == "__main__":
    main()