import os
import sys
import json
import time

# Resolve workspace root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.api.main import create_app

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def main():
    print("Initializing CrackLaw FastAPI Application Factory...")
    app = create_app()
    client = TestClient(app)
    
    # 1. Diagnostic Checks
    print_header("1. DIAGNOSTICS & SYSTEM METRICS")
    
    health_resp = client.get("/health")
    print(f"[GET /health] Status: {health_resp.status_code}")
    print(f"Response: {json.dumps(health_resp.json(), indent=2)}")

    status_resp = client.get("/status")
    print(f"\n[GET /status] Status: {status_resp.status_code}")
    print(f"Response (Partial): {list(status_resp.json().keys())}")
    print(f"Database registered documents count: {status_resp.json()['database']['registered_documents']}")

    # 2. Chatting synchronously
    print_header("2. LEGAL CHAT ENDPOINT")
    chat_payload = {
        "session_id": "demo_session_789",
        "query": "Is there a grace period for contract performance?"
    }
    print(f"Submitting chat message: '{chat_payload['query']}'")
    t0 = time.time()
    chat_resp = client.post("/api/v1/chat", json=chat_payload)
    latency = (time.time() - t0) * 1000
    print(f"[POST /api/v1/chat] Status: {chat_resp.status_code} | Latency: {latency:.2f} ms")
    if chat_resp.status_code == 200:
        data = chat_resp.json()
        print(f"Assistant response: {data['response_text']}")
        print(f"Intent detected: {data['intent']}")
        print(f"Confidence score: {data['confidence_score']:.2f}")
        print(f"Citations count: {len(data['citations'])}")

    # 3. Chat Streaming
    print_header("3. CHAT STREAMING (SSE)")
    print("Initiating streaming connection...")
    t0 = time.time()
    with client.stream("POST", "/api/v1/chat/stream", json=chat_payload) as stream_resp:
        print(f"[POST /api/v1/chat/stream] Status: {stream_resp.status_code}")
        print("Receiving tokens: ", end="", flush=True)
        for line in stream_resp.iter_lines():
            if line:
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if line_str.startswith("event: token"):
                    # Extract data payload
                    # Next line is data: ...
                    pass
                elif line_str.startswith("data: "):
                    try:
                        data_payload = json.loads(line_str[6:])
                        if "token" in data_payload:
                            print(data_payload["token"], end="", flush=True)
                    except Exception:
                        pass
        print(f"\nStream completed in {(time.time() - t0) * 1000:.2f} ms")

    # 4. Search API
    print_header("4. HYBRID & SEMANTIC SEARCH")
    search_payload = {
        "query": "Limitation of liability",
        "top_k": 2,
        "min_similarity": 0.2
    }
    
    hybrid_resp = client.post("/api/v1/search/hybrid", json=search_payload)
    print(f"[POST /api/v1/search/hybrid] Status: {hybrid_resp.status_code}")
    if hybrid_resp.status_code == 200:
        data = hybrid_resp.json()
        print(f"Found {len(data['results'])} matches.")
        for idx, result in enumerate(data["results"]):
            print(f"  [{idx + 1}] Source: {result['source']} | Section: {result['section']} | Score: {result['score']:.4f}")

    # 5. Contract Risk Scoring
    print_header("5. CONTRACT RISK ANALYTICS")
    contract_text = "The vendor shall indemnify the client from all liability up to a maximum of $1000."
    print("Analyzing clause text...")
    contract_resp = client.post("/api/v1/contracts/analyze", json={"text": contract_text})
    print(f"[POST /api/v1/contracts/analyze] Status: {contract_resp.status_code}")
    if contract_resp.status_code == 200:
        data = contract_resp.json()
        print(f"Calculated Risk Score: {data['risk_score']:.4f}")
        print(f"Contract Risk Level: {data['risk_level']}")
        print(f"Feature vectors: {data['engineered_features']}")

    # 6. Document Upload Simulation
    print_header("6. DOCUMENT UPLOAD & PARSING")
    dummy_doc_content = b"This is a manual uploaded document contract. Section 45. Governing Law."
    files = {"file": ("manual_contract.txt", dummy_doc_content, "text/plain")}
    print("Uploading file manual_contract.txt...")
    upload_resp = client.post("/api/v1/documents/upload", files=files)
    print(f"[POST /api/v1/documents/upload] Status: {upload_resp.status_code}")
    if upload_resp.status_code == 200:
        data = upload_resp.json()
        print(f"Indexed document ID: {data['document_id']}")
        print(f"Status: {data['status_detail']}")
        print(f"Chunks generated: {data['chunks_count']}")
    else:
        print(f"Upload error: {upload_resp.text}")

    # 7. Global Metrics Summary
    print_header("7. TELEMETRY REQUEST COUNTS")
    metrics_resp = client.get("/metrics")
    print(f"Response: {json.dumps(metrics_resp.json(), indent=2)}")

if __name__ == "__main__":
    main()
