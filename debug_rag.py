"""Quick diagnostic: check what's in the vectorstore and test retrieval."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("RAG DIAGNOSTIC TOOL")
print("=" * 60)

# 1. Check what files are in data/uploads
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
manuals_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "manuals")

print("\n[1] FILES IN data/uploads:")
if os.path.exists(uploads_dir):
    for f in os.listdir(uploads_dir):
        fpath = os.path.join(uploads_dir, f)
        print(f"  - {f} ({os.path.getsize(fpath)} bytes)")
else:
    print("  (directory does not exist)")

print("\n[2] FILES IN data/manuals:")
if os.path.exists(manuals_dir):
    for f in os.listdir(manuals_dir):
        fpath = os.path.join(manuals_dir, f)
        print(f"  - {f} ({os.path.getsize(fpath)} bytes)")
else:
    print("  (directory does not exist)")

# 2. Try loading manuals
print("\n[3] TESTING load_manuals():")
from ingestion.load_manuals import load_manuals
docs = load_manuals()
print(f"  Total chunks loaded: {len(docs)}")

# Show sources
sources = {}
for d in docs:
    src = d.metadata.get("source", "unknown")
    sources[src] = sources.get(src, 0) + 1
print("  Chunks per source:")
for src, count in sorted(sources.items()):
    print(f"    {src}: {count} chunks")

# Show a sample from each source
print("\n  Sample content from each source:")
seen_sources = set()
for d in docs:
    src = d.metadata.get("source", "unknown")
    if src not in seen_sources:
        seen_sources.add(src)
        preview = d.page_content[:150].replace("\n", " ")
        print(f"    [{src}] {preview}...")

# 3. Test vectorstore search
print("\n[4] TESTING VECTORSTORE SEARCH:")
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(persist_dir):
    vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    collection = vs._collection
    total = collection.count()
    print(f"  Total docs in vectorstore: {total}")

    # Check what types are in the store
    all_meta = collection.get(include=["metadatas"])
    type_counts = {}
    source_counts = {}
    for m in all_meta["metadatas"]:
        t = m.get("type", "unknown")
        s = m.get("source", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        source_counts[s] = source_counts.get(s, 0) + 1

    print("  By type:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")
    print("  By source:")
    for s, c in sorted(source_counts.items()):
        print(f"    {s}: {c}")

    # Test a search
    test_query = "milling machine specifications"
    print(f"\n[5] TEST SEARCH: '{test_query}'")
    results = vs.similarity_search(test_query, k=4, filter={"type": {"$eq": "manual"}})
    print(f"  Results: {len(results)}")
    for i, r in enumerate(results):
        src = r.metadata.get("source", "?")
        preview = r.page_content[:120].replace("\n", " ")
        print(f"  [{i+1}] source={src} | {preview}...")

    # Test with ATLAS filter
    test_query2 = "ATLAS milling machine"
    print(f"\n[6] TEST SEARCH with source filter: '{test_query2}' source=ATLAS_Milling_Machine.pdf")
    results2 = vs.similarity_search(
        test_query2, k=4,
        filter={"$and": [{"type": {"$eq": "manual"}}, {"source": {"$eq": "ATLAS_Milling_Machine.pdf"}}]}
    )
    print(f"  Results: {len(results2)}")
    for i, r in enumerate(results2):
        src = r.metadata.get("source", "?")
        preview = r.page_content[:120].replace("\n", " ")
        print(f"  [{i+1}] source={src} | {preview}...")
else:
    print("  Vectorstore directory does not exist!")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
