"""
Import synthetic data into Hybrid RAG System
"""

import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env in the project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import HybridRAGSystem

def import_data(neo4j_uri, neo4j_user, neo4j_password, openai_key):
    """Import products and receipts into RAG system"""

    print("Initializing RAG system...")
    rag_system = HybridRAGSystem(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openai_key=openai_key
    )

    rag_system.setup_graph_schema()

    with open('products.json', 'r') as f:
        products = json.load(f)

    print(f"\nImporting {len(products)} products...")

    for product in products:
        success, msg = rag_system.add_product_to_graph(product)
        if success:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}")

        if os.path.exists(product['receipt_path']):
            success, msg = rag_system.load_pdf_to_vector_store(
                product['receipt_path'],
                product['id']
            )
            if success:
                print(f"    ✓ Loaded receipt PDF")
            else:
                print(f"    ✗ {msg}")

    print("\n✓ Import complete!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Import synthetic data')
    parser.add_argument('--neo4j-uri',      default=os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687"))
    parser.add_argument('--neo4j-user',     default=os.getenv("NEO4J_USER",     "neo4j"))
    parser.add_argument('--neo4j-password', default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument('--openai-key',     default=os.getenv("OPENAI_API_KEY", ""))

    args = parser.parse_args()

    if not args.openai_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or pass --openai-key.")
        sys.exit(1)

    import_data(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.openai_key
    )
