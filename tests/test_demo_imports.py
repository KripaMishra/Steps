import os
import subprocess
import sys


def test_demo_startup_does_not_import_or_initialize_full_mode_dependencies():
    code = """
import json
import sys
from components.s5_rag_llm import RAGModel
blocked = {'torch', 'transformers', 'pymilvus', 'elasticsearch'}
assert not (blocked & set(sys.modules)), blocked & set(sys.modules)
model = RAGModel()
result = model.process_query('What is CUDA occupancy?')
assert result['mode'] == 'demo'
assert result['sources']
assert result['model'] == 'extractive-fallback'
assert not (blocked & set(sys.modules)), blocked & set(sys.modules)
print(json.dumps({'mode': result['mode'], 'sources': len(result['sources'])}))
"""
    env = {
        **os.environ,
        "RAG_MODE": "demo",
        "GEMINI_API_KEY": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert '"mode": "demo"' in completed.stdout
