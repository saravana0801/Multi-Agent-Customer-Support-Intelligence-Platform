from setuptools import setup, find_packages

setup(
    name="customer-support-intelligence-platform",
    version="1.0.0",
    description="Multi-Agent Customer Support Intelligence Platform with CrewAI, Sub-3ms ML Triage, ChromaDB RAG, Guardrails AI, LiteLLM, and MCP",
    author="AgenticAI Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "chromadb>=0.4.0",
        "onnxruntime>=1.18.0",
        "crewai>=1.15.0",
        "mcp>=1.28.0",
        "litellm>=1.100.0",
        "guardrails-ai>=0.11.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.30.0",
        "python-multipart>=0.0.9",
        "streamlit>=1.35.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "postgres": ["psycopg2-binary>=2.9.0"],
    },
    entry_points={
        "console_scripts": [
            "support-api=src.api.server:app",
        ],
    },
)
