"""
Multi-Agent Customer Support Intelligence Platform.
Global package initialization with telemetry and environment sanitization.
"""

import os

# Suppress external background telemetry calls to avoid network retry delays
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("GUARDRAILS_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import warnings

# Suppress harmless upstream pydantic forward-ref warnings
warnings.filterwarnings("ignore", message=".*lifespan.*")
warnings.filterwarnings("ignore", message=".*IncompleteFieldDefinitionWarning.*")


try:
    import posthog
    posthog.disabled = True
except Exception:
    pass


