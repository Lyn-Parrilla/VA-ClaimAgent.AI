"""Provide dummy Google Cloud config before main.py runs its startup validation.

main.py fails fast at import time when GOOGLE_APPLICATION_CREDENTIALS or
GOOGLE_CLOUD_PROJECT are missing, so these must be set before the test module
imports it. The credentials file is a throwaway stub: no test performs real
Cloud DLP or Vertex AI calls, they monkeypatch those boundaries instead.
"""

import json
import os
import tempfile

_STUB_CREDENTIALS = {
    "type": "service_account",
    "project_id": "test-project",
    "private_key_id": "stub",
    "client_email": "stub@test-project.iam.gserviceaccount.com",
}

_credentials_file = tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False
)
json.dump(_STUB_CREDENTIALS, _credentials_file)
_credentials_file.close()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _credentials_file.name
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ.setdefault("CLOUD_ML_REGION", "us-east5")
