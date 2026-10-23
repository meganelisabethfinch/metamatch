from __future__ import annotations

from urllib.parse import urlparse

import requests


def describe(error: requests.RequestException) -> str:
    # a one-line reason for a failed request, e.g. "can't resolve host wikidata-reconciliation.wmcloud.org (DNS)"
    url = getattr(error.request, "url", None) or ""
    host = urlparse(url).hostname or url or "the server"
    text = str(error)
    if "NameResolutionError" in text or "Failed to resolve" in text:
        return f"can't resolve host {host} (DNS)"
    if isinstance(error, requests.Timeout):
        return f"timed out waiting for {host}"
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code} from {host}"
    if isinstance(error, requests.ConnectionError):
        return f"can't connect to {host}"
    return f"{type(error).__name__}: {text[:200]}"
