"""Post-deploy smoke test (M4): health check + one real prediction call.

Uses only the Python standard library so it runs anywhere (CI runner,
laptop) without installing dependencies. Exits non-zero on any failure,
which fails the CD pipeline.

Usage:
    python scripts/smoke_test.py [--base-url http://localhost:8000]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SAMPLE_IMAGE = Path(__file__).parent.parent / "tests" / "assets" / "sample.jpg"


def get_json(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def post_image(url: str, image_path: Path, timeout: int = 30) -> dict:
    """POST a multipart/form-data image using only the standard library."""
    boundary = uuid.uuid4().hex
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'.encode(),
            b"Content-Type: image/jpeg\r\n\r\n",
            image_path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_health(base_url: str, attempts: int = 12, delay: float = 5.0) -> dict:
    """Poll /health until the service is up and the model is loaded."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            health = get_json(f"{base_url}/health")
            if health.get("status") == "ok" and health.get("model_loaded"):
                return health
            last_error = f"unexpected health payload: {health}"
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        print(f"  attempt {attempt}/{attempts}: not ready ({last_error})")
        time.sleep(delay)
    raise RuntimeError(f"service never became healthy: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"[1/2] Health check at {base_url}/health ...")
    health = wait_for_health(base_url)
    print(f"  OK: {health}")

    print(f"[2/2] Prediction call with {SAMPLE_IMAGE.name} ...")
    if not SAMPLE_IMAGE.exists():
        raise RuntimeError(f"sample image missing: {SAMPLE_IMAGE}")
    result = post_image(f"{base_url}/predict", SAMPLE_IMAGE)
    if "label" not in result or "probabilities" not in result:
        raise RuntimeError(f"malformed prediction response: {result}")
    print(f"  OK: {result}")

    print("SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
