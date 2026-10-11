"""Object storage: raw ticket payloads live in an S3 bucket.

The second persistence tier, behind the
:class:`~support_triage.ports.ObjectStore` port. Records (Postgres)
hold the decision; the bucket holds the raw intake payloads as
received — the audit copy the record was decided from.

Two implementations:

- :class:`S3ObjectStore` — any S3-compatible endpoint, signed with
  AWS Signature Version 4 using only the standard library (no SDK
  dependency). Point ``S3_ENDPOINT_URL`` at AWS S3, a self-hosted
  gateway (the compose stack runs SeaweedFS's S3 gateway), or another
  compatible store. Signatures are computed over a canonical
  ``UNSIGNED-PAYLOAD`` request — the deployments this prototype
  targets accept that form.
- :class:`InMemoryObjectStore` — a dict. A TEST DOUBLE: objects
  vanish on restart and the API says so at startup.

:func:`default_object_store` builds the S3 store when ``S3_BUCKET``
is set, the double otherwise.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from .config import env_str, load_dotenv
from .ports import ObjectStore as ObjectStorePort  # noqa: F401  (documentation export)

_ALGORITHM = "AWS4-HMAC-SHA256"


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def _signing_key(secret: str, date_stamp: str, region: str) -> bytes:
    k_date = _hmac_sha256(("AWS4" + secret).encode(), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, "s3")
    return _hmac_sha256(k_service, "aws4_request")


class S3ObjectStore:
    """S3-compatible object store over stdlib HTTP + SigV4 signing."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
    ) -> None:
        aws_access = env_str("AWS_ACCESS_KEY_ID") or ""
        aws_secret = env_str("AWS_SECRET_ACCESS_KEY") or ""
        aws_token = env_str("AWS_SESSION_TOKEN") or ""
        aws_region = env_str("AWS_DEFAULT_REGION") or env_str("AWS_REGION") or ""
        if endpoint_url:
            # Gateway credentials pair with the endpoint (compose/SDK style).
            self._access_key = aws_access or access_key or ""
            self._secret_key = aws_secret or secret_key or ""
            self._session_token = aws_token or session_token or ""
        else:
            # Endpoint-less means real AWS: the AWS_* env is authoritative;
            # constructor-passed creds are ignored so a stale .env can't
            # shadow the deployment's identity.
            self._access_key = aws_access
            self._secret_key = aws_secret
            self._session_token = aws_token
        self._bucket = bucket
        self._endpoint = (endpoint_url or "").rstrip("/")
        self._region = region or aws_region or "us-east-1"

    # -- signing / addressing -------------------------------------------

    def _host_and_path(self, key: str) -> tuple[str, str]:
        encoded_key = urllib.parse.quote(key, safe="/~")
        if self._endpoint:
            # Path-style for gateways (SeaweedFS/MinIO-style endpoints).
            parsed = urllib.parse.urlparse(self._endpoint)
            scheme = parsed.scheme or "https"
            return parsed.netloc, f"{scheme}://{parsed.netloc}/{self._bucket}/{encoded_key}"
        # Virtual-host style for AWS proper.
        host = f"{self._bucket}.s3.{self._region}.amazonaws.com"
        return host, f"https://{host}/{encoded_key}"

    def _signed_request(
        self, method: str, key: str, data: bytes | None, content_type: str | None
    ) -> urllib.request.Request:
        host, url = self._host_and_path(key)
        now = datetime.now(timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        payload_hash = "UNSIGNED-PAYLOAD"
        headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
        if self._session_token:
            headers["x-amz-security-token"] = self._session_token
        if content_type:
            headers["content-type"] = content_type
        signed_headers = ";".join(sorted(headers))
        canonical_headers = "".join(f"{name}:{headers[name]}\n" for name in sorted(headers))
        canonical_uri = urllib.parse.urlparse(url).path
        canonical_request = "\n".join(
            [method, canonical_uri, "", canonical_headers, signed_headers, payload_hash]
        )
        scope = f"{date_stamp}/{self._region}/s3/aws4_request"
        string_to_sign = "\n".join(
            [_ALGORITHM, amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest()]
        )
        signature = hmac.new(
            _signing_key(self._secret_key, date_stamp, self._region),
            string_to_sign.encode(),
            hashlib.sha256,
        ).hexdigest()
        authorization = (
            f"{_ALGORITHM} Credential={self._access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        request = urllib.request.Request(url, data=data, method=method)
        for name, value in headers.items():
            request.add_header(name, value)
        request.add_header("Authorization", authorization)
        return request

    # -- the port ---------------------------------------------------------

    def ensure_bucket(self) -> None:
        """Create the bucket (idempotent against compatible gateways).

        A PUT on the bucket path itself — which is what the signed
        request for an empty key addresses.
        """
        if self.exists(""):
            return
        request = self._signed_request("PUT", "", None, None)
        try:
            with urllib.request.urlopen(request, timeout=15):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code not in (409, 200):  # already exists is fine
                raise

    def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str:
        request = self._signed_request("PUT", key, data, content_type)
        with urllib.request.urlopen(request, timeout=30):
            pass
        return key

    def get(self, key: str) -> bytes | None:
        request = self._signed_request("GET", key, None, None)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def exists(self, key: str) -> bool:
        if key == "":
            # Bucket probe: HEAD on the bucket path (empty key).
            request = self._signed_request("HEAD", "", None, None)
            try:
                with urllib.request.urlopen(request, timeout=15):
                    return True
            except urllib.error.HTTPError as exc:
                return exc.code == 409  # some gateways answer HEAD-bucket oddly
            except Exception:
                return False
        request = self._signed_request("HEAD", key, None, None)
        try:
            with urllib.request.urlopen(request, timeout=15):
                return True
        except urllib.error.HTTPError:
            return False


class InMemoryObjectStore:
    """In-memory object store — TEST DOUBLE only."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str:
        with self._lock:
            self._objects[key] = data
        return key

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._objects.get(key)

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._objects


def default_object_store():
    """S3 when S3_BUCKET is set; the in-memory double otherwise."""
    load_dotenv()
    bucket = env_str("S3_BUCKET", "")
    if bucket:
        store = S3ObjectStore(
            bucket,
            endpoint_url=env_str("S3_ENDPOINT_URL"),
            region=env_str("S3_REGION", "us-east-1") or "us-east-1",
            access_key=env_str("S3_ACCESS_KEY_ID"),
            secret_key=env_str("S3_SECRET_ACCESS_KEY"),
        )
        try:
            store.ensure_bucket()
        except Exception:
            # Bucket provisioning is best-effort at startup; a failed
            # ensure surfaces on first real put with a clearer error.
            pass
        return store
    return InMemoryObjectStore()
