"""Verify range resumption, exact bytes and rejection of incorrect asset digests."""

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from image_editing.transport import download_file, download_ranges


@pytest.fixture
def range_server():
    payload = b"abcdefgh" * (512 * 1024) + b"tail"
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/file")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            requests.append((start, end))
            body = payload[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            if self.path == "/interrupted" and start == 0:
                self.wfile.write(body[: len(body) // 2])
                self.close_connection = True
            else:
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/redirect", payload, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_completed_ranges_are_reused(tmp_path, range_server):
    url, payload, requests = range_server
    partial = tmp_path / "asset.partial"
    chunks = tmp_path / "asset.partial.chunks"
    chunks.mkdir()
    (chunks / "0").write_bytes(payload[: 4 * 1024**2])
    download_ranges(url, partial, len(payload), connections=2)
    assert partial.read_bytes() == payload
    assert requests == [(4 * 1024**2, len(payload) - 1)]


def test_bad_checksum_does_not_publish_an_asset(tmp_path, range_server, monkeypatch):
    from image_editing import transport

    url, payload, _ = range_server
    monkeypatch.setattr(transport, "endpoint", lambda: url.removesuffix("/redirect"))
    item = dict(
        rfilename="weights.bin",
        size=len(payload),
        lfs=dict(sha256=hashlib.sha256(b"different payload").hexdigest()),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_file("test/repo", "revision", item, tmp_path, connections=2)
    assert not list(tmp_path.iterdir())


def test_interrupted_range_resumes_from_received_bytes(tmp_path, range_server):
    url, payload, requests = range_server
    partial = tmp_path / "asset.partial"
    download_ranges(url.replace("/redirect", "/interrupted"), partial, len(payload), connections=1)
    assert partial.read_bytes() == payload
    assert requests[:2] == [(0, 4 * 1024**2 - 1), (2 * 1024**2, 4 * 1024**2 - 1)]
