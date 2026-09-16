"""Exercise the RICH POST -> session cookie -> redirected Range download flow."""
import importlib.util
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
import urllib.parse
import zipfile
from unittest import mock

PIPELINE = Path(__file__).resolve().parents[1]


class RichSessionDownloadTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "rich_acquisition_under_test", PIPELINE / "scripts/acquire_rich.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temporary = tempfile.TemporaryDirectory(prefix="rich_session_test_", dir=PIPELINE / "runs")
        self.addCleanup(self.temporary.cleanup)
        self.module.ROOT = Path(self.temporary.name)
        self.module.LOGS = self.module.ROOT / "logs"
        self.module.LOGS.mkdir()
        self.module.PRIVATE = self.module.ROOT / "private"
        self.module.PRIVATE.mkdir(mode=0o700)
        self.module.AUTH = self.module.ROOT / "test_credentials.post"
        self.form = urllib.parse.urlencode(
            {"username": "local+test@example.org", "password": "test &+= value", "commit": "Log in"}
        ).encode()
        self.module.AUTH.write_bytes(self.form)
        self.module.AUTH.chmod(0o600)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("numeric.txt", "1234567890" * 20000)
        self.archive = buffer.getvalue()
        self.get_ranges = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(302)
                if body == fixture.form:
                    self.send_header("Set-Cookie", "PHPSESSID=test-session; Path=/")
                self.send_header("Location", self.path)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                if "PHPSESSID=test-session" not in self.headers.get("Cookie", ""):
                    body = b"<html><form>Sign in</form></html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                else:
                    first, last = 0, len(fixture.archive) - 1
                    range_header = self.headers.get("Range")
                    fixture.get_ranges.append(range_header)
                    if range_header:
                        left, right = range_header.removeprefix("bytes=").split("-")
                        first = int(left)
                        last = min(int(right) if right else last, last)
                        self.send_response(206)
                        self.send_header("Content-Range", f"bytes {first}-{last}/{len(fixture.archive)}")
                    else:
                        self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    body = fixture.archive[first:last + 1]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Download clients may close an initial metadata request.

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.item = {
            "name": "session_archive",
            "url": f"http://127.0.0.1:{self.server.server_port}/archive.zip",
            "relative_path": "test/archive.zip",
        }

    @unittest.skipUnless(shutil.which("wget"), "requires wget for actual resume")
    def test_session_survives_redirect_and_wget_resumes(self):
        info = self.module.probe(self.item, self.form)
        self.assertTrue(info["range_supported"])
        self.assertEqual(info["expected_bytes"], len(self.archive))
        target = self.module.ROOT / "raw/test/archive.zip"
        target.parent.mkdir(parents=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(self.archive[:50000])
        self.assertTrue(self.module.download(self.item, self.form, engine="wget"))
        self.assertEqual(target.read_bytes(), self.archive)
        self.assertIn("bytes=50000-", self.get_ranges)
        self.assertFalse(partial.exists())

    def test_login_html_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "login/error page"):
            self.module.probe(self.item, b"username=wrong&password=wrong")

    def test_aria2_reuses_cookie_and_imports_wget_partial(self):
        if not self.module.aria2_binary():
            self.skipTest("aria2 is not installed")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("numeric.txt", "1234567890" * 500000)
        self.archive = buffer.getvalue()
        target = self.module.ROOT / "raw/test/archive.zip"
        target.parent.mkdir(parents=True)
        partial = target.with_name(target.name + ".part")
        prefix_size = 2 * 1024 * 1024 + 12345
        partial.write_bytes(self.archive[:prefix_size])
        self.assertTrue(self.module.download(self.item, self.form, engine="aria2"))
        self.assertEqual(target.read_bytes(), self.archive)
        # aria2 imports full pieces; it may re-fetch the last incomplete piece.
        starts = [int(value.removeprefix("bytes=").split("-")[0])
                  for value in self.get_ranges if value]
        self.assertTrue(any(0 < value <= prefix_size for value in starts), self.get_ranges)
        cookies = self.module.PRIVATE / "session_archive.cookies.txt"
        self.assertEqual(cookies.stat().st_mode & 0o777, 0o600)

    def test_wget_does_not_resume_aria2_sparse_partial(self):
        partial = self.module.ROOT / "raw/test/archive.zip.part"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(self.archive[:50000])
        partial.with_name(partial.name + ".aria2").write_bytes(b"control")
        self.assertFalse(self.module.download(self.item, self.form, engine="wget"))
        self.assertEqual(partial.read_bytes(), self.archive[:50000])

    def test_transfer_retry_refreshes_but_does_not_retry_validation_errors(self):
        state_path = self.module.LOGS / (self.item["name"] + ".json")
        def failure(*args):
            self.module.save_json(state_path, {"error": "aria2 exited 1; see log"})
            return False
        with mock.patch.object(self.module, "download", side_effect=failure) as call, mock.patch.object(self.module.time, "sleep") as sleep:
            self.assertFalse(self.module.download_with_retries(self.item, self.form, rounds=3))
            self.assertEqual(call.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
        self.module.save_json(state_path, {"error": "login/error page"})
        with mock.patch.object(self.module, "download", return_value=False) as call, mock.patch.object(self.module.time, "sleep") as sleep:
            self.assertFalse(self.module.download_with_retries(self.item, self.form, rounds=3))
            self.assertEqual(call.call_count, 1)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
