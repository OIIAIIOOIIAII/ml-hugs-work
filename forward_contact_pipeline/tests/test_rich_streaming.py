"""Ensure provisional extraction cannot consume holes or bypass full gzip CRC."""
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import struct
import sys
import tarfile
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.rich_streaming import PiecePrefixReader, complete_prefix, image_identity, jpeg_dimensions, stream_jpg


def write_control(path, size, piece, complete):
    count = math.ceil(size / piece); bits = bytearray(math.ceil(count / 8))
    for i in complete:
        bits[i // 8] |= 128 >> (i % 8)
    data = struct.pack(">HIIIQQI", 1, 0, 0, piece, size, 0, len(bits)) + bits
    temporary = path.with_suffix(".tmp"); temporary.write_bytes(data); temporary.replace(path)


def jpeg_header_fixture():
    # Numerical JPEG marker fixture only; no pixel decoding is needed/performed.
    return b"\xff\xd8\xff\xc0" + struct.pack(">HBHHB", 11, 8, 480, 640, 1) + b"\x01\x11\x00\xff\xda" + random.Random(17).randbytes(5000) + b"\xff\xd9"


def tar_fixture(name="train/seq/cam_00/00001_00.jpeg"):
    payload = jpeg_header_fixture(); buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        item = tarfile.TarInfo(name); item.size = len(payload)
        archive.addfile(item, io.BytesIO(payload))
    return gzip.compress(buffer.getvalue(), mtime=0), payload


class RichStreamingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs", prefix="rich_stream_")
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        (self.root / "raw/train").mkdir(parents=True)
        catalog = self.root / "processed/annotations_v1/catalog.json"; catalog.parent.mkdir(parents=True)
        catalog.write_text(json.dumps({"tracks": [{"sequence": "seq", "subject": "001", "frames": [1], "body_only_frames": [], "camera_ids": [0]}]}))

    def test_only_contiguous_completed_pieces_are_read(self):
        partial, final, control = [self.root / name for name in ("x.part", "x.gz", "x.part.aria2")]
        actual = b"abcdefghABCDEFGH12345678"
        partial.write_bytes(actual[:8] + b"\0" * 8 + actual[16:])
        write_control(control, len(actual), 8, [0, 2])
        self.assertEqual(complete_prefix(control, len(actual)), 8)
        waits = []
        reader = PiecePrefixReader(partial, final, control, len(actual), poll_seconds=.005, on_wait=lambda _: waits.append(1))
        self.addCleanup(reader.close)
        self.assertEqual(reader.read(24), actual[:8])

        def finish():
            time.sleep(.05)
            with partial.open("r+b") as stream:
                stream.seek(8); stream.write(actual[8:16])
            partial.replace(final)
        worker = threading.Thread(target=finish); worker.start()
        self.assertEqual(reader.read(24), actual[8:]); worker.join()
        self.assertTrue(waits); self.assertEqual(reader.read(1), b"")
        self.assertEqual(reader.digest.hexdigest(), hashlib.sha256(actual).hexdigest())

    def test_wait_timeout_never_returns_hole_as_eof(self):
        partial, final, control = [self.root / name for name in ("x.part", "x.gz", "x.part.aria2")]
        partial.write_bytes(b"\0" * 16); write_control(control, 16, 8, [])
        reader = PiecePrefixReader(partial, final, control, 16, wait_hours=0, poll_seconds=.001)
        self.addCleanup(reader.close)
        with self.assertRaises(TimeoutError):
            reader.read(1)

    def test_stream_waits_then_publishes_verified_members_and_index(self):
        data, payload = tar_fixture(); raw = self.root / "raw/train"
        partial, final = raw / "train.tar.gz.part", raw / "train.tar.gz"
        control = raw / "train.tar.gz.part.aria2"
        partial.write_bytes(data[:1024] + b"\0" * (len(data) - 1024)); write_control(control, len(data), 1024, [0])

        def finish():
            time.sleep(.05)
            with partial.open("r+b") as stream:
                stream.seek(1024); stream.write(data[1024:])
            partial.replace(final)
        worker = threading.Thread(target=finish); worker.start()
        state = stream_jpg(self.root, expected_bytes=len(data), poll_seconds=.005); worker.join()
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["archive_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual((self.root / "extracted/train/seq/cam_00/00001_00.jpeg").read_bytes(), payload)
        record = json.loads(Path(state["index"]).read_text())
        self.assertEqual((record["width"], record["height"]), (640, 480))
        self.assertEqual(record["subjects_with_contact"], ["001"])
        self.assertFalse(state["training_ready"])
        self.assertEqual(stream_jpg(self.root, expected_bytes=len(data))["status"], "complete")

    def test_bad_gzip_crc_keeps_output_provisional(self):
        data, _ = tar_fixture(); damaged = bytearray(data); damaged[-8] ^= 1
        (self.root / "raw/train/train.tar.gz").write_bytes(damaged)
        with self.assertRaises((gzip.BadGzipFile, OSError)):
            stream_jpg(self.root, expected_bytes=len(data))
        self.assertFalse((self.root / "extracted/train").exists())
        state = json.loads((self.root / "processing_logs/train_jpg.json").read_text())
        self.assertEqual(state["status"], "failed")

    def test_complete_pieces_wait_for_final_name_and_restart_replays(self):
        data, _ = tar_fixture(); raw = self.root / "raw/train"
        partial = raw / "train.tar.gz.part"; partial.write_bytes(data)
        write_control(raw / "train.tar.gz.part.aria2", len(data), 1024, range(math.ceil(len(data) / 1024)))
        with self.assertRaises(TimeoutError):
            stream_jpg(self.root, expected_bytes=len(data), wait_hours=0)
        self.assertFalse((self.root / "extracted/train").exists())
        partial.replace(raw / "train.tar.gz")
        state = stream_jpg(self.root, expected_bytes=len(data))
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["files"], 1)
        self.assertEqual(len(Path(state["index"]).read_text().splitlines()), 1)

    def test_unsafe_tar_member_is_rejected(self):
        data, _ = tar_fixture("train/../../outside.jpeg")
        (self.root / "raw/train/train.tar.gz").write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            stream_jpg(self.root, expected_bytes=len(data))
        self.assertFalse((self.root / "extracted/train").exists())
        self.assertFalse((self.root / "outside.jpeg").exists())

    def test_jpeg_header_and_camera_identity_validation(self):
        path = self.root / "headers.jpeg"; path.write_bytes(jpeg_header_fixture())
        self.assertEqual(jpeg_dimensions(path)["width"], 640)
        path.write_bytes(b"\xff\xd8\xff\xd9")
        with self.assertRaisesRegex(ValueError, "no SOF"):
            jpeg_dimensions(path)
        with self.assertRaisesRegex(ValueError, "camera differs"):
            image_identity("seq/cam_00/00001_01.jpeg", {})


if __name__ == "__main__":
    unittest.main()
