import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.rich_assets import extract_archive


class RichAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "runs", prefix="rich_extract_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_zip_integrity_and_idempotence(self):
        source = self.root / "body.zip"
        with zipfile.ZipFile(source, "w") as z:
            z.writestr("train_body/seq/00001/001.txt", "numerical fixture")
        state = self.root / "state.json"
        first = extract_archive(source, self.root / "extracted", state, "body")
        self.assertEqual(first["files"], 1)
        self.assertEqual(extract_archive(source, self.root / "extracted", state, "body"), first)
        self.assertEqual((Path(first["output"]) / "seq/00001/001.txt").read_text(), "numerical fixture")

    def test_unsafe_paths_are_rejected_without_publication(self):
        source = self.root / "bad.zip"
        with zipfile.ZipFile(source, "w") as z:
            z.writestr("../escape.txt", "bad")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            extract_archive(source, self.root / "extracted", self.root / "state.json", "bad")
        self.assertFalse((self.root / "escape.txt").exists())

    def test_tar_gzip_trailer_corruption_never_publishes(self):
        plain = io.BytesIO()
        with tarfile.open(fileobj=plain, mode="w") as archive:
            item = tarfile.TarInfo("train/seq/cam_00/numeric.txt"); item.size = 3
            archive.addfile(item, io.BytesIO(b"123"))
        source = self.root / "images.tar.gz"; compressed = gzip.compress(plain.getvalue())
        source.write_bytes(compressed)
        result = extract_archive(source, self.root / "ok", self.root / "ok.json", "images")
        self.assertEqual(result["files"], 1)
        damaged = bytearray(compressed); damaged[-8] ^= 1; source.write_bytes(damaged)
        with self.assertRaises((gzip.BadGzipFile, OSError)):
            extract_archive(source, self.root / "bad", self.root / "bad.json", "images")
        self.assertFalse((self.root / "bad/train").exists())


if __name__ == "__main__":
    unittest.main()
