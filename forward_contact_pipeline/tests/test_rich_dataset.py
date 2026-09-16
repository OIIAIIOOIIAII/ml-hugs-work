"""Checks for dataset leakage, real frame joins, masks and metadata-only reads."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, JpegImagePlugin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.rich_assets import save_json, sha256
from contact_streaming.rich_dataset import (audit_cameras, choose_clip, freeze_splits,
    image_header, index_shards, inside, prepare_dataset, projection_summary, validate_image_identities)


def jpeg_header(width=100, height=80, orientation=1):
    exif = Image.Exif(); exif[274] = orientation
    app = exif.tobytes()
    sof = struct.pack('>BHHB', 8, height, width, 3) + b'\x01\x11\x00\x02\x11\x00\x03\x11\x00'
    # No encoded pixels: opening headers works, decoding must fail.
    return (b'\xff\xd8\xff\xe1' + struct.pack('>H', len(app) + 2) + app
            + b'\xff\xc0' + struct.pack('>H', len(sof) + 2) + sof
            + b'\xff\xda\x00\x0c\x03\x01\x00\x02\x00\x03\x00\x00\x3f\x00')


def fixture(root):
    base = root / 'processed/annotations_v1'; base.mkdir(parents=True)
    tracks = [dict(sequence=name, capture=name, scan='scan', subject=str(i), gender='male',
                   frames=[5, 7], camera_ids=[0], body_only_frames=[])
              for i, name in enumerate(['A', 'B'])]
    config = dict(schema_version=1, name='fixture', source_split='official_train',
                  capture_splits={'A': 'train', 'B': 'val'}, label_space='smplx_10475',
                  require_official_camera=True, require_identity_exif=True, pilot_clips=[])
    catalog = dict(tracks=tracks, metadata={'revision': 'fixture'}, excluded_missing_contact=['A/00008/0'])
    save_json(base / 'catalog.json', catalog)
    np.savez(base / 'smplx_topology.npz', faces=np.array([[0, 1, 2]]))
    state = dict(status='complete', mode='full', completed_frames=4, shards=[],
                 catalog_signature=hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
                 topology_sha256=sha256(base / 'smplx_topology.npz'))
    for track in tracks:
        filename = track['sequence'] + '.npz'
        valid = np.ones((2, 10475), dtype=bool); valid[1] = False
        vertices = np.zeros((2, 10475, 3), dtype=np.float32); vertices[:, :, 2] = 2
        np.savez_compressed(base / filename, frame_ids=[5, 7], contact_smplx=np.zeros_like(valid),
                            contact_smplx_valid=valid, contact_smpl_valid=np.ones((2, 6890), bool),
                            vertices_multicam=vertices)
        state['shards'].append(dict(sequence=track['sequence'], subject=track['subject'], frames=[5, 7],
                                   path=filename, sha256=sha256(base / filename), missing_smplx_color_frames=[7]))
    save_json(base / 'state.json', state)
    camera = dict(Intrinsics=[[60, 0, 50], [0, 60, 40], [0, 0, 1]],
                  CameraMatrix=np.eye(3, 4).tolist(), Distortion=[[0]] * 8)
    calibration = {'captures': {t['capture'] + '/scan': {'cameras': {'0': camera}} for t in tracks}}
    save_json(base / 'calibration.json', calibration)
    rows = []
    for track in tracks:
        for frame in [5, 7]:
            for cam in [0, 10]:
                relative = f"{track['sequence']}/cam_{cam:02d}/{frame:05d}_{cam:02d}.jpeg"
                path = root / 'extracted/train' / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(jpeg_header())
                rows.append(dict(path=relative, sequence=track['sequence'], camera_id=cam, frame_id=frame,
                                 bytes=path.stat().st_size, sha256=sha256(path), width=100, height=80))
    images = root / 'processed/images_v1'; images.mkdir()
    (images / 'images.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    save_json(root / 'processing_logs/train_jpg.json', dict(status='complete', archive_integrity='verified',
        image_files=len(rows), uncompressed_bytes=sum(r['bytes'] for r in rows), archive_sha256='fixture'))
    save_json(root / 'config.json', config)
    return tracks, config, state, rows, calibration


class RichDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'runs')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_join_filters_and_relocation(self):
        fixture(self.root)
        output = self.root / 'processed/dataset_v1'
        with patch.object(JpegImagePlugin.JpegImageFile, 'load', side_effect=AssertionError('pixel decode forbidden')):
            report = prepare_dataset(self.root, self.root / 'config.json', output, 2)
        self.assertFalse(report['training_ready'])
        self.assertEqual(report['valid_smplx_frames'], 2)
        self.assertEqual(report['counts']['excluded_reason_missing_calibration'], 4)
        self.assertEqual(report['counts']['excluded_reason_missing_smplx_contact_colors'], 4)
        samples = [json.loads(x) for x in (output / 'samples_train.jsonl').read_text().splitlines()]
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]['frame_id'], 5)
        self.assertEqual(samples[0]['annotation']['row'], 0)
        self.assertNotIn(str(self.root), (output / 'samples_train.jsonl').read_text())
        renamed = self.root / 'relocated'; renamed.mkdir()
        for path in ['processed', 'extracted', 'processing_logs', 'config.json']:
            (self.root / path).rename(renamed / path)
        self.assertEqual(prepare_dataset(renamed, renamed / 'config.json', renamed / 'processed/dataset_v1'), report)
        (renamed / 'processed/dataset_v1/samples_train.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            prepare_dataset(renamed, renamed / 'config.json', renamed / 'processed/dataset_v1')

    def test_shared_subject_split_is_rejected(self):
        tracks, config, *_ = fixture(self.root)
        tracks[1]['subject'] = tracks[0]['subject']
        with self.assertRaisesRegex(ValueError, 'Split leakage: subject'):
            freeze_splits(tracks, config)

    def test_wrong_frame_or_camera_in_index_is_rejected(self):
        _, _, _, rows, _ = fixture(self.root)
        validate_image_identities(rows)
        for key in ('frame_id', 'camera_id', 'sequence'):
            bad = {**rows[0], key: 'incorrect' if key == 'sequence' else 12345}
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                validate_image_identities([bad])

    def test_nonconsecutive_frame_row_and_tamper_detection(self):
        tracks, _, state, *_ = fixture(self.root)
        indexed = index_shards(self.root, state, tracks, 1)
        self.assertEqual(indexed['A', '0', 7]['row'], 1)
        self.assertEqual(indexed['A', '0', 7]['smplx_valid_vertices'], 0)
        self.assertEqual(indexed['A', '0', 7]['smpl_valid_vertices'], 6890)
        path = self.root / 'processed/annotations_v1/A.npz'
        path.write_bytes(path.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            index_shards(self.root, state, tracks, 1)

    def test_exif_is_read_without_pixels_and_size_mismatch_fails(self):
        _, _, _, rows, _ = fixture(self.root)
        row = rows[0]; path = self.root / 'extracted/train' / row['path']
        path.write_bytes(jpeg_header(orientation=6))
        with patch.object(JpegImagePlugin.JpegImageFile, 'load', side_effect=AssertionError('decode')):
            self.assertEqual(image_header(self.root, row)['orientation'], 6)
        row['width'] = 80
        with self.assertRaisesRegex(ValueError, 'header disagrees'):
            image_header(self.root, row)

    def test_rotated_only_image_cannot_publish_ready_supervision(self):
        _, _, _, rows, _ = fixture(self.root)
        for row in rows:
            (self.root / 'extracted/train' / row['path']).write_bytes(jpeg_header(orientation=6))
        output = self.root / 'processed/dataset_v1'
        with self.assertRaisesRegex(ValueError, 'lack a calibrated identity-EXIF image'):
            prepare_dataset(self.root, self.root / 'config.json', output, 1)
        self.assertFalse(output.exists())
        # Replacing a source invalidates cached EXIF metadata on retry.
        for row in rows:
            (self.root / 'extracted/train' / row['path']).write_bytes(jpeg_header(orientation=1))
        self.assertTrue(prepare_dataset(self.root, self.root / 'config.json', output, 1)['supervision_index_ready'])

    def test_cached_header_rechecks_stat_without_decoding(self):
        _, _, _, rows, _ = fixture(self.root)
        row = rows[0]; cached = image_header(self.root, row)
        with patch.object(Image, 'open', side_effect=AssertionError('unchanged cached file reopened')):
            self.assertEqual(image_header(self.root, row, cached), cached)
        path = self.root / 'extracted/train' / row['path']
        path.write_bytes(jpeg_header(orientation=8))
        self.assertEqual(image_header(self.root, row, cached)['orientation'], 8)

    def test_portrait_camera_and_projection(self):
        tracks, _, _, rows, calibration = fixture(self.root)
        row = dict(rows[0], width=80, height=100)
        calibration['captures']['A/scan']['cameras']['0']['Intrinsics'] = [[60, 0, 40], [0, 60, 50], [0, 0, 1]]
        result = audit_cameras([row], [{'orientation': 1}], tracks, calibration)
        camera = result['cameras']['A/scan/cam_00']
        stats = projection_summary(np.array([[0, 0, 2], [0, 0, -1]]), camera)
        self.assertEqual(stats['front_fraction'], .5)
        self.assertEqual(stats['in_image_fraction'], .5)
        self.assertFalse(result['pixel_alignment_verified'])

    def test_clip_preserves_gaps_and_real_frames(self):
        spec = dict(split='train', sequence='seq', subject='001', camera_id=0, length=3)
        rows = [{**spec, 'frame_id': f} for f in [5, 7, 8, 9]]
        self.assertEqual([r['frame_id'] for r in choose_clip(rows, spec)['frames']], [7, 8, 9])
        with self.assertRaisesRegex(ValueError, 'No consecutive'):
            choose_clip(rows, {**spec, 'length': 4})

    def test_unsafe_paths_rejected(self):
        with self.assertRaises(ValueError):
            inside(self.root, '../outside')
        (self.root / 'escape').symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            inside(self.root, 'escape/file')


if __name__ == '__main__':
    unittest.main()
