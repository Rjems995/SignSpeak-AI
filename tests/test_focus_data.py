"""Public focus-data provenance and isolation; no model predictions or extraction."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ml.download_aslyset import SHA256 as ASLYSET_SHA256
from ml.features import LABELS, normalize

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    with (ROOT / name).open(encoding='utf-8') as stream:
        return json.load(stream)


def digest(name):
    with (ROOT / name).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@pytest.fixture(scope='module')
def focus():
    data = load('datasets/focus-depth.json')
    manifest = load('datasets/focus-depth-selection.json')
    original = load('datasets/static-depth.json')['samples']
    old_validation = {row['fingerprint']: row for row in original if row['split'] == 'validation'}
    fields = ['source_file', 'source_sha256', 'fingerprint', 'participant', 'letter']
    old_training = {row['fingerprint']: {key: row[key] for key in fields}
                    for row in original if row['split'] == 'train'}
    old_test = [{key: row[key] for key in fields} for row in original if row['split'] == 'test']
    del original
    balanced = [{key: row[key] for key in fields} for row in load('datasets/public-balanced.json')]
    return data, manifest, old_training, old_validation, old_test, balanced


def test_focus_counts_and_fresh_image_holdout_are_explicit(focus):
    data, manifest, _, _, old_test, _ = focus
    counts = Counter(row['split'] for row in data['samples'])
    assert counts == data['selected_counts'] == {'train': 8980, 'validation': 1224, 'fresh_test': 840}
    assert 'previously assessed' in data['holdout_type'] and 'not new unseen signers' in data['holdout_type']
    fresh = [row for row in data['samples'] if row['split'] == 'fresh_test']
    people = {row['participant'] for row in old_test}
    assert people == set(manifest['partition_audit']['participant_ids']['test'])
    assert {row['participant'] for row in fresh} == people == {'asl_hg_p8', 'asl_hg_p10', 'aslyset_user2'}
    cells = Counter((row['participant'], row['letter']) for row in fresh)
    assert len(cells) == len(people) * len(LABELS)
    for person in people:
        for letter in LABELS:
            assert cells[(person, letter)] == (20 if letter in 'EMRT' else 10)
    letters = Counter(row['letter'] for row in fresh)
    assert all(letters[letter] == (60 if letter in 'EMRT' else 30) for letter in LABELS)
    assert not data['fresh_test_gaps']


def test_only_frozen_training_people_and_eligible_original_sources_enter_training(focus):
    data, manifest, old_training, _, _, _ = focus
    train = [row for row in data['samples'] if row['split'] == 'train']
    assert {row['participant'] for row in train} == set(manifest['partition_audit']['participant_ids']['train'])
    assert {row['source'] for row in data['samples']} == {'ASLYset', 'ASL-HG'}
    assert all(row.get('training_eligible') is not False for row in data['samples'])
    assert all(row['label_review'] == 'dataset_provided' and row['license'] == 'CC BY 4.0'
               for row in data['samples'])
    new = [row for row in train if row['fingerprint'] not in old_training]
    assert len(new) == 3188
    assert all(row['source'] == 'ASL-HG' and row['letter'] in 'EMRTSNUV' for row in new)
    assert all(row['hand'] == row['lighting'] == row['angle'] == 'unknown' for row in new)


def test_fresh_images_never_overlap_the_original_balanced_images_or_fingerprints(focus):
    data, _, _, _, _, balanced = focus
    fresh = [row for row in data['samples'] if row['split'] == 'fresh_test']
    for field in ['source_file', 'source_sha256', 'fingerprint']:
        assert not {row[field] for row in fresh} & {row[field] for row in balanced}, field


def test_all_final_fingerprints_and_image_hashes_are_unique_and_people_are_isolated(focus):
    data = focus[0]
    rows = data['samples']
    for field in ['source_sha256', 'fingerprint']:
        assert len({row[field] for row in rows}) == len(rows), field
    for field in ['participant', 'source_sha256', 'fingerprint']:
        values = {split: {row[field] for row in rows if row['split'] == split}
                  for split in ['train', 'validation', 'fresh_test']}
        assert not values['train'] & values['validation']
        assert not values['train'] & values['fresh_test']
        assert not values['validation'] & values['fresh_test']


def test_cross_person_duplicate_groups_are_quarantined_including_old_training(focus):
    data, manifest, old_training, _, _, _ = focus
    quarantine = data['quarantine']['source_image_hashes']
    assert len(quarantine) == len(manifest['image_hash_quarantine']) == 100
    assert len({group['source_sha256'] for group in quarantine}) == 100
    final_hashes = {row['source_sha256'] for row in data['samples']}
    for group in quarantine:
        assert group['source_sha256'] not in final_hashes
        rows = group['rows']
        assert {row['participant'] for row in rows} == {'asl_hg_p5', 'asl_hg_p6'}
        assert {row['letter'] for row in rows} == {'V'}
        assert {row['split'] for row in rows} == {'train'}
        assert any(len({row.get(field) for row in rows}) > 1 for field in ['participant', 'split', 'letter'])
    removed = data['removed_old_rows']
    assert len(removed) == 36
    assert {row['source_file'] for row in removed} == set(manifest['old_hash_quarantine_files'])
    assert all(row['split'] == 'train' and row['letter'] == 'V' for row in removed)
    kept = {row['fingerprint'] for row in data['samples'] if row['split'] == 'train'}
    assert set(old_training) - kept == {row['fingerprint'] for row in removed}
    assert not data['quarantine']['canonical_xy_fingerprints']


def test_frozen_validation_and_old_test_exclusion_remain_unchanged(focus):
    data, _, _, old_validation, old_test, _ = focus
    current = {row['fingerprint']: row for row in data['samples'] if row['split'] == 'validation'}
    assert current == old_validation
    assert data['frozen_old_test'] == old_test
    assert not {row['fingerprint'] for row in data['samples'] if row['split'] == 'fresh_test'} & {
        row['fingerprint'] for row in old_test}


def test_real_xyz_and_world_geometry_is_finite_and_xy_fingerprints_still_match(focus):
    data = focus[0]
    for row in data['samples']:
        assert len(row['points']) == len(row['world_points']) == 63
        assert np.isfinite(row['points']).all() and np.isfinite(row['world_points']).all()
        features = normalize(row['points'])  # The canonical fingerprint intentionally uses XY only.
        assert hashlib.sha256(features.round(5).tobytes()).hexdigest() == row['fingerprint']
        assert len(row['source_sha256']) == 64 and int(row['source_sha256'], 16) >= 0
    extractor = data['extractor']
    assert extractor['mediapipe_version'] == '0.10.35' and extractor['mode'] == 'IMAGE'
    assert extractor['num_hands'] == 2
    assert extractor['detection_confidence'] == extractor['presence_confidence'] == .6
    assert extractor['task_sha256'] == 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'


def test_frozen_manifest_dataset_and_source_archive_hashes_match(focus):
    data, manifest, _, _, _, _ = focus
    assert data['selection_manifest_sha256'] == digest('datasets/focus-depth-selection.json')
    assert manifest['static_depth_sha256'] == digest('datasets/static-depth.json')
    assert manifest['balanced_dataset_sha256'] == digest('datasets/public-balanced.json')
    source = load('datasets/public-sources.json')['asl_hg']
    assert manifest['archive_sha256'] == {'ASL-HG': source['sha256'], 'ASLYset': ASLYSET_SHA256}
    for name, expected in manifest['partition_audit']['source_hashes'].items():
        assert digest('datasets/' + name) == expected
