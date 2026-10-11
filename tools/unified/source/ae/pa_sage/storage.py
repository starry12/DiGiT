"""Check existing SSD receipts and plan a separate g2 range without device writes."""
from pathlib import Path
from ae.pa_sage.common import *
from ae.common import check_payload


def audit(data):
    data = Path(data)
    prepared = check_prepared(data)
    manifest = read(data/'final/bundle/manifest.json')
    payload_bytes = manifest['feature']['num_storage_rows'] * manifest['feature']['row_bytes']
    device = read(ROOT/'configs/device.json')
    recorded = []
    paths = list((ROOT/'ssd_state').glob('*.json')) + list((ROOT/'data/papers/ssd').glob('*state.json'))
    production = Path(digit_paths.locations()['storage_root'])/'ssd_state'
    if production.is_dir():
        paths += list(production.glob('*.json'))
    for path in paths:
        entry = read(path)
        offset = entry.get('device_offset_bytes', 0)
        size = entry.get('payload_bytes')
        if size is not None:
            recorded.append(dict(state=str(path),state_sha256=sha(path),offset=offset,
                                 end=offset+size,status=entry.get('status','unknown')))
    candidate = 3 * 2**40
    overlaps = [item for item in recorded if candidate < item['end'] and candidate+payload_bytes > item['offset']]
    baseline = check_payload('papers_gids')
    expected = read(ROOT/'configs/ssd_payloads.json')['papers_full']
    # Old g8 physical bytes must never be relabeled as the new g2 layout.
    require(manifest['grouping']['group_size'] == 2, 'Wrong bundle')
    require(manifest['files']['reordered_features']['sha256'] != expected['file_sha256'], 'Unexpected old payload identity')
    return dict(schema='digit-pa-g2-ssd-audit-v1', protocol_sha256=sha(PROTOCOL),
        filesystem_payload_verified=prepared['filesystem_payload_verified'],
        new_payload=dict(file=str(data/'final/bundle/reordered_features.npy'),
                         file_sha256=manifest['files']['reordered_features']['sha256'],
                         payload_bytes=payload_bytes,manifest_sha256=sha(data/'final/bundle/manifest.json')),
        baseline_historical_receipt_valid=True,baseline_offset=baseline['offset'],
        existing_g8_payload_reusable=False,recorded_ranges=recorded,
        proposed_range=dict(offset=candidate,end=candidate+payload_bytes,
                            within_capacity=candidate+payload_bytes<=device['capacity_bytes'],
                            alignment_bytes=4096,recorded_overlaps=overlaps),
        allocation_status='proposal_only_not_reserved_or_authorized_for_write',
        limitation='Recorded ranges are not proof that unregistered device ranges are unused. Recheck all users/ranges before any future payload write.',
        new_g2_payload_written=False,new_g2_full_raw_readback_passed=False,
        raw_ssd_read=False,raw_ssd_write=False,full_training_ready=False)
