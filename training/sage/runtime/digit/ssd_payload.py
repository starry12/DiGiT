"""Manifest-driven, guarded write and exact readback for a DiGiT SSD payload."""

# Locate DiGiT independently of the checkout directory and working directory.
from pathlib import Path as _DigitPath
import sys as _digit_sys
_digit_root = next((p for p in _DigitPath(__file__).resolve().parents
                    if (p / ".digit-root").is_file()), None)
if _digit_root is None:
    raise RuntimeError("Cannot locate the DiGiT project root")
if str(_digit_root) not in _digit_sys.path:
    _digit_sys.path.insert(0, str(_digit_root))
import digit_paths as _digit_paths

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping

import numpy as np

from digit.artifacts import ArtifactValidationError, load_artifact_bundle
from digit.io_geometry import IOGeometry


SUPPORTED_DEVICE = "/dev/libnvm0"
WRITE_RECEIPT = "ssd_write_receipt.json"
VERIFY_RECEIPT = "ssd_verify_receipt.json"
ACTIVE_STATE_DIR = "ssd_state"
RECEIPT_SCHEMA_NAME = "digit-ssd-payload-receipt"
RECEIPT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PayloadPlan:
    artifact: str
    feature_file: str
    feature_file_sha256: str
    npy_header_bytes: int
    payload_bytes: int
    storage_rows: int
    feature_dim: int
    dtype: str
    row_bytes: int
    page_size: int
    alignment_rows: int
    payload_pages: int
    num_ele: int
    device: str
    confirmation: str
    device_offset_bytes: int = 0
    artifact_schema_version: int = 0
    artifact_geometry_sha256: str = ""
    runtime_geometry_sha256: str = ""
    runtime_geometry: Mapping = None
    receipt_schema_name: str = RECEIPT_SCHEMA_NAME
    receipt_schema_version: int = RECEIPT_SCHEMA_VERSION


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if sudo_uid and sudo_gid:
        os.chown(path.parent, int(sudo_uid), int(sudo_gid))
        os.chown(path, int(sudo_uid), int(sudo_gid))


def _active_state_path(report_dir: Path, device: str) -> Path:
    return report_dir.resolve().parent / ACTIVE_STATE_DIR / (Path(device).name + ".json")


def _plan_active_state_path(report_dir: Path, plan: PayloadPlan) -> Path:
    if plan.device_offset_bytes == 0:
        return _active_state_path(report_dir, plan.device)
    name = "{}.offset{}.json".format(
        Path(plan.device).name, plan.device_offset_bytes
    )
    return report_dir.resolve().parent / ACTIVE_STATE_DIR / name


def find_active_range_overlaps(plan: PayloadPlan, state_dir: Path) -> List[Mapping]:
    """Return verified/in-progress payloads overlapping ``plan`` on its device.

    State parsing is fail-closed: a malformed registry entry could hide an
    occupied extent and therefore prevents a destructive write.
    """

    directory = Path(state_dir).resolve()
    if not directory.exists():
        return []
    overlaps = []
    plan_start = int(plan.device_offset_bytes)
    plan_end = plan_start + int(plan.payload_bytes)
    for path in sorted(directory.glob("*.json")):
        try:
            state = _digit_paths.json_loads(path.read_text(encoding="utf-8"))
            device = str(state["device"])
            payload_bytes = int(state["payload_bytes"])
            offset = int(state.get("device_offset_bytes", 0))
            status = str(state["status"])
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "cannot safely interpret SSD active-state entry: {}".format(path)
            ) from error
        if payload_bytes <= 0 or offset < 0:
            raise RuntimeError("invalid SSD extent in active-state entry: {}".format(path))
        if device != plan.device or status not in (
            "write_in_progress", "written_unverified", "verified"
        ):
            continue
        state_end = offset + payload_bytes
        if plan_start < state_end and offset < plan_end:
            overlaps.append({
                "state_file": str(path),
                "status": status,
                "artifact": state.get("artifact"),
                "start_bytes": offset,
                "end_bytes_exclusive": state_end,
            })
    return overlaps


def require_unoccupied_active_range(plan: PayloadPlan, state_dir: Path) -> None:
    overlaps = find_active_range_overlaps(plan, state_dir)
    if overlaps:
        raise RuntimeError(
            "refusing SSD write because the target extent overlaps active state:\n{}".format(
                json.dumps(overlaps, indent=2, sort_keys=True)
            )
        )


def _active_binding(plan: PayloadPlan, status: str) -> Mapping:
    binding = {
        "status": status,
        "artifact": plan.artifact,
        "feature_file_sha256": plan.feature_file_sha256,
        "payload_bytes": plan.payload_bytes,
        "storage_rows": plan.storage_rows,
        "page_size": plan.page_size,
        "device": plan.device,
        "artifact_schema_version": plan.artifact_schema_version,
        "artifact_geometry_sha256": plan.artifact_geometry_sha256,
        "runtime_geometry_sha256": plan.runtime_geometry_sha256,
        "receipt_schema_name": plan.receipt_schema_name,
        "receipt_schema_version": plan.receipt_schema_version,
        "updated_at_utc": _utc_now(),
    }
    if plan.device_offset_bytes:
        binding["device_offset_bytes"] = plan.device_offset_bytes
    return binding


def _validate_active_state(
    path: Path, plan: PayloadPlan, status: str, *, require_geometry: bool = True
) -> Mapping:
    try:
        state = _digit_paths.json_loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read active SSD payload state: {}".format(path)) from error
    expected = _active_binding(plan, status)
    if not require_geometry:
        expected = {
            key: value for key, value in expected.items()
            if key not in (
                "artifact_schema_version", "artifact_geometry_sha256",
                "runtime_geometry_sha256", "receipt_schema_name",
                "receipt_schema_version",
            )
        }
    for key, value in expected.items():
        if key != "updated_at_utc" and state.get(key) != value:
            raise RuntimeError("active SSD payload state mismatch: {}".format(key))
    return state


def build_payload_plan(
    artifact, *, device: str = SUPPORTED_DEVICE, verify_checksums: bool = True,
    device_offset_bytes: int = 0,
) -> PayloadPlan:
    bundle = load_artifact_bundle(
        artifact, mmap_mode="r", verify_checksums=verify_checksums
    )
    manifest = bundle.manifest
    entry = manifest["files"]["reordered_features"]
    feature_path = (bundle.root / entry["path"]).resolve()
    features = bundle.arrays["reordered_features"]
    dtype = np.dtype(features.dtype)
    if dtype != np.dtype("float32"):
        raise ArtifactValidationError("GIDS float storage requires float32 features")
    if features.ndim != 2 or not features.flags.c_contiguous:
        raise ArtifactValidationError("reordered features must be contiguous and 2-D")

    header_bytes = int(getattr(features, "offset", 0))
    payload_bytes = int(features.nbytes)
    storage_rows, feature_dim = (int(value) for value in features.shape)
    row_bytes = feature_dim * dtype.itemsize
    page_size = int(manifest["io"]["page_size"])
    alignment_rows = int(manifest["io"]["alignment_rows"])
    if header_bytes <= 0:
        raise ArtifactValidationError("reordered_features.npy has no detectable NPY header")
    if feature_path.stat().st_size != header_bytes + payload_bytes:
        raise ArtifactValidationError("NPY file size does not equal header plus payload")
    if int(manifest["feature"]["row_bytes"]) != row_bytes:
        raise ArtifactValidationError("manifest feature row size is inconsistent")
    if int(manifest["feature"]["num_storage_rows"]) != storage_rows:
        raise ArtifactValidationError("manifest storage-row count is inconsistent")
    if page_size % row_bytes != 0 or page_size // row_bytes != alignment_rows:
        raise ArtifactValidationError("feature rows do not match manifest page alignment")
    if payload_bytes % page_size:
        raise ArtifactValidationError(
            "SSD payload must end on a full page; regenerate the artifact with padding"
        )
    checksum = str(entry["sha256"])
    if not checksum:
        raise ArtifactValidationError("reordered feature checksum is missing")
    device = str(Path(device))
    token = "ERASE-{}-{}-{}".format(
        Path(device).name, checksum[:16], payload_bytes
    )
    if device_offset_bytes < 0 or device_offset_bytes % page_size:
        raise ArtifactValidationError("device offset must be non-negative and page aligned")
    if device_offset_bytes:
        token += "-OFFSET{}".format(device_offset_bytes)
    artifact_geometry = bundle.io_geometry
    runtime_geometry = artifact_geometry.with_payload_offset(device_offset_bytes)
    token += "-GEO{}".format(runtime_geometry.semantic_sha256()[:16])
    return PayloadPlan(
        artifact=str(bundle.root),
        feature_file=str(feature_path),
        feature_file_sha256=checksum,
        npy_header_bytes=header_bytes,
        payload_bytes=payload_bytes,
        storage_rows=storage_rows,
        feature_dim=feature_dim,
        dtype=dtype.str,
        row_bytes=row_bytes,
        page_size=page_size,
        alignment_rows=alignment_rows,
        payload_pages=payload_bytes // page_size,
        num_ele=storage_rows * feature_dim,
        device=device,
        confirmation=token,
        device_offset_bytes=int(device_offset_bytes),
        artifact_schema_version=int(manifest["schema_version"]),
        artifact_geometry_sha256=artifact_geometry.semantic_sha256(),
        runtime_geometry_sha256=runtime_geometry.semantic_sha256(),
        runtime_geometry=runtime_geometry.to_dict(),
    )


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plain_payload_plan(
    feature_file, *, page_size: int = 4096,
    device: str = SUPPORTED_DEVICE, device_offset_bytes: int = 0,
) -> PayloadPlan:
    """Build a guarded plan for the original contiguous float32 NPY payload."""

    feature_path = Path(feature_file).resolve()
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    dtype = np.dtype(features.dtype)
    if dtype != np.dtype("float32"):
        raise ArtifactValidationError("GIDS float storage requires float32 features")
    if features.ndim != 2 or not features.flags.c_contiguous:
        raise ArtifactValidationError("plain features must be contiguous and 2-D")
    header_bytes = int(getattr(features, "offset", 0))
    payload_bytes = int(features.nbytes)
    storage_rows, feature_dim = (int(value) for value in features.shape)
    row_bytes = feature_dim * dtype.itemsize
    if header_bytes <= 0 or feature_path.stat().st_size != header_bytes + payload_bytes:
        raise ArtifactValidationError("invalid or non-canonical NPY payload")
    if page_size <= 0 or page_size % row_bytes:
        raise ArtifactValidationError("page size must contain complete feature rows")
    if payload_bytes % page_size:
        raise ArtifactValidationError("plain SSD payload must end on a full page")
    checksum = _sha256(feature_path)
    identity = "plain:{}".format(feature_path)
    device = str(Path(device))
    token = "ERASE-{}-{}-{}".format(
        Path(device).name, checksum[:16], payload_bytes
    )
    if device_offset_bytes < 0 or device_offset_bytes % int(page_size):
        raise ArtifactValidationError("device offset must be non-negative and page aligned")
    if device_offset_bytes:
        token += "-OFFSET{}".format(device_offset_bytes)
    minimum_transfer = min(int(page_size), 4096)
    artifact_geometry = IOGeometry.create(
        feature_row_bytes=row_bytes,
        group_size=int(page_size) // row_bytes,
        minimum_transfer_bytes=minimum_transfer,
        target_request_bytes=int(page_size),
    )
    runtime_geometry = artifact_geometry.with_payload_offset(device_offset_bytes)
    token += "-GEO{}".format(runtime_geometry.semantic_sha256()[:16])
    return PayloadPlan(
        artifact=identity,
        feature_file=str(feature_path),
        feature_file_sha256=checksum,
        npy_header_bytes=header_bytes,
        payload_bytes=payload_bytes,
        storage_rows=storage_rows,
        feature_dim=feature_dim,
        dtype=dtype.str,
        row_bytes=row_bytes,
        page_size=int(page_size),
        alignment_rows=int(page_size) // row_bytes,
        payload_pages=payload_bytes // int(page_size),
        num_ele=storage_rows * feature_dim,
        device=device,
        confirmation=token,
        device_offset_bytes=int(device_offset_bytes),
        artifact_schema_version=0,
        artifact_geometry_sha256=artifact_geometry.semantic_sha256(),
        runtime_geometry_sha256=runtime_geometry.semantic_sha256(),
        runtime_geometry=runtime_geometry.to_dict(),
    )


def validate_device(device: str) -> None:
    if os.path.abspath(device) != SUPPORTED_DEVICE:
        raise RuntimeError(
            "DiGiT v1 is pinned to {}; refusing device {}".format(
                SUPPORTED_DEVICE, device
            )
        )
    try:
        mode = os.stat(device).st_mode
    except FileNotFoundError as error:
        raise RuntimeError("BaM device is absent: {}".format(device)) from error
    if not stat.S_ISCHR(mode):
        raise RuntimeError("BaM target is not a character device: {}".format(device))


def require_root() -> None:
    if os.geteuid() != 0:
        raise PermissionError("SSD write/readback requires sudo/root")


def ensure_device_idle(device: str) -> None:
    result = subprocess.run(
        ["fuser", device], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode == 0 and (result.stdout.strip() or result.stderr.strip()):
        raise RuntimeError(
            "BaM device is in use; stop GIDS/BaM processes first: {} {}".format(
                result.stdout.strip(), result.stderr.strip()
            )
        )
    if result.returncode not in (0, 1):
        raise RuntimeError("unable to determine whether the BaM device is idle")


def build_write_command(
    plan: PayloadPlan,
    benchmark,
    *,
    gpu: int = 0,
    cache_pages: int = 10000,
) -> List[str]:
    benchmark = Path(benchmark).resolve()
    if not benchmark.is_file() or not os.access(str(benchmark), os.X_OK):
        raise FileNotFoundError("BaM writer is absent or not executable: {}".format(benchmark))
    if cache_pages <= 0:
        raise ValueError("cache_pages must be positive")
    if plan.payload_pages % cache_pages:
        raise ValueError(
            "cache_pages must divide payload_pages exactly; the BaM writer emits full cache chunks"
        )
    return [
        str(benchmark),
        "--input", plan.feature_file,
        "--queue_depth", "1024",
        "--access_type", "1",
        "--num_queues", "128",
        "--threads", "102400",
        "--n_ctrls", "1",
        "--ioffset", str(plan.npy_header_bytes),
        "--loffset", str(plan.device_offset_bytes),
        "--page_size", str(plan.page_size),
        "--pages", str(cache_pages),
        "--libnvmName", plan.device,
        "--gpu", str(gpu),
    ]


def validate_verify_receipt(receipt, plan: PayloadPlan) -> Mapping:
    receipt_path = Path(receipt).resolve()
    try:
        payload = _digit_paths.json_loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read SSD verification receipt: {}".format(receipt_path)) from error
    expected = {
        "status": "verified",
        "artifact": plan.artifact,
        "feature_file_sha256": plan.feature_file_sha256,
        "payload_bytes": plan.payload_bytes,
        "storage_rows": plan.storage_rows,
        "page_size": plan.page_size,
        "device": plan.device,
    }
    if plan.artifact_schema_version >= 2:
        expected.update({
            "receipt_schema_name": RECEIPT_SCHEMA_NAME,
            "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
            "artifact_schema_version": plan.artifact_schema_version,
            "artifact_geometry_sha256": plan.artifact_geometry_sha256,
            "runtime_geometry_sha256": plan.runtime_geometry_sha256,
        })
    if plan.device_offset_bytes:
        expected["device_offset_bytes"] = plan.device_offset_bytes
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                "SSD verification receipt {} mismatch: expected {!r}, got {!r}".format(
                    key, value, payload.get(key)
                )
            )
    if plan.artifact_schema_version >= 2:
        try:
            runtime_geometry = IOGeometry.from_dict(payload["runtime_geometry"])
        except (KeyError, ValueError) as error:
            raise RuntimeError("SSD verification receipt runtime geometry is invalid") from error
        if runtime_geometry.semantic_sha256() != plan.runtime_geometry_sha256:
            raise RuntimeError("SSD verification receipt runtime geometry mismatch")
    active_state_file = payload.get("active_state_file")
    if active_state_file:
        _validate_active_state(
            Path(active_state_file).resolve(), plan, "verified",
            require_geometry=plan.artifact_schema_version >= 2,
        )
    return payload


def validate_bundle_verify_receipt(
    receipt, bundle, *, device: str = SUPPORTED_DEVICE,
    device_offset_bytes: int = None,
) -> Mapping:
    """Validate a verified-device receipt against an already loaded bundle."""

    feature = bundle.manifest["feature"]
    entry = bundle.manifest["files"]["reordered_features"]
    expected = {
        "status": "verified",
        "artifact": str(bundle.root.resolve()),
        "feature_file_sha256": str(entry["sha256"]),
        "payload_bytes": int(feature["num_storage_rows"])
        * int(feature["row_bytes"]),
        "storage_rows": int(feature["num_storage_rows"]),
        "page_size": int(bundle.manifest["io"]["page_size"]),
        "device": device,
    }
    artifact_geometry = bundle.io_geometry
    geometry_expected = {
        "receipt_schema_name": RECEIPT_SCHEMA_NAME,
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact_schema_version": int(bundle.manifest["schema_version"]),
        "artifact_geometry_sha256": artifact_geometry.semantic_sha256(),
    }
    receipt_path = Path(receipt).resolve()
    try:
        payload = _digit_paths.json_loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read SSD verification receipt: {}".format(receipt_path)) from error
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                "SSD verification receipt {} mismatch: expected {!r}, got {!r}".format(
                    key, value, payload.get(key)
                )
            )
    if int(bundle.manifest["schema_version"]) >= 2:
        for key, value in geometry_expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    "SSD verification receipt {} mismatch: expected {!r}, got {!r}".format(
                        key, value, payload.get(key)
                    )
                )
        try:
            runtime_geometry = IOGeometry.from_dict(payload["runtime_geometry"])
        except (KeyError, ValueError) as error:
            raise RuntimeError("SSD verification receipt runtime geometry is invalid") from error
        if payload.get("runtime_geometry_sha256") != runtime_geometry.semantic_sha256():
            raise RuntimeError("SSD verification receipt runtime_geometry_sha256 mismatch")
        if runtime_geometry.with_payload_offset(0) != artifact_geometry:
            raise RuntimeError("SSD verification receipt geometry does not match artifact")
        if device_offset_bytes is not None:
            expected_runtime = artifact_geometry.with_payload_offset(
                int(device_offset_bytes)
            )
            if runtime_geometry != expected_runtime:
                raise RuntimeError(
                    "SSD verification receipt payload offset does not match runtime"
                )
    active_state_file = payload.get("active_state_file")
    if active_state_file:
        state_path = Path(active_state_file).resolve()
    else:
        candidate = (
            receipt_path.parent.parent
            / ACTIVE_STATE_DIR
            / (Path(device).name + ".json")
        )
        state_path = candidate.resolve() if candidate.exists() else None
    if state_path is not None:
        try:
            state = _digit_paths.json_loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("cannot read active SSD payload state: {}".format(state_path)) from error
        for key in (
            "status", "artifact", "feature_file_sha256", "payload_bytes",
            "storage_rows", "page_size", "device",
        ):
            value = expected[key]
            if state.get(key) != value:
                raise RuntimeError("active SSD payload state {} mismatch".format(key))
        if int(bundle.manifest["schema_version"]) >= 2:
            for key, value in geometry_expected.items():
                if state.get(key) != value:
                    raise RuntimeError("active SSD payload state {} mismatch".format(key))
            if state.get("runtime_geometry_sha256") != payload.get(
                "runtime_geometry_sha256"
            ):
                raise RuntimeError(
                    "active SSD payload state runtime geometry mismatch"
                )
    return payload


def _common_parser(subparser) -> None:
    source = subparser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact")
    source.add_argument("--plain-feature-file")
    subparser.add_argument("--plain-page-size", type=int, default=4096)
    subparser.add_argument("--device-offset-bytes", type=int, default=0)
    subparser.add_argument("--device", default=SUPPORTED_DEVICE)
    subparser.add_argument(
        "--skip-checksums", action="store_true",
        help="Skip artifact hashes (not permitted for write or verify)",
    )


def _parser(repo_root: Path):
    parser = argparse.ArgumentParser(
        description="Safely write and exactly verify a DiGiT feature payload"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    plan = subparsers.add_parser("plan", help="validate layout and print the write plan")
    _common_parser(plan)
    plan.add_argument(
        "--state-dir", default=str(repo_root / ACTIVE_STATE_DIR),
        help="active SSD extent registry used for overlap auditing",
    )
    plan.add_argument(
        "--cache-pages", type=int, default=10000,
        help="writer cache pages shown in the guarded command",
    )
    plan.add_argument(
        "--output", help="optional path for an atomic JSON copy of the plan",
    )

    write = subparsers.add_parser("write", help="destructively write the payload")
    _common_parser(write)
    write.add_argument("--confirm", required=True)
    write.add_argument("--gpu", type=int, default=0)
    write.add_argument("--cache-pages", type=int, default=10000)
    write.add_argument(
        "--benchmark",
        default=str(repo_root / "third_party/bam/build/bin/nvm-readwrite_stripe-bench-jc"),
    )
    write.add_argument(
        "--report-dir", default=str(repo_root / "phase5_results")
    )

    verify = subparsers.add_parser("verify", help="read every row through GIDS")
    _common_parser(verify)
    verify.add_argument("--gpu", type=int, default=0)
    verify.add_argument("--cache-size", type=int, default=64)
    verify.add_argument("--chunk-rows", type=int, default=8192)
    verify.add_argument(
        "--report-dir", default=str(repo_root / "phase5_results")
    )
    return parser


def _print_plan(
    plan: PayloadPlan, benchmark: Path, state_dir: Path, cache_pages: int,
    output: str = None,
) -> None:
    payload = asdict(plan)
    payload["device_present"] = Path(plan.device).exists()
    payload["target_extent"] = {
        "start_bytes": plan.device_offset_bytes,
        "end_bytes_exclusive": plan.device_offset_bytes + plan.payload_bytes,
    }
    payload["active_state_dir"] = str(Path(state_dir).resolve())
    payload["active_range_overlaps"] = find_active_range_overlaps(plan, state_dir)
    payload["range_safe_against_active_state"] = not payload["active_range_overlaps"]
    payload["write_command"] = build_write_command(
        plan, benchmark, cache_pages=cache_pages
    )
    if output:
        _atomic_json(Path(output).resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if output:
        print("Guarded SSD write plan: {}".format(Path(output).resolve()))


def _write(args, plan: PayloadPlan) -> None:
    if args.skip_checksums:
        raise RuntimeError("--skip-checksums is forbidden for an SSD write")
    require_root()
    validate_device(plan.device)
    ensure_device_idle(plan.device)
    if args.confirm != plan.confirmation:
        raise RuntimeError(
            "confirmation token mismatch; run the plan command and copy its exact token"
        )
    command = build_write_command(
        plan, args.benchmark, gpu=args.gpu, cache_pages=args.cache_pages
    )
    report_dir = Path(args.report_dir).resolve()
    require_unoccupied_active_range(plan, report_dir.parent / ACTIVE_STATE_DIR)
    active_state_path = _plan_active_state_path(report_dir, plan)
    _atomic_json(active_state_path, _active_binding(plan, "write_in_progress"))
    started = _utc_now()
    start_time = time.perf_counter()
    subprocess.run(command, check=True)
    receipt = {
        **asdict(plan),
        "status": "written_unverified",
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "command": command,
        "active_state_file": str(active_state_path),
    }
    _atomic_json(active_state_path, _active_binding(plan, "written_unverified"))
    path = report_dir / WRITE_RECEIPT
    _atomic_json(path, receipt)
    print("Write completed; unverified receipt: {}".format(path))


def _verify(args, plan: PayloadPlan) -> None:
    if args.skip_checksums:
        raise RuntimeError("--skip-checksums is forbidden for SSD verification")
    if args.chunk_rows <= 0 or args.cache_size <= 0:
        raise ValueError("chunk_rows and cache_size must be positive")
    require_root()
    validate_device(plan.device)
    report_dir = Path(args.report_dir).resolve()
    write_receipt_path = report_dir / WRITE_RECEIPT
    try:
        write_receipt = _digit_paths.json_loads(write_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("a completed write receipt is required before verify") from error
    for key in (
        "artifact", "feature_file_sha256", "payload_bytes", "storage_rows",
        "page_size", "device", "artifact_schema_version",
        "artifact_geometry_sha256", "runtime_geometry_sha256",
        "receipt_schema_name", "receipt_schema_version",
    ):
        if write_receipt.get(key) != getattr(plan, key):
            raise RuntimeError("write receipt does not match current plan: {}".format(key))
    if write_receipt.get("runtime_geometry") != plan.runtime_geometry:
        raise RuntimeError("write receipt does not match current runtime geometry")
    if write_receipt.get("status") != "written_unverified":
        raise RuntimeError("write receipt has an invalid status")
    active_state_path = Path(
        write_receipt.get(
            "active_state_file", _plan_active_state_path(report_dir, plan)
        )
    ).resolve()
    _validate_active_state(active_state_path, plan, "written_unverified")

    import torch
    import GIDS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GIDS SSD verification")
    torch.cuda.set_device(args.gpu)
    mixed_io = plan.artifact_schema_version >= 2 and not plan.artifact.startswith("plain:")
    mixed_io_geometry = None
    group_flags_by_row = None
    if mixed_io:
        bundle = load_artifact_bundle(plan.artifact, mmap_mode="r", verify_checksums=True)
        runtime_geometry = bundle.io_geometry.with_payload_offset(
            plan.device_offset_bytes
        )
        mixed_io_geometry = runtime_geometry.to_native_mapping()
        group_flags_by_row = np.zeros(plan.storage_rows, dtype=np.bool_)
        group_bases = np.asarray(
            bundle.arrays["group_storage_base"], dtype=np.int64
        )
        if group_bases.size:
            group_rows = (
                group_bases[:, None]
                + np.arange(bundle.group_size, dtype=np.int64)[None, :]
            ).reshape(-1)
            group_flags_by_row[group_rows] = True

    gids = GIDS.GIDS(
        page_size=plan.page_size,
        off=plan.device_offset_bytes,
        num_ele=plan.num_ele,
        num_ssd=1,
        cache_size=args.cache_size,
        cache_dim=plan.feature_dim,
        ssd_list=[0],
        ctrl_idx=args.gpu,
        feature_index_mode="explicit",
        mixed_io=mixed_io,
        mixed_io_geometry=mixed_io_geometry,
    )
    source = np.load(plan.feature_file, mmap_mode="r", allow_pickle=False)
    chunks = 0
    started = _utc_now()
    start_time = time.perf_counter()
    for start in range(0, plan.storage_rows, args.chunk_rows):
        end = min(start + args.chunk_rows, plan.storage_rows)
        count = end - start
        rows = torch.arange(start, end, dtype=torch.int64, device="cuda")
        actual = torch.empty(
            (count, plan.feature_dim), dtype=torch.float32, device="cuda"
        ).contiguous()
        group_flags = None
        if mixed_io:
            group_flags = torch.from_numpy(
                group_flags_by_row[start:end].copy()
            ).to(device="cuda")
        gids.BAM_FS.read_feature(
            actual.data_ptr(), rows.data_ptr(), count,
            plan.feature_dim, plan.feature_dim, 0,
            group_flags.data_ptr() if group_flags is not None else 0,
        )
        expected_cpu = torch.from_numpy(np.array(source[start:end], copy=True))
        expected = expected_cpu.to(device="cuda", non_blocking=False)
        torch.cuda.synchronize()
        different = actual.view(torch.uint8) != expected.view(torch.uint8)
        if bool(torch.any(different).item()):
            first_byte = int(torch.nonzero(different.reshape(-1), as_tuple=True)[0][0])
            row = start + first_byte // plan.row_bytes
            raise RuntimeError("SSD payload mismatch at storage row {}".format(row))
        chunks += 1
        print(
            "Verified rows {}..{} ({:.2f}%)".format(
                start, end - 1, 100.0 * end / plan.storage_rows
            ),
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    receipt = {
        **asdict(plan),
        "status": "verified",
        "verification": "full GIDS/BaM bit-exact row comparison",
        "verified_at_utc": _utc_now(),
        "started_at_utc": started,
        "elapsed_seconds": elapsed,
        "chunks": chunks,
        "verified_rows": plan.storage_rows,
        "verified_bytes": plan.payload_bytes,
        "useful_bytes": plan.payload_bytes,
        "transferred_page_bytes": plan.payload_pages * plan.page_size,
        "read_amplification": (plan.payload_pages * plan.page_size) / plan.payload_bytes,
        "verification_io_mode": "mixed" if mixed_io else "standard",
        "active_state_file": str(active_state_path),
    }
    _atomic_json(active_state_path, _active_binding(plan, "verified"))
    path = report_dir / VERIFY_RECEIPT
    _atomic_json(path, receipt)
    print("Full SSD verification passed: {}".format(path))


def main(argv=None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = _parser(repo_root)
    args = parser.parse_args(argv)
    try:
        if args.plain_feature_file:
            if args.skip_checksums:
                raise RuntimeError("--skip-checksums is forbidden for a plain payload")
            plan = build_plain_payload_plan(
                args.plain_feature_file,
                page_size=args.plain_page_size,
                device=args.device,
                device_offset_bytes=args.device_offset_bytes,
            )
        else:
            plan = build_payload_plan(
                args.artifact,
                device=args.device,
                verify_checksums=not args.skip_checksums,
                device_offset_bytes=args.device_offset_bytes,
            )
        if args.action == "plan":
            _print_plan(
                plan,
                repo_root / "third_party/bam/build/bin/nvm-readwrite_stripe-bench-jc",
                Path(args.state_dir),
                args.cache_pages,
                args.output,
            )
        elif args.action == "write":
            _write(args, plan)
        elif args.action == "verify":
            _verify(args, plan)
        else:
            parser.error("unknown action")
        return 0
    except (ArtifactValidationError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
