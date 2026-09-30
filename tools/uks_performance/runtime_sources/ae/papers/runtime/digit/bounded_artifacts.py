"""Chunked invariant validation with disposable bitmap state and mmap page release."""
import contextlib
import math
import mmap
import tempfile
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from .artifacts import (_validate_manifest_structure, _require_keys, _as_int,
                        ARRAY_FILENAMES, INTEGER_ARRAYS, ArtifactValidationError)
from .io_geometry import IOGeometry, GeometryValidationError
from .validation_support import Bits
from . import validation_support as lp


class Reader:
    def __init__(self, array, chunk):
        if not isinstance(array, np.memmap) or array.mode != "r":
            raise ArtifactValidationError("bounded validation requires read-only mmap arrays")
        if not hasattr(array._mmap, "madvise"):
            raise ArtifactValidationError("bounded mmap validation requires madvise support")
        self.a, self.chunk, self.cached = array, chunk, None
        self.base = -1

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size and (indices.min()<0 or indices.max()>=self.a.size):
            raise ArtifactValidationError("array index outside range")
        result = np.array(self.a[np.unravel_index(indices, self.a.shape)],copy=True)
        self.a._mmap.madvise(mmap.MADV_DONTNEED)
        return result

    def get(self, start, count):
        if count > self.chunk:
            raise ArtifactValidationError("group/alignment exceeds validation chunk")
        base = (start//self.chunk)*self.chunk
        if start < 0 or start+count > self.a.size:
            raise ArtifactValidationError("array range outside bounds")
        if start+count > base+self.chunk:
            return self.take(np.arange(start,start+count,dtype=np.int64))
        if base != self.base:
            self.cached=self.take(np.arange(base,min(base+self.chunk,self.a.size),dtype=np.int64))
            self.base=base
        return self.cached[start-base:start-base+count]

    def one(self,index):
        return int(self.get(index,1)[0])


def require(condition,message):
    if not condition:raise ArtifactValidationError(message)


def validate_artifact_bundle(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray], chunk_rows=4096, scratch_dir=None, *, fast=False
) -> None:
    """Validate manifest metadata and cross-array ID invariants."""

    _validate_manifest_structure(manifest)
    _require_keys(arrays, list(ARRAY_FILENAMES), "arrays")

    for name in ARRAY_FILENAMES:
        array = arrays[name]
        if not isinstance(array, np.ndarray):
            raise ArtifactValidationError("arrays.{} must be a NumPy array".format(name))
        if array.dtype.hasobject:
            raise ArtifactValidationError("arrays.{} cannot use object dtype".format(name))
        if name in INTEGER_ARRAYS and not np.issubdtype(array.dtype, np.integer):
            raise ArtifactValidationError("arrays.{} must use an integer dtype".format(name))
        entry = manifest["files"][name]
        if list(array.shape) != list(entry["shape"]):
            raise ArtifactValidationError("arrays.{} shape does not match manifest".format(name))
        if np.dtype(array.dtype).str != entry["dtype"]:
            raise ArtifactValidationError("arrays.{} dtype does not match manifest".format(name))

    dataset = manifest["dataset"]
    feature = manifest["feature"]
    grouping = manifest["grouping"]
    graph = manifest["graph"]
    io = manifest["io"]

    if manifest["schema_version"] == 2:
        try:
            geometry = IOGeometry.from_dict(manifest["io_geometry"])
        except GeometryValidationError as error:
            raise ArtifactValidationError(
                "invalid io_geometry: {}".format(error)
            ) from error
        if manifest["io_geometry_sha256"] != geometry.semantic_sha256():
            raise ArtifactValidationError("io_geometry_sha256 is inconsistent")
    else:
        geometry = None

    num_nodes = _as_int(dataset["num_nodes"], "dataset.num_nodes")
    num_edges = _as_int(dataset["num_edges"], "dataset.num_edges")
    feature_dim = _as_int(feature["dim"], "feature.dim")
    num_storage_rows = _as_int(feature["num_storage_rows"], "feature.num_storage_rows")
    group_size = _as_int(grouping["group_size"], "grouping.group_size")
    num_groups = _as_int(grouping["num_groups"], "grouping.num_groups")
    num_primary = _as_int(grouping["num_primary_groups"], "grouping.num_primary_groups")
    num_replica = _as_int(grouping["num_replica_groups"], "grouping.num_replica_groups")
    supernode_start = _as_int(grouping["supernode_id_start"], "grouping.supernode_id_start")
    page_size = _as_int(io["page_size"], "io.page_size")
    alignment_rows = _as_int(io["alignment_rows"], "io.alignment_rows")

    if min(num_nodes, feature_dim, num_storage_rows, group_size, page_size, alignment_rows) <= 0:
        raise ArtifactValidationError("node, feature, storage, group, and I/O sizes must be positive")
    if num_edges < 0 or num_groups < 0:
        raise ArtifactValidationError("edge and group counts cannot be negative")
    if not 0 <= num_primary <= num_groups:
        raise ArtifactValidationError("num_primary_groups is out of range")
    if num_replica != num_groups - num_primary:
        raise ArtifactValidationError("num_replica_groups is inconsistent")
    if supernode_start != num_nodes:
        raise ArtifactValidationError("v1 supernode IDs must begin at num_nodes")
    if graph["format"] != "csc":
        raise ArtifactValidationError("v1 graph format must be csc")
    if int(graph["num_logical_nodes"]) != num_nodes:
        raise ArtifactValidationError("graph.num_logical_nodes is inconsistent")
    if int(graph["num_supernodes"]) != num_groups:
        raise ArtifactValidationError("graph.num_supernodes is inconsistent")
    if int(graph["num_graph_nodes"]) != num_nodes + num_groups:
        raise ArtifactValidationError("graph.num_graph_nodes is inconsistent")

    group_members = arrays["group_members"]
    group_owner = arrays["group_owner"]
    group_storage_base = arrays["group_storage_base"]
    storage_to_node = arrays["storage_to_node"]
    node_to_primary = arrays["node_to_primary_row"]
    supernode_to_group = arrays["supernode_to_group"]
    indptr = arrays["reordered_indptr"]
    indices = arrays["reordered_indices"]
    features = arrays["reordered_features"]

    if group_members.shape != (num_groups, group_size):
        raise ArtifactValidationError("group_members must have shape [num_groups, group_size]")
    if group_owner.shape != (num_groups,):
        raise ArtifactValidationError("group_owner must have shape [num_groups]")
    if group_storage_base.shape != (num_groups,):
        raise ArtifactValidationError("group_storage_base must have shape [num_groups]")
    if storage_to_node.shape != (num_storage_rows,):
        raise ArtifactValidationError("storage_to_node must have shape [num_storage_rows]")
    if node_to_primary.shape != (num_nodes,):
        raise ArtifactValidationError("node_to_primary_row must have shape [num_nodes]")
    if supernode_to_group.shape != (num_groups,):
        raise ArtifactValidationError("supernode_to_group must have shape [num_groups]")
    if features.shape != (num_storage_rows, feature_dim):
        raise ArtifactValidationError("reordered_features shape is inconsistent")

    row_bytes = features.dtype.itemsize * feature_dim
    if np.dtype(feature["dtype"]) != features.dtype:
        raise ArtifactValidationError("feature.dtype is inconsistent")
    if int(feature["row_bytes"]) != row_bytes:
        raise ArtifactValidationError("feature.row_bytes is inconsistent")
    if page_size % row_bytes != 0:
        raise ArtifactValidationError("page_size must be a multiple of one feature row")
    if alignment_rows != page_size // row_bytes:
        raise ArtifactValidationError("io.alignment_rows is inconsistent")
    if geometry is not None:
        if geometry.feature_row_bytes != row_bytes:
            raise ArtifactValidationError("io_geometry feature row is inconsistent")
        if geometry.group_size != group_size:
            raise ArtifactValidationError("io_geometry group size is inconsistent")
        if geometry.cache_slot_bytes != page_size:
            raise ArtifactValidationError("io_geometry cache slot is inconsistent")
        if geometry.rows_per_slot != alignment_rows:
            raise ArtifactValidationError("io_geometry alignment rows are inconsistent")


    require(0 < chunk_rows <= 4096, "validation chunk must be in [1,4096]")
    require(max(group_size,alignment_rows) <= chunk_rows, "group/alignment exceeds chunk bound")
    require(hasattr(mmap, "MADV_DONTNEED"), "bounded validation needs Linux mmap advice")
    lp.check_heap(48*lp.MIB,64)
    ratio=float(grouping["replication_ratio"])
    require(math.isfinite(ratio) and ratio>=0 and num_replica*group_size<=math.floor(ratio*num_nodes),
            "invalid replication budget")
    require(indptr.shape==(num_nodes+num_groups+1,), "wrong CSC pointer shape")
    if fast:
        from .fast_artifacts import validate_arrays
        return validate_arrays(manifest, arrays)
    scratch_parent=Path(scratch_dir or tempfile.gettempdir())
    lp.check_space(scratch_parent,(num_nodes+num_storage_rows+num_groups+7)//8+lp.MIB)
    readers={name:Reader(arrays[name],chunk_rows) for name in ARRAY_FILENAMES}
    def get(name,start,count):return readers[name].get(start,count)
    def one(name,index):return readers[name].one(index)
    with tempfile.TemporaryDirectory(prefix="digit-validate-",dir=str(scratch_parent)) as tmp, contextlib.ExitStack() as stack:
        def bits(name,count):
            result=Bits(Path(tmp)/name,count);stack.callback(result.close);return result
        occupied=bits("occupied",num_storage_rows)
        primary_seen=bits("primary",num_nodes)
        groups_seen=bits("groups",num_groups)
        for gid in range(num_groups):
            base=one("group_storage_base",gid)
            require(0<=one("group_owner",gid)<num_nodes,"group owner outside range")
            width=alignment_rows if geometry else group_size
            require(base>=0 and base%alignment_rows==0 and base+max(width,group_size)<=num_storage_rows,
                    "group storage range/alignment invalid")
            for row in range(base,base+width):
                require(not occupied.set(row),"overlapping group storage")
            members=get("group_members",gid*group_size,group_size)
            require(np.all((members>=0)&(members<num_nodes)),"invalid group member")
            require(np.array_equal(get("storage_to_node",base,group_size),members),"group storage mapping mismatch")
            if geometry and alignment_rows>group_size:
                require(np.all(get("storage_to_node",base+group_size,alignment_rows-group_size)==-1),"group padding used")
            if gid<num_primary:
                for node in members:require(not primary_seen.set(int(node)),"primary groups overlap")
                require(np.array_equal(readers["node_to_primary_row"].take(members),
                                       np.arange(base,base+group_size,dtype=np.int64)),"primary row mismatch")
            mapped=one("supernode_to_group",gid)
            require(0<=mapped<num_groups and not groups_seen.set(mapped),"invalid supernode permutation")
        for start in range(0,num_storage_rows,chunk_rows):
            values=get("storage_to_node",start,min(chunk_rows,num_storage_rows-start))
            require(np.all((values>=-1)&(values<num_nodes)),"invalid storage node")
        for start in range(0,num_nodes,chunk_rows):
            count=min(chunk_rows,num_nodes-start)
            rows=get("node_to_primary_row",start,count)
            require(np.all((rows>=0)&(rows<num_storage_rows)),"invalid primary row")
            require(np.array_equal(readers["storage_to_node"].take(rows),np.arange(start,start+count)),
                    "primary inverse mismatch")
        previous=0
        for start in range(0,num_nodes+num_groups+1,chunk_rows):
            p=get("reordered_indptr",start,min(chunk_rows,num_nodes+num_groups+1-start))
            require((start!=0 or p[0]==0) and p[0]>=previous and np.all(p[1:]>=p[:-1]),
                    "CSC pointer is not monotonic")
            previous=int(p[-1])
        require(previous==indices.size,"CSC terminal pointer mismatch")
        for start in range(0,indices.size,chunk_rows):
            values=get("reordered_indices",start,min(chunk_rows,indices.size-start))
            require(np.all((values>=0)&(values<num_nodes+num_groups)),"invalid CSC node")
    for array in arrays.values():array._mmap.madvise(mmap.MADV_DONTNEED)

