"""UKS row/page addresses; never infer a 512-byte PA row from the pool name."""
from .common import require
DIM=256
ROW_BYTES=1024
PAGE_BYTES=4096
ROWS_PER_PAGE=4


def extent(storage_rows,offset,verified_bytes):
    require(type(storage_rows) is int and storage_rows>0 and storage_rows%ROWS_PER_PAGE==0,'Payload must have complete 4-row pages')
    require(type(offset) is int and offset>=0 and offset%PAGE_BYTES==0,'Unaligned payload base')
    require(type(verified_bytes) is int and verified_bytes>=storage_rows*ROW_BYTES and verified_bytes%PAGE_BYTES==0,'Payload outside verified extent')
    require(offset+verified_bytes<2**63,'Payload address overflows supported signed range')
    return storage_rows*ROW_BYTES


def address(row,storage_rows,offset=0):
    require(type(row) is int and 0<=row<storage_rows,'Invalid storage row')
    extent(storage_rows,offset,storage_rows*ROW_BYTES)
    page,lane=divmod(row,ROWS_PER_PAGE)
    return dict(page=page,byte_in_page=lane*ROW_BYTES,row_offset_bytes=offset+row*ROW_BYTES,
        nvme_read_offset_bytes=offset+page*PAGE_BYTES,minimum_read_bytes=PAGE_BYTES,useful_feature_bytes=ROW_BYTES)


def grouped_rows(base,storage_rows):
    require(type(base) is int and base%ROWS_PER_PAGE==0,'Group must start at a complete cache slot')
    return [address(base+i,storage_rows) for i in range(2)]
