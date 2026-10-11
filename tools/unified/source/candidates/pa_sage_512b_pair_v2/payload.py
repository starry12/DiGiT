"""Distinguish valid runtime feature rows from already-verified file padding."""
from .common import read,sha,require


def validate_extent(payload,state,runtime_rows,arm,row_bytes=512,alignment=4096):
    require(arm in ('gids','digit'),'Unknown system')
    require(type(runtime_rows) is int and runtime_rows>0,'Invalid feature extent')
    require(row_bytes==512 and alignment==4096,'Unexpected feature/write geometry')
    useful=runtime_rows*row_bytes
    expected=((useful+alignment-1)//alignment)*alignment if arm=='gids' else useful
    require(payload['full_readback_passed'],'Payload lacks full readback')
    require(state['status']=='verified' and state['device']=='/dev/libnvm0','Wrong SSD state')
    require(state['device_offset_bytes']==payload['offset'] and payload['offset']%alignment==0,'SSD offset mismatch')
    require(state['feature_file_sha256']==payload['file_sha256'],'SSD data digest mismatch')
    require(state['page_size']==alignment,'Unexpected recorded write geometry')
    require(payload['bytes']==state['payload_bytes']==expected,'Wrong SSD payload extent')
    require(state['storage_rows']*row_bytes==expected,'SSD row count mismatch')
    padding=expected-useful
    require(0<=padding<alignment and padding%row_bytes==0,'Unexpected trailing padding')
    return dict(passed=True,arm=arm,runtime_rows=runtime_rows,runtime_bytes=useful,
        verified_payload_bytes=expected,verified_payload_rows=state['storage_rows'],
        trailing_padding_bytes=padding,trailing_padding_rows=padding//row_bytes,
        cache_page_bytes=512,payload_alignment_bytes=4096,offset=payload['offset'])


def validate_payload(binding,protocol,arm):
    payload=binding['ssd_payload'] if arm=='gids' else binding['digit_payload']
    require(sha(payload['state'])==payload['state_sha256'],'SSD state checksum changed')
    state=read(payload['state'])
    rows=protocol['nodes'] if arm=='gids' else read(binding['bundle']+'/manifest.json')['feature']['num_storage_rows']
    return validate_extent(payload,state,rows,arm)
