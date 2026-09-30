"""Checks adapted preparation while preserving the original accepted provenance."""
from .common import ARMS,read,sha,require,identity


def check(value,p,protocol_path,binding_path,execution):
    require(value['schema']=='digit-cache-prepared-v6' and value['passed'] and not value['fixture'] and
            value['source_sha256']==execution and value['protocol_sha256']==sha(protocol_path) and
            value['binding_sha256']==sha(binding_path),'Prepared context changed')
    require(set(value['arms'])==set(ARMS) and value['arms']['freq']==value['arms']['digit'],'Wrong shared hot sets')
    for desc in (value['ranking'],value['profile'],value['inherited_from']):
        require(sha(desc['path'])==desc['sha256'],'Original preparation receipt changed')
    require(identity(value['counts']['path'])==value['counts']['identity'],'Frequency vector changed')
    for name,item in value['arms'].items():
        require(item['rows']==p['arms'][name]['cpu_rows'] and identity(item['path'])==item['identity'],
                'Selected hot file changed: '+name)
    return value
