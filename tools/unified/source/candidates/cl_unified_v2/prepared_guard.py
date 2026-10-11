from .common import verify,require
from .source_guard import identity


class PreparedGuard:
    def __init__(self):
        from candidates.group_cache_migration_v1.prepare import check_prepared,DATA
        self.path=DATA/'CL/prepared.json'
        self.before=identity(self.path)
        self.record=check_prepared('CL')
        if identity(self.path)!=self.before:raise RuntimeError('Prepared receipt changed during admission')

    def check(self):
        from candidates.group_cache_migration_v1.common import read,identity as file_identity,stable
        from candidates.group_cache_migration_v1.prepare import source
        verify()
        require(identity(self.path)==self.before,'Prepared receipt identity changed')
        r=read(self.path)
        require(r==self.record and r['passed'],'Prepared receipt changed')
        require(r['base']==source('CL'),'Original graph binding changed')
        s=stable('CL')
        for v in r['files'].values():
            now=file_identity(v['path'])
            require(now[1:]==v['identity'][1:] and s.filesystem_uuid(v['path'])==v['filesystem_uuid'],'Overlay identity changed')
        return r
