"""New-workspace checkpoints with hash-bound, crash-recoverable retirement."""

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
import fcntl
import json
from pathlib import Path

if __package__:
    from . import large_preprocess as lp
else:
    import large_preprocess as lp


class Job(lp.Job):
    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.root / 'state.json'
            if path.exists():
                self.state = _digit_paths.json_loads(path.read_text())
                if self.state['binding'] != self.binding or self.state['schema'] != 'digit-retiring-job-v2':
                    raise ValueError('checkpoint binding changed; use a NEW workspace')
                self.validate()
                self.finish_retirement()
            else:
                if any(p.name != '.lock' for p in self.root.iterdir()):
                    raise ValueError('new workspace must be empty')
                self.state = dict(schema='digit-retiring-job-v2', binding=self.binding,
                                  files={}, retired={}, pending=[], phase='running')
                self.save()
            return self
        except BaseException:
            self.lock.close()
            raise

    def validate(self):
        files, retired = self.state['files'], self.state['retired']
        pending = set(self.state['pending'])
        if not pending.issubset(retired):
            raise ValueError('invalid retirement intent')
        done = set()
        def visit(name, chain):
            if name not in files or Path(name).name != name or name in chain:
                raise ValueError('invalid retirement dependency')
            if name in done:
                return
            if name in retired:
                if not retired[name]:
                    raise ValueError('retirement has no successor')
                for successor in retired[name]:
                    visit(successor, chain | {name})
            else:
                self.check_file(name)
            done.add(name)
        for name in files:
            visit(name, set())
            path = self.root / name
            if name in retired and (path.exists() or path.is_symlink()):
                if name not in pending:
                    raise ValueError('retired file unexpectedly reappeared')
                self.check_file(name)

    def check_file(self, name):
        path, info = self.root / name, self.state['files'][name]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != info['bytes'] or lp.digest(path) != info['sha256']:
            raise ValueError('committed checkpoint corrupt: ' + name)

    def finish_retirement(self):
        for name in self.state['pending']:
            path = self.root / name
            if path.exists() or path.is_symlink():
                self.check_file(name)
                path.unlink()
        lp.fsync_dir(self.root)
        self.state['pending'] = []
        self.save()

    def retire(self, paths, successors):
        names = [Path(p).name for p in paths if Path(p).name not in self.state['retired']]
        if not names:
            return
        successors = [Path(p).name for p in successors]
        if not successors or any(s in names or s in self.state['retired'] for s in successors):
            raise ValueError('retirement needs live successors')
        for successor in successors:
            self.check_file(successor)
        if any(Path(p).parent.resolve() != self.root for p in paths):
            raise ValueError('retirement must target this workspace')
        for name in names:
            self.check_file(name)
            self.state['retired'][name] = successors
        self.state['pending'] = names
        self.save()  # Intent durable BEFORE unlink; successors already durable.
        self.finish_retirement()
