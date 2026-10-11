"""Narrow, evidence-backed classification of ext4's three-line error summary.

Only complete notice-level triples matching unchanged sysfs error metadata
captured before monitoring are historical. Unknown/partial reports stay alerts.
No kernel record is deleted or changed. No block device is opened.
"""
import re
import time
from pathlib import Path

FIELDS = ('errors_count', 'first_error_time', 'last_error_time',
          'first_error_func', 'last_error_func', 'first_error_line', 'last_error_line')
PREFIX = r'EXT4-fs \(([A-Za-z0-9_.-]+)\): '
COUNT = re.compile(PREFIX + r'error count since last fsck: ([1-9][0-9]*)')
DETAIL = re.compile(PREFIX + r'(initial|last) error at time ([1-9][0-9]*): '
                    r'([A-Za-z0-9_]+):([1-9][0-9]*)')


class Ext4History:
    def __init__(self, root='/sys/fs/ext4', clock=time.time):
        self.root, self.clock = Path(root), clock
        self.started = None
        self.baseline, self.unavailable = {}, {}

    def read(self, device):
        path = self.root / device
        stat = path.stat()
        value = {'identity': [stat.st_dev, stat.st_ino]}
        for field in FIELDS:
            with (path / field).open() as stream:
                text = stream.read(256).strip()
            if field.endswith('_func'):
                if not re.fullmatch(r'[A-Za-z0-9_]+', text):
                    raise ValueError('invalid function')
                value[field] = text
            else:
                if not re.fullmatch(r'[0-9]{1,20}', text):
                    raise ValueError('invalid error counter/time')
                value[field] = int(text)
        return value

    def capture(self):
        self.started = int(self.clock())
        try:
            devices = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        except OSError as error:
            self.unavailable['root'] = str(error)
            return
        for name in devices:
            try:
                first, second = self.read(name), self.read(name)
                if first != second:
                    raise ValueError('metadata changed during baseline')
                self.baseline[name] = first
            except (OSError, ValueError) as error:
                self.unavailable[name] = str(error)

    def classify(self, records):
        historical, decisions = set(), []
        for i, row in enumerate(records):
            match = COUNT.fullmatch(row['MESSAGE'])
            if match is None:
                continue
            device, count = match.group(1), int(match.group(2))
            decision = dict(device=device, cursor=row['__CURSOR'], historical=False)
            decisions.append(decision)
            try:
                if i + 2 >= len(records):
                    raise ValueError('incomplete summary')
                triple = records[i:i+3]
                if any(int(r['PRIORITY']) != 5 or r.get('_TRANSPORT') != 'kernel'
                       or r.get('SYSLOG_IDENTIFIER') != 'kernel' for r in triple):
                    raise ValueError('not a notice-level kernel summary')
                details = [DETAIL.fullmatch(r['MESSAGE']) for r in triple[1:]]
                if any(m is None for m in details):
                    raise ValueError('unrecognized summary detail')
                first, last = details
                if (first.group(1), first.group(2), last.group(1), last.group(2)) != (
                        device, 'initial', device, 'last'):
                    raise ValueError('mixed or reordered summary')
                old = self.baseline.get(device)
                if old is None:
                    raise ValueError('no readable baseline for device')
                expected = dict(old)
                expected.update(errors_count=count,
                                first_error_time=int(first.group(3)),
                                first_error_func=first.group(4), first_error_line=int(first.group(5)),
                                last_error_time=int(last.group(3)),
                                last_error_func=last.group(4), last_error_line=int(last.group(5)))
                if expected != old:
                    raise ValueError('summary differs from pre-run error metadata')
                if not 0 < old['first_error_time'] <= old['last_error_time'] < self.started:
                    raise ValueError('error is not strictly older than monitoring baseline')
                # Compare twice so a concurrent metadata update cannot be used as
                # evidence for suppressing a new report. Any ambiguity stops.
                current, again = self.read(device), self.read(device)
                if current != old or again != old:
                    raise ValueError('filesystem identity or error metadata changed')
                decision.update(historical=True, reason='unchanged_pre_run_ext4_summary',
                                baseline=old, current=current,
                                cursors=[r['__CURSOR'] for r in triple])
                historical.update(decision['cursors'])
            except (OSError, ValueError) as error:
                decision['reason'] = str(error)
        return historical, decisions
