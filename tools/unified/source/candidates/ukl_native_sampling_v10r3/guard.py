"""Frozen v9 protection with evidence-backed ext4 summary classification."""
import json
import subprocess
from candidates.ukl_real_multi_v9.guard import *
from candidates.ukl_real_multi_v9.guard import _io_stat, _read
from candidates.ukl_real_multi_v9 import guard as V9
from .ext4_history import Ext4History


class KernelCursor(V9.KernelCursor):
    def __init__(self, *args, ext4root='/sys/fs/ext4', **kwargs):
        super().__init__(*args, **kwargs)
        self.history = Ext4History(ext4root)
        self.classifications = []

    def establish(self):
        if self.cursor is None:
            self.history.capture()
        return super().establish()

    def _classify(self, records):
        historical, self.classifications = self.history.classify(records)
        return [r for r in records if int(r['PRIORITY']) <= 4 or
                (KERNEL_BAD.search(r['MESSAGE']) and r['__CURSOR'] not in historical)]

    def _result(self, query_ok, records, alerts, current_error=None):
        result = super()._result(query_ok, records, alerts, current_error)
        result.update(ext4_summary_classification=self.classifications if records else [],
                      ext4_history_baseline=dict(started=self.history.started,
                          devices=self.history.baseline, unavailable=self.history.unavailable))
        return result

    def _query(self, initial):
        source = 'boot_identity'
        try:
            if self.boot_invalid:
                raise RuntimeError('Kernel monitor boot identity invalidated; old cursor cannot be reused')
            boot = _read(self.procroot / 'sys/kernel/random/boot_id').strip()
            if not boot or (self.boot_id is not None and boot != self.boot_id):
                self.boot_invalid = True
                raise RuntimeError('Kernel monitor boot identity changed/missing')
            # Bind this instance to one boot even if its first journal query fails.
            self.boot_id = boot
            source = 'cursor_state'
            argv = ['/usr/bin/journalctl', '-k', '-b', '-o', 'json', '--show-cursor', '--no-pager', '--quiet']
            if initial:
                argv += ['-n', '1']
            elif self.cursor is None:
                raise RuntimeError('Kernel monitor has no baseline cursor')
            else:
                argv += ['--after-cursor=' + self.cursor]
            source = 'journal_query'
            text = self.runner(argv, self.timeout, self.max_bytes)
            if len(text.encode('utf-8')) > self.max_bytes:
                raise RuntimeError('Kernel journal output overflow')
            source = 'journal_parse'
            records, cursor, batch_cursors = [], None, set()
            for line in text.splitlines():
                if not line.strip() or line == '-- No entries --':
                    continue
                if line.startswith('-- cursor: '):
                    if cursor is not None:
                        raise ValueError('Duplicate journal cursor')
                    cursor = line[len('-- cursor: '):].strip()
                    if not cursor:
                        raise ValueError('Empty journal cursor')
                else:
                    if cursor is not None:
                        raise ValueError('Kernel records follow journal cursor trailer')
                    record = json.loads(line)
                    if (not isinstance(record, dict)
                            or not isinstance(record.get('__CURSOR'), str)
                            or not record['__CURSOR'].strip()):
                        raise ValueError('Kernel record missing cursor')
                    record_cursor = record['__CURSOR']
                    if record_cursor in batch_cursors or record_cursor in self.seen_cursors:
                        raise ValueError('Kernel journal repeated an already observed cursor')
                    if record.get('_BOOT_ID') != boot.replace('-', ''):
                        raise ValueError('Kernel record boot identity mismatch')
                    if not isinstance(record.get('MESSAGE'), str):
                        raise ValueError('Kernel record MESSAGE is not text')
                    priority = int(record['PRIORITY'])
                    if priority < 0 or priority > 7:
                        raise ValueError('Invalid kernel priority')
                    records.append(record)
                    batch_cursors.add(record_cursor)
            if records and (not cursor or cursor != records[-1]['__CURSOR']):
                raise ValueError('Kernel journal cursor not aligned with records')
            if initial and (not cursor or len(records) != 1):
                raise ValueError('Cannot establish kernel journal cursor')
            if not records and cursor is not None and cursor != self.cursor:
                raise ValueError('Kernel cursor advanced without records')
            # Commit only after the entire bounded query has been validated.
            # A query/parse failure will retry from the previous valid cursor.
            self.cursor = cursor or self.cursor
            self.seen_cursors.update(batch_cursors)
            alerts = [] if initial else self._classify(records)
            if alerts:
                self._failure('New kernel alert', 'kernel_alert')
            return self._result(True, records, alerts, 'New kernel alert' if alerts else None)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            self._failure(str(error), source)
            return self._result(False, [], [], str(error))

