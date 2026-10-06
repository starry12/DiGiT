"""Fail-closed runtime observation across systemd's post-exit cgroup removal.

This module performs no launch, signals, GPU work, or service mutation. The
caller collects the worker cgroup before every observation, including exits.
"""
import errno
import os
from pathlib import Path


class RuntimeMonitor:
    def __init__(self, guard, cgroup, worker_passed):
        if (not isinstance(cgroup, str) or not cgroup.startswith('/')
                or cgroup == '/' or '..' in cgroup.split('/')):
            raise ValueError('Expected an absolute worker cgroup path')
        self.guard, self.cgroup = guard, cgroup
        self.worker_passed = worker_passed
        self.directory = Path('/sys/fs/cgroup') / cgroup.lstrip('/')
        self.valid_runtime_samples = 0

    @staticmethod
    def _terminal(state):
        return (state.get('ActiveState') in ('inactive', 'failed')
                or (state.get('ActiveState') == 'active'
                    and state.get('SubState') == 'exited'))

    @classmethod
    def _successful_exit(cls, state):
        return (state.get('LoadState') == 'loaded'
                and state.get('MainPID') == '0'
                and state.get('Result') == 'success'
                and state.get('ExecMainStatus') == '0'
                and state.get('ActiveState') != 'failed'
                and cls._terminal(state))

    def _fail(self, reason, decision=None):
        result = dict(decision or {})
        reasons = list(self.guard.failed) + list(result.get('reasons', [])) + [reason]
        self.guard.failed = list(dict.fromkeys(reasons))
        result.update(ok=False, reasons=list(self.guard.failed))
        result.setdefault('observations', [])
        return result

    def _check(self, sample, host_only=False):
        require_cgroup = self.guard.require_cgroup
        try:
            if host_only:
                self.guard.require_cgroup = False
            return self.guard.check(sample)
        except Exception as error:
            return self._fail('runtime_guard_exception: ' + repr(error))
        finally:
            self.guard.require_cgroup = require_cgroup

    def _receipt_failure(self, state, report):
        if not self._successful_exit(state):
            return 'worker_exit_not_successful'
        try:
            receipt = report()
            if not isinstance(receipt, dict) or not self.worker_passed(state, receipt):
                return 'worker_receipt_not_accepted'
        except Exception as error:
            return 'worker_receipt_unavailable: ' + repr(error)
        return None

    def _removal_error(self, sample):
        """Only the first read may vanish; later partial cgroup data is unsafe."""
        details = sample.get('error_details', [])
        errors = sample.get('errors', [])
        if not isinstance(details, list) or not isinstance(errors, list):
            return None
        cgroup_errors = [item for item in details
                         if isinstance(item, dict) and item.get('component') == 'cgroup']
        if len(cgroup_errors) != 1:
            return None
        error = cgroup_errors[0]
        message = error.get('message')
        if (error.get('exception') != 'FileNotFoundError'
                or error.get('errno') != errno.ENOENT
                or error.get('filename') != str(self.directory / 'memory.stat')
                or not isinstance(message, str) or not message.startswith('cgroup: ')
                or errors.count(message) != 1):
            return None
        return error

    def observe(self, state_before, sample, show, report):
        """Return the original guard decision or a documented terminal transition.

        ``show`` and ``report`` are zero-argument readers for this worker only.
        Missing/invalid receipts and reader failures become sticky rejections.
        The input sample is never mutated or replaced with a later observation.
        """
        state_after, transition, terminal = {}, None, False

        def result(decision):
            return dict(decision=decision, terminal=terminal,
                        state_after=state_after, transition=transition,
                        valid_runtime_samples=self.valid_runtime_samples)

        try:
            state_after = show()
            if not isinstance(state_before, dict) or not isinstance(state_after, dict):
                raise ValueError('Malformed service state')
        except Exception as error:
            return result(self._fail('worker_state_unavailable: ' + repr(error)))
        if not isinstance(sample, dict):
            return result(self._fail('runtime_sample_malformed'))
        if self._terminal(state_before) and not self._successful_exit(state_before):
            return result(self._fail('prior_worker_exit_not_successful'))
        if self._terminal(state_before) and not self._terminal(state_after):
            return result(self._fail('worker_terminal_state_regressed'))
        if state_after.get('LoadState') != 'loaded':
            return result(self._fail('worker_unit_not_loaded'))

        if 'cgroup' in sample:
            if (not isinstance(sample['cgroup'], dict)
                    or sample['cgroup'].get('path') != self.cgroup):
                return result(self._fail('runtime_cgroup_identity_mismatch'))
            decision = self._check(sample)
            if decision['ok']:
                self.valid_runtime_samples += 1
            if not decision['ok'] or not self._terminal(state_after):
                return result(decision)
            failure = self._receipt_failure(state_after, report)
            if failure:
                return result(self._fail(failure, decision))
            terminal = True
            transition = dict(kind='observed_exit', cgroup=self.cgroup,
                              state_before=dict(state_before))
            return result(decision)

        missing = self._removal_error(sample)
        if missing is None:
            return result(self._check(sample))
        if self.guard.failed:
            return result(self._fail('cgroup_removal_after_guard_failure'))
        if self.valid_runtime_samples < 1:
            return result(self._fail('cgroup_removal_without_runtime_observation'))
        failure = self._receipt_failure(state_after, report)
        if failure:
            return result(self._fail(failure))
        try:
            os.stat(str(self.directory))
        except OSError as error:
            if error.errno != errno.ENOENT:
                return result(self._fail('cgroup_directory_observation_failed: ' + repr(error)))
        except Exception as error:
            return result(self._fail('cgroup_directory_observation_failed: ' + repr(error)))
        else:
            return result(self._fail('cgroup_directory_still_present'))

        host_sample = dict(sample)
        host_sample['errors'] = [item for item in sample['errors']
                                 if item != missing['message']]
        host_sample['error_details'] = [item for item in sample['error_details']
                                        if item is not missing]
        decision = self._check(host_sample, host_only=True)
        if not decision['ok']:
            return result(decision)
        terminal = True
        transition = dict(kind='cgroup_removed_after_exit', cgroup=self.cgroup,
                          missing_file=missing['filename'], error=dict(missing),
                          state_before=dict(state_before))
        return result(decision)
