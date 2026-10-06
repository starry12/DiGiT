"""Keep observed loaded service state separate from systemd's absent defaults."""


def terminal(state):
    return (state.get('LoadState') == 'loaded' and state.get('MainPID') == '0'
            and (state.get('ActiveState') in ('inactive', 'failed')
                 or (state.get('ActiveState') == 'active' and state.get('SubState') == 'exited')))


class ServiceEvidence:
    def __init__(self):
        self.last_loaded = None
        self.first_terminal_loaded = None
        self.final_lookup = None
        self.before_stop = None

    def observe(self, state):
        self.final_lookup = dict(state)
        if state.get('LoadState') == 'loaded':
            self.last_loaded = dict(state)
            if terminal(state) and self.first_terminal_loaded is None:
                self.first_terminal_loaded = dict(state)
        return state

    def stopping(self, state):
        self.before_stop = dict(state)
        self.observe(state)

    def worker_state(self):
        # A previously running process stays unproven until a loaded terminal
        # state is actually observed. not-found never becomes a successful exit.
        return dict(self.first_terminal_loaded or self.last_loaded or {})

    def receipt(self):
        return dict(last_loaded=self.last_loaded,
                    first_terminal_loaded=self.first_terminal_loaded,
                    before_stop=self.before_stop, final_lookup=self.final_lookup,
                    terminal_exit_observed=self.first_terminal_loaded is not None)
