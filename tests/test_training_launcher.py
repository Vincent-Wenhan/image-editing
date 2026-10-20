"""The lease-owning launcher waits for worker termination, including cancellation."""

import signal
import subprocess

import pytest

from src.training import launcher


@pytest.mark.parametrize("outcome", [0, 1, "interrupt"])
def test_launcher_waits_and_restores_handlers(tmp_path, monkeypatch, outcome):
    events = []

    class Process:
        def __init__(self, argv, cwd):
            events.append(("start", argv, cwd))
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            events.append(("wait", timeout))
            if outcome == "interrupt" and self.waits == 1:
                raise KeyboardInterrupt
            return 0 if outcome == "interrupt" else outcome

        def poll(self):
            return None

        def send_signal(self, signum):
            events.append(("signal", signum))

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    previous = object()

    def handler(signum, function):
        events.append(("handler", signum, function))
        return previous

    monkeypatch.setattr(launcher.signal, "signal", handler)
    if outcome == 0:
        launcher.wait_for_worker(["worker"], tmp_path)
    else:
        error = KeyboardInterrupt if outcome == "interrupt" else subprocess.CalledProcessError
        with pytest.raises(error):
            launcher.wait_for_worker(["worker"], tmp_path)
    assert events[0] == ("start", ["worker"], tmp_path)
    assert events[-1] == ("handler", signal.SIGTERM, previous)
    if outcome == "interrupt":
        assert ("signal", signal.SIGINT) in events and ("wait", 30) in events
