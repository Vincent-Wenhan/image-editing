"""Check GPU idleness and hold a per-user UUID lease while the child runs."""

import argparse
import csv
import fcntl
import io
import os
import signal
import subprocess
import tempfile
from pathlib import Path

from src.infra.process import command


def inventory():
    fields = "index,uuid,name,memory.total,memory.used,utilization.gpu"
    output = command(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    processes = command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    )
    busy = {line.split(",")[0].strip() for line in processes.splitlines() if line.strip()}
    cards = []
    for row in csv.reader(io.StringIO(output)):
        index, uuid, name, total, used, utilization = (item.strip() for item in row)
        cards.append(
            dict(
                index=int(index),
                uuid=uuid,
                name=name,
                total=int(total),
                used=int(used),
                utilization=int(utilization),
                process=uuid in busy,
            )
        )
    return cards


def is_idle(card, max_used=128, max_util=5):
    return not card["process"] and card["used"] <= max_used and card["utilization"] <= max_util


def run(args):
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise ValueError("Supply a command after --")
    lock_root = Path(tempfile.gettempdir()) / f"image-edit-gpu-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    for card in inventory():
        if args.device is not None and card["index"] != args.device:
            continue
        if not is_idle(card):
            continue
        # Keep the file: unlinking a live lock file would permit a second inode/lease.
        with (lock_root / f"{card['uuid']}.lock").open("a+") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            current = next(c for c in inventory() if c["uuid"] == card["uuid"])
            if not is_idle(current):
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = card["uuid"]
            env["IMAGE_EDIT_GPU_UUID"] = card["uuid"]
            env.setdefault("OMP_NUM_THREADS", "8")
            print(f"Using idle GPU {card['index']}: {card['name']} ({card['uuid']})", flush=True)
            process = subprocess.Popen(cmd, env=env, start_new_session=True)
            previous = {}

            def forward(signum, _frame, process=process):
                if process.poll() is None:
                    os.killpg(process.pid, signum)

            try:
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous[signum] = signal.signal(signum, forward)
                result = process.wait()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
            raise SystemExit(result if result >= 0 else 128 - result)
    raise RuntimeError("No idle, unlocked GPU available; no task was started")


def require_lease():
    uuid = os.environ.get("IMAGE_EDIT_GPU_UUID")
    if not uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("Launch this task with: image-edit gpu -- image-edit ...")


def configure(parser):
    parser.add_argument("--device", type=int, help="Physical GPU index; otherwise first idle card")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parser.set_defaults(func=run)
