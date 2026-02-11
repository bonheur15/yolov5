#!/usr/bin/env python3
"""Launch multiple YOLOv5 RTSP -> HLS pipelines, one output folder per camera tag."""

import signal
import subprocess
import sys
from pathlib import Path

# Edit this list with your cameras.
CAMERAS = [
    # {"tag": "camera2", "url": "rtsp://user:pass@ip:554/stream"},
]


# Global output root. Each camera writes to runs/hls/<tag>/
OUTPUT_ROOT = Path("runs/hls")


# Detection/HLS defaults requested:
# - only persons
# - no text labels
# - ellipse around persons
# - lower confidence for person detection
COMMON_ARGS = [
    "--weights",
    "yolov5s.pt",
    "--save-hls",
    "--hls-time",
    "2",
    "--hls-list-size",
    "100",
    "--hls-delete-threshold",
    "2",
    "--hls-segment-type",
    "ts",
    "--hls-vcodec",
    "auto",
    "--rtsp-transport",
    "tcp",
    "--hls-fps",
    "20",
    "--classes",
    "0",
    "--conf-thres",
    "0.05",
    "--hide-labels",
    "--hide-conf",
    "--person-ellipse",
]


def build_cmd(camera):
    return [
        sys.executable,
        "detect.py",
        "--source",
        camera["url"],
        "--project",
        str(OUTPUT_ROOT),
        "--name",
        camera["tag"],
        "--exist-ok",
        *COMMON_ARGS,
    ]


def main():
    if not CAMERAS:
        raise SystemExit("CAMERAS list is empty.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    procs = []

    for cam in CAMERAS:
        tag = cam["tag"].strip()
        url = cam["url"].strip()
        if not tag or not url:
            raise SystemExit(f"Invalid camera config: {cam}")

        cmd = build_cmd({"tag": tag, "url": url})
        print(f"[start] {tag}: {' '.join(cmd)}")
        procs.append((tag, subprocess.Popen(cmd)))

    stopping = False

    def shutdown(_signum=None, _frame=None):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print("\n[stop] terminating camera processes...")
        for tag, p in procs:
            if p.poll() is None:
                print(f"[stop] {tag}")
                p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    failed = False
    try:
        for tag, p in procs:
            code = p.wait()
            if code != 0:
                failed = True
                print(f"[exit] {tag} exited with code {code}")
    finally:
        shutdown()

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
