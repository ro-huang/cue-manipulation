#!/usr/bin/env python3
"""RunPod helpers for Koriat GPU work: create a pod with agent SSH + secrets, wait for SSH, stop.

Requires:
  - RUNPOD_API_KEY in the environment (Cursor secret)
  - ~/.runpod/ssh/<key>.pub or --public-key-file for the SSH key injected at pod start (PUBLIC_KEY)
  - For run-gemma-v3: `runs/mve_llama8b/items_filtered.jsonl` + `primes.jsonl` must exist in the local repo before sync (discovery artifacts).
  - Create pods with `--ports "22/tcp"` so RunPod publishes a public SSH port (the RunPod API may not expose 22 otherwise).

Examples:
  python scripts/runpod_koriat.py create --name koriat-gemma-v3 --gpu h100-80 --disk 200
  python scripts/runpod_koriat.py wait-ssh <pod_id>
  python scripts/runpod_koriat.py sync <pod_id> --local /path/to/cue-manipulation
  python scripts/runpod_koriat.py bootstrap <pod_id>
  python scripts/runpod_koriat.py run-gemma-v3 <pod_id>
  python scripts/runpod_koriat.py ssh-test <pod_id>
  python scripts/runpod_koriat.py stop <pod_id>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _ssh_key_paths() -> tuple[Path, Path]:
    d = Path.home() / ".runpod" / "ssh"
    priv = d / "cursor-koriat-pod"
    pub = d / "cursor-koriat-pod.pub"
    return priv, pub


def _read_public_key(path: Path | None) -> str:
    if path and path.is_file():
        return path.read_text().strip()
    _, pub = _ssh_key_paths()
    if pub.is_file():
        return pub.read_text().strip()
    priv, _ = _ssh_key_paths()
    if priv.is_file():
        out = subprocess.check_output(
            ["ssh-keygen", "-y", "-f", str(priv)], text=True
        ).strip()
        return out
    sys.exit("No SSH public key: pass --public-key-file or create ~/.runpod/ssh/cursor-koriat-pod")


def _graphql_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for k in ("HF_TOKEN", "ANTHROPIC_API_KEY"):
        v = os.environ.get(k)
        if v:
            out[k] = v
    return out


def _install_runpod_api_key(api_key: str) -> None:
    """runpod.api.ctl_commands.create_pod reads `import runpod; runpod.api_key`."""
    import runpod

    runpod.api_key = api_key


def _create(args: argparse.Namespace) -> None:
    from runpod.api.ctl_commands import create_pod, get_gpu

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY is not set")
    _install_runpod_api_key(api_key)

    pub = _read_public_key(Path(args.public_key_file) if args.public_key_file else None)

    gpu_map = {
        "h100-80": "NVIDIA H100 80GB HBM3",
        "h100-pcie": "NVIDIA H100 PCIe",
        "a100-80": None,  # resolve below
    }
    gpu_type_id = gpu_map.get(args.gpu)
    if gpu_type_id is None and args.gpu == "a100-80":
        from runpod.api.graphql import run_graphql_query
        from runpod.api.queries import gpus as gpus_mod

        raw = run_graphql_query(gpus_mod.QUERY_GPU_TYPES, api_key=api_key)
        for t in raw["data"]["gpuTypes"]:
            if t.get("memoryInGb") == 80 and "A100" in t.get("displayName", ""):
                gpu_type_id = t["id"]
                break
        if gpu_type_id is None:
            sys.exit("Could not resolve A100 80GB gpu type id; use --gpu-type-id")
    elif gpu_type_id is None:
        gpu_type_id = args.gpu_type_id
    if not gpu_type_id:
        sys.exit("Unknown --gpu; use h100-80, h100-pcie, a100-80, or --gpu-type-id")

    get_gpu(gpu_type_id, api_key=api_key)  # validate id

    env = {"PUBLIC_KEY": pub}
    env.update(_graphql_env())
    if args.extra_env:
        for pair in args.extra_env:
            if "=" not in pair:
                sys.exit(f"Bad --env {pair}, expected KEY=value")
            k, _, v = pair.partition("=")
            env[k] = v

    image = args.image
    result = create_pod(
        name=args.name,
        image_name=image,
        gpu_type_id=gpu_type_id,
        cloud_type=args.cloud,
        gpu_count=1,
        volume_in_gb=args.volume_gb,
        container_disk_in_gb=args.disk_gb,
        env=env,
        ports=args.ports,
        start_ssh=True,
        support_public_ip=True,
    )
    pod_id = result.get("id")
    print(json.dumps({"pod_id": pod_id, "raw": result}, indent=2))
    if pod_id:
        print(f"\nPod ID: {pod_id}\nWait for SSH: python scripts/runpod_koriat.py wait-ssh {pod_id}")


def _get_pod(pod_id: str) -> dict:
    from runpod.api.ctl_commands import get_pod

    return get_pod(pod_id, api_key=os.environ["RUNPOD_API_KEY"])


def _ssh_endpoint(pod_id: str) -> tuple[str, int] | None:
    p = _get_pod(pod_id)
    rt = p.get("runtime") or {}
    ports = rt.get("ports") or []
    for port in ports:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return str(port["ip"]), int(port["publicPort"])
    return None


def _wait_ssh(args: argparse.Namespace) -> None:
    _install_runpod_api_key(os.environ["RUNPOD_API_KEY"])
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        ep = _ssh_endpoint(args.pod_id)
        if ep:
            ip, port = ep
            print(f"SSH endpoint: {ip} port {port}")
            priv, _ = _ssh_key_paths()
            if not priv.is_file():
                sys.exit(f"Missing private key {priv}")
            r = subprocess.run(
                [
                    "ssh",
                    "-oBatchMode=yes",
                    "-oStrictHostKeyChecking=accept-new",
                    "-oConnectTimeout=12",
                    "-i",
                    str(priv),
                    "-p",
                    str(port),
                    f"root@{ip}",
                    "echo ok",
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                print(r.stdout.strip())
                return
        time.sleep(args.interval)
    sys.exit("Timeout waiting for SSH")


def _ssh_test(args: argparse.Namespace) -> None:
    _install_runpod_api_key(os.environ["RUNPOD_API_KEY"])
    ep = _ssh_endpoint(args.pod_id)
    if not ep:
        sys.exit("No public SSH port yet (pod still starting?)")
    ip, port = ep
    priv, _ = _ssh_key_paths()
    cmd = args.remote_command
    r = subprocess.run(
        [
            "ssh",
            "-oBatchMode=yes",
            "-oStrictHostKeyChecking=no",
            "-oConnectTimeout=15",
            "-i",
            str(priv),
            "-p",
            str(port),
            f"root@{ip}",
            cmd,
        ]
    )
    sys.exit(r.returncode)


def _stop(args: argparse.Namespace) -> None:
    from runpod.api.ctl_commands import stop_pod

    _install_runpod_api_key(os.environ["RUNPOD_API_KEY"])
    out = stop_pod(args.pod_id)
    print(json.dumps(out, indent=2))


def _sync_code(args: argparse.Namespace) -> None:
    """Copy project tree to /workspace/koriat (rsync over SSH, fallback: tar)."""
    _install_runpod_api_key(os.environ["RUNPOD_API_KEY"])
    ep = _ssh_endpoint(args.pod_id)
    if not ep:
        sys.exit("No public SSH port (pod starting?) — run wait-ssh first")
    ip, port = ep
    priv, _ = _ssh_key_paths()
    local = Path(args.local_dir).resolve()
    if not (local / "pyproject.toml").is_file():
        sys.exit(f"Expected a Koriat repo at {local}")

    mve = local / "runs" / "mve_llama8b" / "items_filtered.jsonl"
    if not mve.is_file():
        print(
            "WARNING: runs/mve_llama8b/items_filtered.jsonl not found locally — "
            "bootstrap_replication.sh will fail until you copy the discovery run artifacts.",
            file=sys.stderr,
        )

    ssh_target = f"ssh -oBatchMode=yes -oStrictHostKeyChecking=no -i {priv} -p {port}"

    if shutil.which("rsync"):
        rsync_excl = [
            "--exclude=.git/",
            "--exclude=.venv/",
            "--exclude=__pycache__/",
            "--exclude=.pytest_cache/",
            "--exclude=.mypy_cache/",
            "--exclude=*.egg-info/",
            "--exclude=runs/replication_*/",
            "--exclude=runs/replication_v*/",
            "--exclude=runs/mve_llama8b/trials.parquet",
        ]
        cmd = [
            "rsync",
            "-az",
            "--delete",
            "--omit-dir-times",
            "-e",
            ssh_target,
            *rsync_excl,
            f"{local}/",
            f"root@{ip}:/workspace/koriat/",
        ]
        print(f"rsync {local}/ → root@{ip}:/workspace/koriat/ …")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"rsync failed ({r.returncode})")
        print("Sync done.")
        return

    # Fallback: tar stream (no partial runs/ control beyond excluding whole runs/)
    excludes = [
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        "--exclude=runs/replication_v2",
        "--exclude=runs/replication_v3_gemma",
        "--exclude=runs/replication_qwen7b",
        "--exclude=.mypy_cache",
        "--exclude=*.egg-info",
    ]
    tar_cmd = ["tar", "czf", "-", "-C", str(local)] + excludes + ["."]
    remote_prep = "rm -rf /workspace/koriat && mkdir -p /workspace/koriat && cd /workspace/koriat && tar xzf -"
    ssh_prefix = [
        "ssh",
        "-oBatchMode=yes",
        "-oStrictHostKeyChecking=no",
        "-oConnectTimeout=30",
        "-i",
        str(priv),
        "-p",
        str(port),
        f"root@{ip}",
        remote_prep,
    ]
    print(f"tar-sync {local} → root@{ip}:/workspace/koriat/ …")
    p1 = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
    r = subprocess.run(ssh_prefix, stdin=p1.stdout)
    p1.stdout.close()
    p1.wait()
    if p1.returncode != 0:
        sys.exit(f"tar create failed ({p1.returncode})")
    if r.returncode != 0:
        sys.exit(f"remote tar failed ({r.returncode})")
    print("Sync done.")


def _bootstrap(args: argparse.Namespace) -> None:
    _install_runpod_api_key(os.environ["RUNPOD_API_KEY"])
    ep = _ssh_endpoint(args.pod_id)
    if not ep:
        sys.exit("No SSH endpoint")
    ip, port = ep
    priv, _ = _ssh_key_paths()
    remote = r"""
set -euo pipefail
while IFS= read -r -d '' line; do export "$line"; done < /proc/1/environ
cd /workspace/koriat
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -e ".[bnb]"
mkdir -p ~/.cache/huggingface
if [ -n "${HF_TOKEN:-}" ]; then printf %s "$HF_TOKEN" > ~/.cache/huggingface/token; fi
echo '[bootstrap] ok'
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
"""
    r = subprocess.run(
        [
            "ssh",
            "-oBatchMode=yes",
            "-oStrictHostKeyChecking=no",
            "-oConnectTimeout=60",
            "-i",
            str(priv),
            "-p",
            str(port),
            f"root@{ip}",
            remote,
        ]
    )
    if r.returncode != 0:
        sys.exit(r.returncode)


def _run_gemma_v3(args: argparse.Namespace) -> None:
    """sync + pip install + nohup replication v3 (log under /workspace/koriat/logs/)."""
    _install_runpod_api_key(os.environ["RUNPOD_API_KEY"])
    local = Path(args.local_dir).resolve()
    if not (local / "runs" / "mve_llama8b" / "items_filtered.jsonl").is_file():
        sys.exit(
            "Missing runs/mve_llama8b/items_filtered.jsonl locally. "
            "Copy the discovery run directory from your machine where the MVE exists, then re-run."
        )
    _sync_code(args)
    _bootstrap(args)
    ep = _ssh_endpoint(args.pod_id)
    if not ep:
        sys.exit("No SSH endpoint")
    ip, port = ep
    priv, _ = _ssh_key_paths()
    log = "/workspace/koriat/logs/replication_v3_gemma.log"
    remote = f"""
set -euo pipefail
while IFS= read -r -d '' line; do export "$line"; done < /proc/1/environ
cd /workspace/koriat
mkdir -p logs
nohup bash scripts/run_replication_v3_gemma.sh > {log} 2>&1 < /dev/null & disown
echo "Started; log: {log}"
sleep 2
tail -20 {log} || true
"""
    r = subprocess.run(
        [
            "ssh",
            "-oBatchMode=yes",
            "-oStrictHostKeyChecking=no",
            "-oConnectTimeout=30",
            "-i",
            str(priv),
            "-p",
            str(port),
            f"root@{ip}",
            remote,
        ]
    )
    sys.exit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="RunPod helpers for Koriat pods")
    ap.add_argument(
        "--api-key",
        default=os.environ.get("RUNPOD_API_KEY"),
        help="Override RUNPOD_API_KEY",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="Deploy GPU pod with PUBLIC_KEY + HF_TOKEN env")
    c.add_argument("--name", required=True)
    c.add_argument(
        "--gpu",
        default="h100-80",
        help="Shortcut: h100-80, h100-pcie, a100-80, or pass raw gpu type id via --gpu-type-id",
    )
    c.add_argument("--gpu-type-id", default=None, help="Raw RunPod gpuTypes.id")
    c.add_argument("--disk", dest="disk_gb", type=int, default=200)
    c.add_argument("--volume-gb", dest="volume_gb", type=int, default=0)
    c.add_argument(
        "--image",
        default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    )
    c.add_argument("--cloud", default="ALL", choices=["ALL", "COMMUNITY", "SECURE"])
    c.add_argument(
        "--ports",
        default="22/tcp",
        help="RunPod port spec (must include 22/tcp for SSH), e.g. 22/tcp,8888/http",
    )
    c.add_argument("--public-key-file", default=None)
    c.add_argument(
        "--env",
        dest="extra_env",
        action="append",
        default=[],
        help="Extra KEY=value for pod env (repeatable)",
    )
    c.set_defaults(func=_create)

    w = sub.add_parser("wait-ssh", help="Poll until SSH accepts our key")
    w.add_argument("pod_id")
    w.add_argument("--timeout", type=int, default=600)
    w.add_argument("--interval", type=int, default=10)
    w.set_defaults(func=_wait_ssh)

    t = sub.add_parser("ssh-test", help="Run one remote command via SSH")
    t.add_argument("pod_id")
    t.add_argument(
        "remote_command",
        nargs="?",
        default="echo connected && nvidia-smi -L",
    )
    t.set_defaults(func=_ssh_test)

    s = sub.add_parser("stop", help="Stop pod (billing pause; container disk may persist briefly)")
    s.add_argument("pod_id")
    s.set_defaults(func=_stop)

    sy = sub.add_parser("sync", help="Tar-copy repo to /workspace/koriat on the pod")
    sy.add_argument("pod_id")
    sy.add_argument(
        "--local",
        dest="local_dir",
        default=".",
        help="Path to repo root (default: cwd)",
    )
    sy.set_defaults(func=_sync_code)

    b = sub.add_parser(
        "bootstrap",
        help="pip install -e .[bnb] on pod and write HF_TOKEN to huggingface cache",
    )
    b.add_argument("pod_id")
    b.set_defaults(func=_bootstrap)

    g = sub.add_parser(
        "run-gemma-v3",
        help="sync + bootstrap + nohup scripts/run_replication_v3_gemma.sh",
    )
    g.add_argument("pod_id")
    g.add_argument("--local", dest="local_dir", default=".", help="Repo root to sync")
    g.set_defaults(func=_run_gemma_v3)

    args = ap.parse_args()
    if args.api_key:
        os.environ["RUNPOD_API_KEY"] = args.api_key
    if not os.environ.get("RUNPOD_API_KEY"):
        sys.exit("RUNPOD_API_KEY not set")
    args.func(args)


if __name__ == "__main__":
    main()
