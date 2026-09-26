#!/usr/bin/env python3
"""
pi_client.py - Raspberry Pi Remote Client & StickCam HDMI Display Controller.

Provides remote management, diagnostics, and testing commands from Windows/PC:
  - status:   Query live HTTP telemetry (focus score, OCR result) and systemd service state
  - trigger:  Simulate Button 1 (Burst Capture & OCR) remotely
  - reset:    Simulate Button 2 (Re-capture & Reset Viewfinder) remotely
  - mirror:   Download a live snapshot of the 3.5" HDMI display screen
  - capture:  Capture a raw photo on the Pi and download to PC
  - scan:     Run on-device OCR on a remote image
  - deploy:   Automated deployment of the 3.5" HDMI scanner service to the Pi
  - start:    Start the seal-hdmi service on the Pi
  - stop:     Stop the seal-hdmi service on the Pi
  - restart:  Restart the seal-hdmi service on the Pi
  - logs:     Tail recent seal-hdmi systemd journal logs
  - exec:     Run arbitrary command on the Pi via SSH
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path

import paramiko

# Ensure safe UTF-8 printing on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def safe_print(*args, **kwargs):
    """Safely prints text on any Windows console encoding without crashing."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        sanitized = sep.join(str(a).encode("ascii", errors="replace").decode("ascii") for a in args)
        print(sanitized, end=end)

FALLBACK_HOSTS = ["pizero2.local", "192.168.68.151", "192.168.68.145", "192.168.68.129"]
DEFAULT_HOST = os.environ.get("PI_HOST", "pizero2.local")
DEFAULT_USER = os.environ.get("PI_USER", "stickcam")
DEFAULT_PASS = os.environ.get("PI_PASS", "Dubo2024")
DEFAULT_PORT = int(os.environ.get("PI_HTTP_PORT", "8000"))


def get_ssh_client(host_override=None, user_override=None, pass_override=None):
    """Establishes an SSH connection with smart fallback across known hosts."""
    user = user_override or DEFAULT_USER
    pwd = pass_override or DEFAULT_PASS
    target_hosts = [host_override] if host_override else ([DEFAULT_HOST] + [h for h in FALLBACK_HOSTS if h != DEFAULT_HOST])

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    for h in target_hosts:
        try:
            client.connect(h, port=22, username=user, password=pwd, timeout=6)
            return client, h
        except Exception:
            continue

    print(f"[Error] Failed to connect to {user} on any host: {target_hosts}")
    sys.exit(1)


def get_active_host(host_override=None) -> str:
    """Finds first reachable host IP/domain."""
    if host_override:
        return host_override
    hosts = [DEFAULT_HOST] + [h for h in FALLBACK_HOSTS if h != DEFAULT_HOST]
    import socket
    for h in hosts:
        try:
            socket.gethostbyname(h)
            return h
        except Exception:
            continue
    return DEFAULT_HOST


def cmd_status(args):
    """Checks live HTTP state (focus, result) and systemd service status."""
    host = get_active_host(args.host)
    print("=" * 64)
    print(f"  STICK CAM STATUS CHECK: {host}")
    print("=" * 64)

    # 1. Query Web API
    api_url = f"http://{host}:{args.port}/api/status"
    print(f"[*] Querying Web API: {api_url} ...")
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "PiClient/7.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("[+] Web API is ONLINE & RESPONDING:")
            res = data.get("last_result", {})
            print(f"    - Focus Sharpness: {data.get('sharpness', 'N/A')} ({data.get('focus_quality', 'N/A')})")
            print(f"    - Is Capturing:    {data.get('is_capturing', False)}")
            print(f"    - Last Serial:     {res.get('serial_number', 'None')}")
            print(f"    - Confidence:      {res.get('confidence', 0.0) * 100:.1f}%")
            print(f"    - OCR Time:        {res.get('ocr_time_ms', 0.0):.1f} ms")
            print(f"    - Last Status:     {res.get('status', 'N/A')} - {res.get('message', '')}")
    except Exception as e:
        print(f"[-] Web API not reachable at port {args.port}: {e}")

    # 2. Check Systemd Service via SSH
    print("\n[*] Checking systemd service state via SSH...")
    client, connected_host = get_ssh_client(args.host)
    try:
        cmd = "systemctl is-active seal-hdmi.service 2>&1; systemctl status seal-hdmi.service --no-pager"
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode("utf-8", errors="replace")
        safe_print(out)
    finally:
        client.close()
    print("=" * 64)


def cmd_trigger(args):
    """Triggers Button 1 (Burst Capture & OCR) via HTTP API."""
    host = get_active_host(args.host)
    url = f"http://{host}:{args.port}/api/capture"
    print(f"[*] Sending Button 1 (Capture & OCR) trigger to {url}...")
    try:
        req = urllib.request.Request(url, data=b"", headers={"User-Agent": "PiClient/7.0"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[+] Capture Triggered! Response: {data}")
    except Exception as e:
        print(f"[-] HTTP trigger failed ({e}). Attempting fallback via SSH...")
        client, connected_host = get_ssh_client(args.host)
        try:
            stdin, stdout, stderr = client.exec_command(
                f"curl -s -X POST http://localhost:{args.port}/api/capture"
            )
            print("[+] SSH Fallback Output:", stdout.read().decode())
        finally:
            client.close()


def cmd_reset(args):
    """Triggers Button 2 (Re-capture & Reset Viewfinder) via HTTP API."""
    host = get_active_host(args.host)
    url = f"http://{host}:{args.port}/api/recapture"
    print(f"[*] Sending Button 2 (Re-capture & Reset) trigger to {url}...")
    try:
        req = urllib.request.Request(url, data=b"", headers={"User-Agent": "PiClient/7.0"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[+] Reset Triggered! Response: {data}")
    except Exception as e:
        print(f"[-] HTTP reset failed ({e}). Attempting fallback via SSH...")
        client, connected_host = get_ssh_client(args.host)
        try:
            stdin, stdout, stderr = client.exec_command(
                f"curl -s -X POST http://localhost:{args.port}/api/recapture"
            )
            print("[+] SSH Fallback Output:", stdout.read().decode())
        finally:
            client.close()


def cmd_mirror(args):
    """Downloads a snapshot of the current 3.5" HDMI screen."""
    host = get_active_host(args.host)
    output_path = Path(args.output)
    url = f"http://{host}:{args.port}/snapshot"

    print(f"[*] Downloading screen snapshot from {url} -> {output_path}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PiClient/7.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read()
            with open(output_path, "wb") as f:
                f.write(content)
            print(f"[+] Snapshot saved successfully as {output_path.resolve()} ({len(content)} bytes)")
    except Exception as e:
        print(f"[-] Could not download snapshot via HTTP: {e}")


def cmd_deploy(args):
    """Invokes deploy_hdmi_pi.py to push all files and enable seal-hdmi service."""
    from deploy_hdmi_pi import deploy
    print("[*] Launching deploy_hdmi_pi...")
    deploy()


def cmd_service_ctl(action: str, host_override=None):
    """Controls the seal-hdmi systemd service (start/stop/restart)."""
    client, connected_host = get_ssh_client(host_override)
    user = DEFAULT_USER
    pwd = DEFAULT_PASS
    print(f"[*] Executing 'sudo systemctl {action} seal-hdmi.service' on {connected_host}...")
    try:
        stdin, stdout, stderr = client.exec_command(f"sudo -S systemctl {action} seal-hdmi.service", get_pty=True)
        stdin.write(f"{pwd}\n")
        stdin.flush()
        print(stdout.read().decode("utf-8", errors="replace"))

        # Print status after control action
        stdin, stdout, stderr = client.exec_command("systemctl status seal-hdmi.service --no-pager")
        print(stdout.read().decode("utf-8", errors="replace"))
    finally:
        client.close()


def cmd_logs(args):
    """Fetches recent journalctl logs for seal-hdmi.service."""
    client, connected_host = get_ssh_client(args.host)
    n = args.lines
    print(f"[*] Fetching last {n} log lines for seal-hdmi.service on {connected_host}...")
    try:
        stdin, stdout, stderr = client.exec_command(f"journalctl -u seal-hdmi.service -n {n} --no-pager")
        print(stdout.read().decode("utf-8", errors="replace"))
    finally:
        client.close()


def cmd_capture(args):
    """Legacy/Direct capture command on Pi and download to local PC."""
    client, connected_host = get_ssh_client(args.host)
    user = DEFAULT_USER
    output_filename = args.output
    remote_path = f"/home/{user}/seal-scanner/{output_filename}"
    local_path = Path(output_filename)

    print(f"[*] Capturing photo on Pi Zero ({output_filename})...")
    if args.usb:
        cmd = f"cd ~/seal-scanner && python3 capture_usb.py --width {args.width} --height {args.height}"
    else:
        cmd = f"cd ~/seal-scanner && python3 capture_seal.py {output_filename} --delay {args.delay} --width {args.width} --height {args.height}"

    stdin, stdout, stderr = client.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')

    if exit_code != 0 and not args.usb:
        print("[Notice] CSI capture did not succeed, trying USB camera fallback...")
        cmd_usb = "cd ~/seal-scanner && python3 capture_usb.py"
        stdin2, stdout2, stderr2 = client.exec_command(cmd_usb)
        out += "\n" + stdout2.read().decode('utf-8', errors='replace')
        err += "\n" + stderr2.read().decode('utf-8', errors='replace')

    print(out)
    if err.strip():
        print("[Stderr]:\n" + err)

    # Download the image via SFTP
    sftp = client.open_sftp()
    try:
        files = [f for f in sftp.listdir(f"/home/{user}/seal-scanner") if f.endswith(".jpg")]
        if files:
            files.sort()
            latest = files[-1]
            remote_img = f"/home/{user}/seal-scanner/{latest}"
            print(f"[*] Downloading {latest} -> {local_path.resolve()}...")
            sftp.get(remote_img, str(local_path))
            print(f"[+] Photo saved locally as {local_path}")
    except Exception as e:
        print(f"[-] Could not download photo: {e}")
    finally:
        sftp.close()
        client.close()


def cmd_scan(args):
    """Runs on-device OCR on an image on the Pi and downloads debug crops."""
    client, connected_host = get_ssh_client(args.host)
    user = DEFAULT_USER
    image = args.image
    roi_str = f"--roi {' '.join(str(v) for v in args.roi)}"
    digits_str = f"--digits {args.digits}" if args.digits > 0 else ""
    rotate_str = f"--rotate {args.rotate}" if args.rotate != 0 else ""

    cmd = f"cd ~/seal-scanner && python3 scan_seal.py {image} {roi_str} {digits_str} {rotate_str}"
    print(f"[*] Running OCR on Pi: {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    print(out)
    if err.strip():
        print("[Stderr]:", err)

    stem = Path(image).stem
    debug_remote_dir = f"/home/{user}/seal-scanner/{stem}_ocr"
    local_debug_dir = Path(f"{stem}_ocr")
    sftp = client.open_sftp()
    try:
        debug_files = sftp.listdir(debug_remote_dir)
        local_debug_dir.mkdir(exist_ok=True)
        for df in debug_files:
            sftp.get(f"{debug_remote_dir}/{df}", str(local_debug_dir / df))
        print(f"[+] Debug crops and binarized images saved to {local_debug_dir.resolve()}/")
    except Exception:
        pass
    finally:
        sftp.close()
        client.close()


def cmd_run(cmd_str, host_override=None):
    """Executes arbitrary command on the Pi via SSH."""
    client, connected_host = get_ssh_client(host_override)
    print(f"[*] Remote command on {connected_host}: {cmd_str}")
    stdin, stdout, stderr = client.exec_command(cmd_str)
    print(stdout.read().decode('utf-8', errors='replace'))
    err = stderr.read().decode('utf-8', errors='replace')
    if err.strip():
        print("ERR:", err)
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StickCam Pi Zero 2W 3.5\" HDMI Remote Controller")
    parser.add_argument("--host", default=None, help="Pi IP address or hostname (default: auto-detect)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP API port (default: {DEFAULT_PORT})")

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # status subcommand
    p_stat = subparsers.add_parser("status", help="Check live HTTP telemetry & systemd service status")

    # trigger subcommand (Button 1)
    p_trig = subparsers.add_parser("trigger", help="Simulate Button 1: Burst Capture & OCR")

    # reset subcommand (Button 2)
    p_res = subparsers.add_parser("reset", help="Simulate Button 2: Re-capture & Reset Viewfinder")

    # mirror subcommand
    p_mir = subparsers.add_parser("mirror", help="Download current snapshot from HDMI scanner")
    p_mir.add_argument("-o", "--output", default="hdmi_snapshot.jpg", help="Output local filename")

    # deploy subcommand
    p_dep = subparsers.add_parser("deploy", help="Deploy HDMI service and scripts to Raspberry Pi")

    # service lifecycle subcommands
    p_start = subparsers.add_parser("start", help="Start seal-hdmi service on Pi")
    p_stop = subparsers.add_parser("stop", help="Stop seal-hdmi service on Pi")
    p_rest = subparsers.add_parser("restart", help="Restart seal-hdmi service on Pi")

    # logs subcommand
    p_logs = subparsers.add_parser("logs", help="Tail seal-hdmi service journal logs")
    p_logs.add_argument("-n", "--lines", type=int, default=40, help="Number of log lines to show")

    # capture subcommand
    p_cap = subparsers.add_parser("capture", help="Directly capture photo on Pi and download to PC")
    p_cap.add_argument("output", default="test.jpg", nargs="?", help="Filename for saved photo")
    p_cap.add_argument("--usb", action="store_true", help="Use USB webcam (/dev/video0)")
    p_cap.add_argument("--delay", type=float, default=3.0, help="Exposure delay seconds")
    p_cap.add_argument("--width", type=int, default=1280, help="Width")
    p_cap.add_argument("--height", type=int, default=720, help="Height")

    # scan subcommand
    p_scan = subparsers.add_parser("scan", help="Run standalone OCR on an image on the Pi")
    p_scan.add_argument("image", default="test.jpg", nargs="?", help="Remote image filename")
    p_scan.add_argument("--roi", nargs=4, type=float, default=[0.25, 0.40, 0.50, 0.20],
                        metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Crop fractions (X Y WIDTH HEIGHT)")
    p_scan.add_argument("--digits", type=int, default=0, help="Expected digit count")
    p_scan.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate degrees")

    # exec subcommand
    p_exec = subparsers.add_parser("exec", help="Run arbitrary command on Pi via SSH")
    p_exec.add_argument("cmd", help="Shell command to run on Pi")

    args = parser.parse_args()

    if args.subcommand == "status":
        cmd_status(args)
    elif args.subcommand == "trigger":
        cmd_trigger(args)
    elif args.subcommand == "reset":
        cmd_reset(args)
    elif args.subcommand == "mirror":
        cmd_mirror(args)
    elif args.subcommand == "deploy":
        cmd_deploy(args)
    elif args.subcommand == "start":
        cmd_service_ctl("start", args.host)
    elif args.subcommand == "stop":
        cmd_service_ctl("stop", args.host)
    elif args.subcommand == "restart":
        cmd_service_ctl("restart", args.host)
    elif args.subcommand == "logs":
        cmd_logs(args)
    elif args.subcommand == "capture":
        cmd_capture(args)
    elif args.subcommand == "scan":
        cmd_scan(args)
    elif args.subcommand == "exec":
        cmd_run(args.cmd, args.host)
