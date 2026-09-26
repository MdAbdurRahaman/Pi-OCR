#!/usr/bin/env python3
"""
deploy_hdmi_pi.py - Remote deployment script for 3.5" HDMI Display Scanner to Raspberry Pi.
Configures and launches the systemd service 'seal-hdmi.service' on the Raspberry Pi Zero 2W.
"""
import sys
import time
import socket
import argparse
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


def get_local_pc_ip() -> str:
    """Detects local PC IP address on the network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('192.168.68.1', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.68.123"


DEFAULT_HOSTS = ["pizero2.local", "192.168.68.151", "192.168.68.145", "192.168.68.129"]
DEFAULT_USER = "stickcam"
DEFAULT_PASS = "Dubo2024"

SERVICE_TEMPLATE = """[Unit]
Description=StickCam 3.5" HDMI Display Scanner & Dual-Mode OCR Service
After=network.target

[Service]
Type=simple
User={PI_USER}
WorkingDirectory=/home/{PI_USER}/seal-scanner
ExecStart=/usr/bin/python3 -u /home/{PI_USER}/seal-scanner/pi_hdmi_scanner.py --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""


def deploy():
    parser = argparse.ArgumentParser(description="Deploy 3.5\" HDMI scanner service to Raspberry Pi")
    parser.add_argument("--host", default=None, help="Target Raspberry Pi IP address or hostname")
    parser.add_argument("--user", default=DEFAULT_USER, help=f"SSH username (default: {DEFAULT_USER})")
    parser.add_argument("--password", default=DEFAULT_PASS, help="SSH password")
    parser.add_argument("--pc", default=None, help="PC OCR server URL (optional, e.g. http://192.168.68.123:5000)")
    args = parser.parse_args()

    hosts_to_try = [args.host] if args.host else DEFAULT_HOSTS
    pi_user = args.user
    pi_pass = args.password

    print("=" * 68)
    print("  DEPLOYING 3.5\" HDMI SCANNER TO RASPBERRY PI ZERO 2W")
    print("=" * 68)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connected = False
    connected_host = None

    for h in hosts_to_try:
        print(f"[*] Trying to connect to Raspberry Pi at {h}...")
        try:
            ssh.connect(h, username=pi_user, password=pi_pass, timeout=8)
            print(f"[+] Successfully connected to Pi at {h}!")
            connected = True
            connected_host = h
            break
        except Exception as e:
            print(f"[-] Connection failed to {h}: {e}")

    if not connected:
        print("[-] Could not reach any Raspberry Pi host. Verify WiFi / IP and power.")
        sys.exit(1)

    def run_sudo(cmd: str) -> str:
        stdin, stdout, stderr = ssh.exec_command(f"sudo -S {cmd}", get_pty=True)
        stdin.write(f"{pi_pass}\n")
        stdin.flush()
        return stdout.read().decode()

    # 1. Stop conflicting or old services
    print("[*] Stopping any old streaming or SPI services...")
    run_sudo("systemctl stop seal-stream.service || true")
    run_sudo("systemctl stop seal-burst.service || true")
    run_sudo("systemctl stop seal-hdmi.service || true")
    run_sudo("systemctl disable seal-stream.service || true")
    run_sudo("systemctl disable seal-burst.service || true")

    # 2. Upload Python scripts and assets
    base_dir = Path(__file__).resolve().parent
    files_to_upload = [
        "pi_hdmi_scanner.py",
        "pi_hdmi_ui.py",
        "scan_seal.py",
        "capture_seal.py",
        "run_offline_scanner.py",
        "requirements.txt",
    ]

    print("[*] Preparing target directory /home/" + pi_user + "/seal-scanner...")
    run_sudo(f"mkdir -p /home/{pi_user}/seal-scanner")
    run_sudo(f"chown -R {pi_user}:{pi_user} /home/{pi_user}/seal-scanner")

    sftp = ssh.open_sftp()
    for fname in files_to_upload:
        local_path = base_dir / fname
        if local_path.exists():
            remote_path = f"/home/{pi_user}/seal-scanner/{fname}"
            print(f"[*] Uploading {fname} -> {remote_path}...")
            sftp.put(str(local_path), remote_path)
            sftp.chmod(remote_path, 0o755)

    # 3. Create systemd service
    print("[*] Creating seal-hdmi.service configuration...")
    exec_cmd = f"/usr/bin/python3 -u /home/{pi_user}/seal-scanner/pi_hdmi_scanner.py --port 8000"
    if args.pc:
        exec_cmd += f" --pc {args.pc}"

    service_content = SERVICE_TEMPLATE.format(PI_USER=pi_user).replace(
        f"/usr/bin/python3 -u /home/{pi_user}/seal-scanner/pi_hdmi_scanner.py --port 8000",
        exec_cmd
    )

    remote_service_file = "/tmp/seal-hdmi.service"
    with sftp.file(remote_service_file, "w") as f:
        f.write(service_content)
    sftp.close()

    # 4. Install and enable systemd service
    print("[*] Installing and enabling seal-hdmi.service...")
    run_sudo("mv /tmp/seal-hdmi.service /etc/systemd/system/seal-hdmi.service")
    run_sudo("systemctl daemon-reload")
    run_sudo("systemctl enable seal-hdmi.service")
    run_sudo("systemctl restart seal-hdmi.service")

    time.sleep(2)
    status_out = run_sudo("systemctl status seal-hdmi.service --no-pager")
    print("\n" + "=" * 68)
    print("  DEPLOYMENT STATUS:")
    print("=" * 68)
    safe_print(status_out)
    print("\n[+] 3.5\" HDMI Display scanner is now running on the Raspberry Pi!")
    print(f"[+] Access web mirror at: http://{connected_host}:8000")
    print("=" * 68)

    ssh.close()


if __name__ == "__main__":
    deploy()
