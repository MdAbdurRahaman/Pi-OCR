import argparse
import sys
import paramiko
from pathlib import Path

import os

HOSTNAME = os.environ.get("PI_HOST", "pizero2.local")
USERNAME = os.environ.get("PI_USER", "stickcam")
PASSWORD = os.environ.get("PI_PASS", "")

def get_ssh_client(password_override=None):
    pwd = password_override or PASSWORD
    if not pwd:
        import getpass
        pwd = getpass.getpass(f"Password for {USERNAME}@{HOSTNAME}: ")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOSTNAME, port=22, username=USERNAME, password=pwd, timeout=8)
        return client
    except Exception as e:
        print(f"[Error] Failed to connect to {USERNAME}@{HOSTNAME}: {e}")
        sys.exit(1)

def cmd_capture(args):
    client = get_ssh_client()
    output_filename = args.output
    remote_path = f"/home/{USERNAME}/seal-scanner/{output_filename}"
    local_path = Path(output_filename)

    print(f"Capturing photo on Pi Zero ({output_filename})...")
    
    # Try capture_seal.py (Picamera2/rpicam-still) or fallback to capture_usb.py
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
        # Find latest jpg in remote seal-scanner
        files = [f for f in sftp.listdir(f"/home/{USERNAME}/seal-scanner") if f.endswith(".jpg")]
        if files:
            files.sort()
            latest = files[-1]
            remote_img = f"/home/{USERNAME}/seal-scanner/{latest}"
            print(f"Downloading {latest} -> {local_path.resolve()}...")
            sftp.get(remote_img, str(local_path))
            print(f"[Success] Photo saved locally as {local_path}")
    except Exception as e:
        print(f"[Warning] Could not download photo: {e}")
    finally:
        sftp.close()
        client.close()

def cmd_scan(args):
    client = get_ssh_client()
    image = args.image
    roi_str = f"--roi {' '.join(str(v) for v in args.roi)}"
    digits_str = f"--digits {args.digits}" if args.digits > 0 else ""
    rotate_str = f"--rotate {args.rotate}" if args.rotate != 0 else ""

    cmd = f"cd ~/seal-scanner && python3 scan_seal.py {image} {roi_str} {digits_str} {rotate_str}"
    print(f"Running OCR on Pi: {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    print(out)
    if err.strip():
        print("[Stderr]:", err)

    # Download debug folder
    stem = Path(image).stem
    debug_remote_dir = f"/home/{USERNAME}/seal-scanner/{stem}_ocr"
    local_debug_dir = Path(f"{stem}_ocr")
    sftp = client.open_sftp()
    try:
        debug_files = sftp.listdir(debug_remote_dir)
        local_debug_dir.mkdir(exist_ok=True)
        for df in debug_files:
            sftp.get(f"{debug_remote_dir}/{df}", str(local_debug_dir / df))
        print(f"Debug crop and binarized images saved to {local_debug_dir.resolve()}/")
    except Exception as e:
        pass
    finally:
        sftp.close()
        client.close()

def cmd_run(cmd_str):
    client = get_ssh_client()
    print(f"Remote command: {cmd_str}")
    stdin, stdout, stderr = client.exec_command(cmd_str)
    print(stdout.read().decode('utf-8', errors='replace'))
    err = stderr.read().decode('utf-8', errors='replace')
    if err.strip():
        print("ERR:", err)
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pi Zero 2 Seal Scanner Remote Controller")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # capture subcommand
    p_cap = subparsers.add_parser("capture", help="Capture image on Pi and download to Windows")
    p_cap.add_argument("output", default="test.jpg", nargs="?", help="Filename for saved photo")
    p_cap.add_argument("--usb", action="store_true", help="Use USB webcam (/dev/video0)")
    p_cap.add_argument("--delay", type=float, default=3.0, help="Exposure delay seconds")
    p_cap.add_argument("--width", type=int, default=1280, help="Width")
    p_cap.add_argument("--height", type=int, default=720, help="Height")

    # scan subcommand
    p_scan = subparsers.add_parser("scan", help="Run OCR on an image on the Pi")
    p_scan.add_argument("image", default="test.jpg", nargs="?", help="Remote image filename")
    p_scan.add_argument("--roi", nargs=4, type=float, default=[0.25, 0.40, 0.50, 0.20],
                        metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Crop fractions (X Y WIDTH HEIGHT)")
    p_scan.add_argument("--digits", type=int, default=0, help="Expected digit count")
    p_scan.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate degrees")

    # exec subcommand
    p_exec = subparsers.add_parser("exec", help="Run arbitrary command on Pi")
    p_exec.add_argument("cmd", help="Shell command to run on Pi")

    args = parser.parse_args()
    if args.subcommand == "capture":
        cmd_capture(args)
    elif args.subcommand == "scan":
        cmd_scan(args)
    elif args.subcommand == "exec":
        cmd_run(args.cmd)
