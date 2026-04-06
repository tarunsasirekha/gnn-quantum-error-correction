"""
CPU & Power Monitor for GNN Training
Logs CPU%, memory, and power draw every 60 seconds
"""

import psutil
import subprocess
import time
import csv
import os
from datetime import datetime
from pathlib import Path

LOG_FILE = "monitor_log.csv"
PIDS_TO_WATCH = [3475]   # attention run; residual PID added automatically
INTERVAL = 60            # seconds between samples


def get_power_watts():
    """Read battery power draw from ioreg (macOS)"""
    try:
        out = subprocess.check_output(
            ["ioreg", "-rn", "AppleSmartBattery"], text=True
        )
        voltage_mv, amperage_ma = None, None
        for line in out.splitlines():
            if '"Voltage"' in line and "AppleRaw" not in line:
                voltage_mv = int(line.split("=")[-1].strip())
            if '"InstantAmperage"' in line:
                raw = int(line.split("=")[-1].strip())
                # stored as unsigned 64-bit; convert to signed
                if raw > 2**63:
                    raw -= 2**64
                amperage_ma = raw
        if voltage_mv and amperage_ma:
            watts = abs(amperage_ma * voltage_mv) / 1_000_000
            direction = "discharging" if amperage_ma < 0 else "charging"
            return round(watts, 2), direction
    except Exception:
        pass
    return None, "unknown"


def get_process_stats(pid):
    """Get CPU% and memory for a PID"""
    try:
        p = psutil.Process(pid)
        cpu = p.cpu_percent(interval=1)
        mem_mb = p.memory_info().rss / (1024 * 1024)
        status = p.status()
        return cpu, round(mem_mb, 1), status
    except psutil.NoSuchProcess:
        return None, None, "dead"


def find_residual_pid():
    """Look for the residual training process if it started"""
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = " ".join(p.info['cmdline'] or [])
            if "main.py" in cmd and "residual" in cmd:
                return p.info['pid']
        except Exception:
            pass
    return None


def init_log():
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "run", "pid", "pid_status",
                "cpu_pct", "memory_mb",
                "system_cpu_pct", "power_watts", "power_direction"
            ])


def log_row(run_name, pid, pid_status, cpu, mem, sys_cpu, watts, direction):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_name, pid, pid_status,
            cpu, mem, sys_cpu, watts, direction
        ])


def print_status(run, pid, status, cpu, mem, sys_cpu, watts, direction):
    now = datetime.now().strftime("%H:%M:%S")
    power_str = f"{watts}W ({direction})" if watts else "N/A"
    print(f"[{now}] {run} | PID {pid} ({status}) | "
          f"Process CPU: {cpu}% | Mem: {mem} MB | "
          f"System CPU: {sys_cpu}% | Power: {power_str}")


def main():
    init_log()
    known_pids = {3475: "attention_L6"}
    residual_found = False

    print("=" * 70)
    print("GNN TRAINING MONITOR — CPU & POWER")
    print(f"Logging every {INTERVAL}s to {LOG_FILE}")
    print("=" * 70)

    while True:
        # Check if residual run has started
        if not residual_found:
            rpid = find_residual_pid()
            if rpid:
                known_pids[rpid] = "residual_L6"
                residual_found = True
                print(f"\n>>> Residual run detected! PID {rpid} added to monitor\n")

        watts, direction = get_power_watts()
        sys_cpu = psutil.cpu_percent(interval=None)

        all_dead = True
        for pid, run_name in list(known_pids.items()):
            cpu, mem, status = get_process_stats(pid)

            if status != "dead":
                all_dead = False

            cpu_display  = cpu  if cpu  is not None else "—"
            mem_display  = mem  if mem  is not None else "—"

            print_status(run_name, pid, status, cpu_display, mem_display,
                         sys_cpu, watts, direction)
            log_row(run_name, pid, status, cpu_display, mem_display,
                    sys_cpu, watts, direction)

        if all_dead and residual_found:
            print("\nBoth runs complete. Monitor exiting.")
            break

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
