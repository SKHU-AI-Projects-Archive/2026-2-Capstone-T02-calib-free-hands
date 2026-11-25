"""This PC's addresses for the phone publish URL (used by run_pc.py and qr_window.py).

lan_ip in run_config.yaml: an address = use it; "auto" = the connected Wi-Fi adapter's address if there is one
(the phone is on Wi-Fi), else the address on the default route.
"""

import re
import socket
import subprocess
import sys

_WIFI_QUERY = (
    "Get-NetAdapter | Where-Object { $_.Status -eq 'Up' -and $_.PhysicalMediaType -match '802.11' } | "
    "ForEach-Object { (Get-NetIPAddress -InterfaceIndex $_.ifIndex -AddressFamily IPv4 "
    "-ErrorAction SilentlyContinue).IPAddress }"
)
_ADAPTER_QUERY = (
    "Get-NetAdapter | Where-Object { $_.Status -eq 'Up' } | ForEach-Object { $n = $_.Name; $m = $_.PhysicalMediaType; "
    "$d = $_.InterfaceDescription; "
    "Get-NetIPAddress -InterfaceIndex $_.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | "
    "ForEach-Object { \"$n|$m|$d|$($_.IPAddress)\" } }"
)
_IPV4 = re.compile(r"\d{1,3}(\.\d{1,3}){3}")


def wifi_ips():
    """IPv4 addresses of the connected Wi-Fi adapters (Windows; takes about a second), else []."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", _WIFI_QUERY], capture_output=True,
                             text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if _IPV4.fullmatch(line.strip())]


def route_ip(probe_address):
    """The address on the default route, or None. A UDP connect is only a routing lookup, nothing is sent."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((probe_address, 9))
            return s.getsockname()[0]
    except OSError:
        return None


def all_ips():
    """IPv4 addresses of this PC without loopback and link-local ones."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    return [i[4][0] for i in infos if not i[4][0].startswith(("127.", "169.254."))]


def lan_ip(cfg, wifi=None):
    """The address to put into the phone URL (see the module docstring). `wifi`: wifi_ips() if already known."""
    if str(cfg["lan_ip"]).lower() != "auto":
        return str(cfg["lan_ip"])
    wifi = wifi_ips() if wifi is None else wifi
    if wifi:
        return wifi[0]
    return route_ip(cfg["lan_probe_address"]) or socket.gethostbyname(socket.gethostname())


def adapters():
    """[(label, ip)] of the connected adapters (Windows only): "Wi-Fi" (the PC joined a Wi-Fi network), "Hotspot"
    (the PC's own Mobile Hotspot, which Windows lists as a virtual Wi-Fi adapter: the phone joins the PC), "Wired",
    or the adapter's own name."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", _ADAPTER_QUERY], capture_output=True,
                             text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 4 and _IPV4.fullmatch(parts[3]):
            name, media, description, ip = parts
            if "802.11" in media:
                label = "Hotspot" if "direct virtual" in description.lower() else "Wi-Fi"
            else:
                label = "Wired" if media.strip() == "802.3" else name
            found.append((label, ip))
    return found


def choices(cfg):
    """[(label, ip)] for the QR window, the address lan_ip() picks first. Labels are unique."""
    found = adapters()
    names = {ip: label for label, ip in found}
    first = lan_ip(cfg, [ip for label, ip in found if label in ("Wi-Fi", "Hotspot")])  # both are Wi-Fi adapters
    ordered = []
    for ip in [first, *names, route_ip(cfg["lan_probe_address"]), *all_ips()]:
        if ip and ip not in [i for _, i in ordered]:
            label, n = names.get(ip, "This PC"), 1
            while label in [l for l, _ in ordered]:
                n += 1
                label = f"{names.get(ip, 'This PC')} {n}"
            ordered.append((label, ip))
    return ordered


def candidate_ips(cfg):
    """All usable addresses, the one lan_ip() picks first."""
    wifi = wifi_ips()
    ips = []
    for ip in [lan_ip(cfg, wifi), *wifi, route_ip(cfg["lan_probe_address"]), *all_ips()]:
        if ip and ip not in ips:
            ips.append(ip)
    return ips
