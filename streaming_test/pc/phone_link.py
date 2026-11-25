"""PC -> phone: lock / release the phone's exposure and focus around a recording.

The Android app runs a tiny HTTP server while it streams (RemoteControl.kt); it only answers the PC the
stream goes to. The phone's address is read from the receiver (the SRT connection of the stream path), so
nothing has to be configured; `phone_address` in run_config_manual.yaml overrides it.
"""

import http.client
import json
import logging

import manual_mode

log = logging.getLogger("recorder")


def phone_host(settings):
    """The phone's address, or None if no phone is publishing."""
    override = str(settings.get("phone_address") or "").strip()
    if override:
        return override
    url = f"http://{settings['api_host']}:{settings['api_port']}/v3/srtconns/list"
    try:
        with manual_mode._OPENER.open(url, timeout=settings["api_timeout_s"]) as resp:
            items = json.load(resp)["items"]
    except (OSError, ValueError, KeyError):
        return None
    for item in items:
        if item.get("path") == settings["stream_path"] and item.get("state") == "publish":
            return str(item.get("remoteAddr", "")).rsplit(":", 1)[0].strip("[]") or None
    return None


def set_hold(settings, on):
    """Lock (on=True) or release the phone's exposure/focus.

    Returns True/False = locked/not locked on the phone afterwards, None = not done (switched off, or the phone
    could not be reached: an old app version, a firewall, a Wi-Fi that isolates devices ...). Never raises.
    """
    if not settings["lock_on_record"]:
        return None
    host = phone_host(settings)
    if not host:
        log.warning("phone lock: no phone address")
        return None
    conn = http.client.HTTPConnection(host, settings["phone_port"], timeout=settings["phone_timeout_s"])
    try:
        conn.request("POST", f"/hold?on={int(on)}")
        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            log.warning("phone lock: the phone answered %s", resp.status)
            return None
        return bool(json.loads(body).get("locked"))
    except (OSError, ValueError, http.client.HTTPException) as e:
        log.warning("phone lock: not reachable (%s)", e)
        return None
    finally:
        conn.close()
