import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from probe import make_handler

origin = "http://127.0.0.1"
handler, viewer_key, camera_keys = make_handler(origin)
server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}/signal/"
phone_a, phone_b = camera_keys
viewer_a = viewer_key + "?camera=A"
viewer_b = viewer_key + "?camera=B"
sid = "1" * 36
second = "2" * 36
offer = {"type": "offer", "sdp": "mock"}
report = {"capture": {"width": 1920, "height": 1080}, "controls": {}}
def exchange(key, body=None, source=origin):
    payload = None if body is None else json.dumps(body).encode()
    try:
        with urlopen(Request(base + key, data=payload, headers={"Origin": source}), timeout=3) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)
try:
    assert exchange(phone_a, {"id": sid, "description": offer})[0] == 200
    assert exchange(phone_b, {"id": sid, "description": offer})[0] == 200
    assert exchange(viewer_a, {"id": sid, "diagnostics": report})[0] == 400
    assert exchange(phone_a, {"id": sid, "diagnostics": report}, "http://different") [0] == 403
    assert exchange(phone_a, {"id": sid, "diagnostics": report})[0] == 200
    assert exchange(viewer_a)[1]["diagnostics"] == report
    assert "diagnostics" not in exchange(viewer_b)[1]
    assert "diagnostics" not in exchange(phone_a)[1]
    assert exchange(phone_a, {"id": second, "description": offer})[0] == 200
    assert exchange(phone_a, {"id": sid, "diagnostics": report})[0] == 409
    assert "diagnostics" not in exchange(viewer_a)[1]
    print("Diagnostics reach only the viewer for the current camera session.")
finally:
    server.shutdown()
    server.server_close()
