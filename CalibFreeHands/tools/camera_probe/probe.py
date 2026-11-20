import argparse
import json
import secrets
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parent
CAMERAS = ('A', 'B')


def make_handler(origin):
    viewer_key = secrets.token_urlsafe(18)
    camera_keys = {secrets.token_urlsafe(18): camera for camera in CAMERAS}
    sessions = {camera: {} for camera in CAMERAS}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, payload, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)

        def respond_json(self, status, data):
            self.respond(status, json.dumps(data).encode())

        def signal_target(self, url):
            key = url.path.removeprefix('/signal/')
            if not url.path.startswith('/signal/'):
                return None, False
            if key == viewer_key:
                camera = parse_qs(url.query).get('camera', [''])[0]
                return (camera if camera in CAMERAS else None), True
            return camera_keys.get(key), False

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == f'/view/{viewer_key}' or (
                url.path.startswith('/phone/') and url.path[7:] in camera_keys
            ):
                self.respond(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
                return
            if url.path == '/probe.js':
                self.respond(200, (ROOT / 'probe.js').read_bytes(), 'text/javascript; charset=utf-8')
                return
            camera, viewer = self.signal_target(url)
            if camera is None:
                self.respond_json(404, {})
                return
            with lock:
                session = sessions[camera].copy()
            if viewer:
                self.respond_json(200, session)
            else:
                self.respond_json(200, {key: session[key] for key in ('id', 'answer') if key in session})

        def do_POST(self):
            url = urlsplit(self.path)
            camera, viewer = self.signal_target(url)
            if camera is None:
                self.respond_json(404, {})
                return
            if self.headers.get('Origin') != origin:
                self.respond_json(403, {})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 64 * 1024:
                    raise ValueError
                data = json.loads(self.rfile.read(length))
                call_id = data['id']
                if not isinstance(call_id, str) or len(call_id) != 36:
                    raise ValueError
                if 'diagnostics' in data:
                    diagnostics = data['diagnostics']
                    if viewer or not isinstance(diagnostics, dict) or 'description' in data:
                        raise ValueError
                    with lock:
                        session = sessions[camera]
                        status = 409
                        if session.get('id') == call_id:
                            session['diagnostics'] = diagnostics
                            status = 200
                    self.respond_json(status, {})
                    return
                description = data['description']
                kind = 'answer' if viewer else 'offer'
                if description['type'] != kind or not isinstance(description['sdp'], str):
                    raise ValueError
            except (ValueError, TypeError, KeyError):
                self.respond_json(400, {})
                return
            with lock:
                if not viewer:
                    sessions[camera] = {'id': call_id, 'offer': description}
                    status = 200
                elif sessions[camera].get('id') == call_id:
                    sessions[camera]['answer'] = description
                    status = 200
                else:
                    status = 409
            self.respond_json(status, {})

    return Handler, viewer_key, camera_keys


def main():
    parser = argparse.ArgumentParser(description='CalibFreeHands two-camera WebRTC probe')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', default=8765, type=int)
    parser.add_argument('--origin', help='Browser origin, such as https://camera.example.com:8765')
    parser.add_argument('--cert', help='Trusted HTTPS full chain in PEM format')
    parser.add_argument('--key', help='HTTPS private key in PEM format')
    args = parser.parse_args()
    if bool(args.cert) != bool(args.key):
        parser.error('--cert and --key are required together')
    if args.bind != '127.0.0.1' and not (args.cert and args.origin):
        parser.error('LAN use requires --cert, --key and the matching HTTPS --origin')
    scheme = 'https' if args.cert else 'http'
    origin = (args.origin or f'{scheme}://127.0.0.1:{args.port}').rstrip('/')
    address = urlsplit(origin)
    if not address.hostname or address.path or address.query or address.fragment or (
        address.scheme != 'https' and not (address.scheme == 'http' and address.hostname == '127.0.0.1')
    ):
        parser.error('--origin must be a trusted HTTPS origin or loopback HTTP')
    handler, viewer_key, camera_keys = make_handler(origin)
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        if args.cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(args.cert, args.key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        print(f'PC: {origin}/view/{viewer_key}', flush=True)
        for key, name in camera_keys.items():
            print(f'Camera {name}: {origin}/phone/{key}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
