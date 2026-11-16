"""Read static QuickTime/MP4 metadata (moov meta keys + ilst) without exiftool.

Location keys (location, GPS, ISO6709) are never returned.
iPhone videos carry, among others:
    com.apple.quicktime.model                              e.g. "iPhone 17 Pro"
    com.apple.quicktime.creationdate                       local time with offset
    com.apple.quicktime.camera.lens_model                  "... back triple camera 6.765mm f/1.78"
    com.apple.quicktime.camera.focal_length.35mm_equivalent "24"
    com.apple.quicktime.cinematic-video-intent             "1" for cinematic mode
"""
import struct

SKIP = ("location", "gps", "iso6709")


def _boxes(buf, off=0, end=None):
    end = len(buf) if end is None else end
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[off:off + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[off + 8:off + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - off
        if size < hdr:
            return
        yield typ, off + hdr, off + size
        off += size


def _top_box(path, want):
    with open(path, "rb") as fh:
        while True:
            h = fh.read(8)
            if len(h) < 8:
                return b""
            size, typ = struct.unpack(">I4s", h)
            hdr = 8
            if size == 1:
                size = struct.unpack(">Q", fh.read(8))[0]
                hdr = 16
            elif size == 0:
                return fh.read() if typ == want else b""
            if typ == want:
                return fh.read(size - hdr)
            fh.seek(size - hdr, 1)


def _value(dtype, raw):
    if dtype == 1:
        return raw.decode("utf-8", "replace")
    if dtype == 23:
        return struct.unpack(">f", raw[:4])[0]
    if dtype == 24:
        return struct.unpack(">d", raw[:8])[0]
    if dtype in (21, 65, 66, 67, 74):
        return int.from_bytes(raw, "big", signed=True)
    if dtype in (22, 75, 76, 77, 78):
        return int.from_bytes(raw, "big", signed=False)
    return None


def _metas(buf, start, end):
    out = {}
    for typ, a, b in _boxes(buf, start, end):
        if typ in (b"trak", b"udta", b"mdia", b"minf"):
            out.update(_metas(buf, a, b))
        elif typ == b"meta":
            keys, items = [], {}
            for t2, a2, b2 in _boxes(buf, a, b):
                if t2 == b"keys":
                    n = struct.unpack(">I", buf[a2 + 4:a2 + 8])[0]
                    p = a2 + 8
                    for _ in range(n):
                        sz = struct.unpack(">I", buf[p:p + 4])[0]
                        keys.append(buf[p + 8:p + sz].decode("utf-8", "replace"))
                        p += sz
                elif t2 == b"ilst":
                    for t3, a3, b3 in _boxes(buf, a2, b2):
                        idx = struct.unpack(">I", t3)[0]
                        for t4, a4, b4 in _boxes(buf, a3, b3):
                            if t4 == b"data":
                                dtype = struct.unpack(">I", buf[a4:a4 + 4])[0] & 0xFFFFFF
                                items[idx] = _value(dtype, buf[a4 + 8:b4])
            for idx, v in items.items():
                if 1 <= idx <= len(keys) and v is not None:
                    k = keys[idx - 1]
                    if not any(s in k.lower() for s in SKIP):
                        out[k] = v
    return out


def read(path):
    """Dict of metadata key -> value (location keys removed)."""
    moov = _top_box(path, b"moov")
    return _metas(moov, 0, len(moov)) if moov else {}
