"""MANO pkl 의 chumpy 배열을 순수 numpy 로 변환 (1회 실행).

MPI 배포본 MANO_RIGHT.pkl 은 chumpy 객체를 포함하는데, chumpy 는 최신
pip/numpy 에서 설치 자체가 깨진다. smplx 로더는 배열 값만 쓰므로 chumpy 없이
스텁 언피클로 값을 뽑아 chumpy-free pkl 로 재저장한다.

사용: python tools/strip_chumpy.py mano_data/MANO_RIGHT.pkl
  → 원본을 <이름>.chumpy.bak 으로 백업하고 같은 경로에 변환본을 쓴다.
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


class _ChumpyStub:
    """chumpy.Ch 대역 — pickle 이 넣어주는 상태 dict 만 보관."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyStub
        return super().find_class(module, name)


def _to_numpy(value):
    if isinstance(value, _ChumpyStub):
        d = value.__dict__
        # chumpy.reordering.Select 노드: a(원배열).flat[idxs] → preferred_shape
        # (MANO pkl 의 shapedirs 가 이 형태)
        if "idxs" in d and "a" in d:
            base = np.asarray(_to_numpy(d["a"])).ravel()
            out = base[np.asarray(d["idxs"], dtype=np.int64)]
            shape = d.get("preferred_shape")
            return out.reshape(shape) if shape else out
        # 일반 chumpy Ch: 실값은 'x'
        if "x" in d:
            return np.asarray(_to_numpy(d["x"]))
        arrays = [v for v in d.values() if isinstance(v, np.ndarray)]
        if len(arrays) == 1:
            return arrays[0]
        raise ValueError(f"chumpy 스텁에서 배열을 찾지 못함: keys={list(d)}")
    return value


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    pkl_path = Path(sys.argv[1])

    with open(pkl_path, "rb") as f:
        data = _Unpickler(f, encoding="latin1").load()

    converted = {}
    for key, value in data.items():
        out = _to_numpy(value)
        converted[key] = out
        kind = type(value).__name__
        shape = getattr(out, "shape", None)
        print(f"  {key:20s} {kind:12s} -> {type(out).__name__} {shape}")

    backup = pkl_path.with_suffix(pkl_path.suffix + ".chumpy.bak")
    if not backup.exists():
        shutil.copy2(pkl_path, backup)
        print(f"백업: {backup.name}")

    with open(pkl_path, "wb") as f:
        pickle.dump(converted, f, protocol=2)
    print(f"변환 완료: {pkl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
