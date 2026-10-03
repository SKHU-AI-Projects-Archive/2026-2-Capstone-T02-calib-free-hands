"""
GigaHands manifest를 이용해 Train / Validation / Test split을 생성합니다.

같은 participant가 서로 다른 split에 들어가지 않도록 participant 단위로 나누고,
245개의 unique sequence-camera 기준으로 fx와 fy 분포를 비교합니다.
focal 분포 점수가 같은 후보가 있으면 Train-Test exact calibration overlap이
더 작은 구성을 우선합니다. 이 split은 01~05 실험에서 공통으로 사용하는 확정 split입니다.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path


HANDCALIB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = HANDCALIB_ROOT / "data" / "manifests" / "gigahands.csv"
DEFAULT_OUTPUT = HANDCALIB_ROOT / "data" / "splits" / "gigahands"
N_BINS = 5
LOCKED_SPLIT = {"train": ("p41", "p44"), "val": ("p36",), "test": ("p52",)}
CALIBRATION_FIELDS = (
    "width", "height", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"
)


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unique_conditions(rows):
    return list({row["camera_key"]: row for row in rows}.values())


def calibration_signature(row):
    return tuple(int(row[field]) if field in ("width", "height") else float(row[field]) for field in CALIBRATION_FIELDS)


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def stats(rows, field):
    values = sorted(float(row[field]) for row in rows)
    return {
        "min": values[0],
        "max": values[-1],
        "mean": sum(values) / len(values),
        "median": quantile(values, 0.5),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
    }


def bin_edges(values):
    low, high = min(values), max(values)
    width = (high - low) / N_BINS
    return [low + width * index for index in range(N_BINS)] + [high]


def histogram(values, edges):
    counts = [0] * N_BINS
    for value in values:
        index = next((i for i in range(N_BINS) if value < edges[i + 1]), N_BINS - 1)
        counts[index] += 1
    total = len(values)
    return {"counts": counts, "proportions": [count / total for count in counts]}


def participant_stats(rows):
    result = {}
    for participant in sorted({row["participant"] for row in rows}):
        participant_rows = [row for row in rows if row["participant"] == participant]
        conditions = unique_conditions(participant_rows)
        result[participant] = {
            "sequence_count": len({row["sequence"] for row in participant_rows}),
            "camera_count": len(conditions),
            "video_count": len(participant_rows),
            "fx": stats(conditions, "fx"),
            "fy": stats(conditions, "fy"),
        }
    return result


def participant_calibration_sets(rows):
    return {
        participant: {calibration_signature(row) for row in unique_conditions(rows) if row["participant"] == participant}
        for participant in sorted({row["participant"] for row in rows})
    }


def candidate_record(rows, train, validation, test, global_rows, fx_edges, fy_edges):
    groups = {"train": set(train), "validation": set(validation), "test": set(test)}
    participants = groups["train"] | groups["validation"] | groups["test"]
    by_participant = {
        participant: [row for row in rows if row["participant"] == participant]
        for participant in participants
    }
    conditions = {}
    group_rows = {}
    for name, group in groups.items():
        source = [row for participant in group for row in by_participant[participant]]
        group_rows[name] = source
        conditions[name] = unique_conditions(source)

    global_conditions = unique_conditions(global_rows)
    calibration_sets = participant_calibration_sets(rows)
    group_calibration_sets = {
        name: set().union(*(calibration_sets[participant] for participant in group))
        for name, group in groups.items()
    }
    histograms = {}
    distances = []
    for field, edges in (("fx", fx_edges), ("fy", fy_edges)):
        global_hist = histogram([float(row[field]) for row in global_conditions], edges)
        histograms[field] = {"global": global_hist}
        for name, group in conditions.items():
            current = histogram([float(row[field]) for row in group], edges)
            distance = sum(
                abs(left - right)
                for left, right in zip(current["proportions"], global_hist["proportions"])
            ) / N_BINS
            histograms[field][name] = {**current, "distance": distance}
            distances.append(distance)

    return {
        "train_participants": sorted(train),
        "validation_participant": next(iter(validation)),
        "test_participant": next(iter(test)),
        "sequence_counts": {
            name: len({row["sequence"] for row in group})
            for name, group in group_rows.items()
        },
        "camera_counts": {name: len(group) for name, group in conditions.items()},
        "video_counts": {name: len(group) for name, group in group_rows.items()},
        "frame_counts": {name: sum(int(row["frame_count"]) for row in group) for name, group in group_rows.items()},
        "fx": {name: stats(group, "fx") for name, group in conditions.items()},
        "fy": {name: stats(group, "fy") for name, group in conditions.items()},
        "histogram": histograms,
        "calibration_overlap": {
            "train_validation": len(group_calibration_sets["train"] & group_calibration_sets["validation"]),
            "train_test": len(group_calibration_sets["train"] & group_calibration_sets["test"]),
            "validation_test": len(group_calibration_sets["validation"] & group_calibration_sets["test"]),
        },
        "score": sum(distances) / len(distances),
    }


def candidate_sort_key(candidate, global_stats):
    coverage = 0.0
    for field in ("fx", "fy"):
        span = global_stats[field]["max"] - global_stats[field]["min"]
        for name in ("validation", "test"):
            coverage += (
                candidate[field][name]["max"] - candidate[field][name]["min"]
            ) / span
    names = tuple(
        candidate["train_participants"]
        + [candidate["validation_participant"], candidate["test_participant"]]
    )
    return (round(candidate["score"], 12), -coverage, names)


def select_candidate(candidates, global_stats):
    best_score = min(candidate["score"] for candidate in candidates)
    tied = [candidate for candidate in candidates if math.isclose(candidate["score"], best_score, rel_tol=0.0, abs_tol=1e-12)]
    tied.sort(key=lambda candidate: (
        candidate["calibration_overlap"]["train_test"],
        candidate_sort_key(candidate, global_stats)[1:]
    ))
    return tied[0]


def build_report(rows):
    conditions = unique_conditions(rows)
    global_stats = {
        "participant_count": len({row["participant"] for row in rows}),
        "sequence_count": len({row["sequence"] for row in rows}),
        "camera_count": len(conditions),
        "video_count": len(rows),
        "fx": stats(conditions, "fx"),
        "fy": stats(conditions, "fy"),
    }
    fx_edges = bin_edges([float(row["fx"]) for row in conditions])
    fy_edges = bin_edges([float(row["fy"]) for row in conditions])
    participants = sorted({row["participant"] for row in rows})
    candidates = []
    for train in itertools.combinations(participants, 2):
        remaining = sorted(set(participants) - set(train))
        for validation, test in itertools.permutations(remaining):
            candidate = candidate_record(
                rows, train, {validation}, {test}, rows, fx_edges, fy_edges
            )
            if (
                candidate["sequence_counts"]["train"] > candidate["sequence_counts"]["validation"]
                and candidate["sequence_counts"]["train"] > candidate["sequence_counts"]["test"]
            ):
                candidates.append(candidate)
    candidates.sort(key=lambda item: candidate_sort_key(item, global_stats))
    selected = select_candidate(candidates, global_stats)
    participant_sets = participant_calibration_sets(rows)
    participants = sorted(participant_sets)
    pair_overlap = {
        f"{left}-{right}": len(participant_sets[left] & participant_sets[right])
        for left, right in itertools.combinations(participants, 2)
    }
    total_frames = sum(int(row["frame_count"]) for row in rows)
    return {
        "status": "locked",
        "calibration_signature_fields": list(CALIBRATION_FIELDS),
        "global_stats": global_stats,
        "participant_stats": participant_stats(rows),
        "participant_calibration_overlap": pair_overlap,
        "histogram": {
            "fx_bin_edges": fx_edges,
            "fy_bin_edges": fy_edges,
            "global": selected["histogram"],
        },
        "candidates": candidates,
        "selected_candidate": selected,
        "frame_counts": {"total": total_frames, **selected["frame_counts"]},
        "frame_proportions": {name: selected["frame_counts"][name] / total_frames for name in ("train", "validation", "test")},
        "selection_reason": (
            f"Train={','.join(selected['train_participants'])} 조합의 fx/fy histogram score가 "
            f"가장 낮았습니다. Validation/Test 교환 후보의 score가 동일하여, "
            f"Train-Test exact calibration overlap이 더 작은 {selected['test_participant']}를 "
            "Test로 선택했습니다."
        ),
        "participant_overlap": {
            "train_validation": [],
            "train_test": [],
            "validation_test": [],
        },
    }


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    selected = report["selected_candidate"]
    lines = [
        "# GigaHands Split 분석",
        "",
        "## 전체 데이터",
        "",
        f"- participant: {report['global_stats']['participant_count']}",
        f"- sequence: {report['global_stats']['sequence_count']}",
        f"- camera condition: {report['global_stats']['camera_count']}",
        f"- video: {report['global_stats']['video_count']}",
        f"- 상태: {report['status']}",
        f"- frame: {report['frame_counts']['total']}",
        "",
        "## 전체 Focal 분포",
        "",
    ]
    for field in ("fx", "fy"):
        value = report["global_stats"][field]
        lines.append(
            f"- {field}: min={value['min']:.3f}, max={value['max']:.3f}, "
            f"mean={value['mean']:.3f}, median={value['median']:.3f}, "
            f"Q25={value['q25']:.3f}, Q75={value['q75']:.3f}"
        )
    lines += [
        "",
        "## Participant별 분포",
        "",
        "| Participant | Sequence | Camera | Video | fx median | fy median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for participant, value in report["participant_stats"].items():
        lines.append(
            f"| {participant} | {value['sequence_count']} | "
            f"{value['camera_count']} | {value['video_count']} | "
            f"{value['fx']['median']:.3f} | {value['fy']['median']:.3f} |"
        )
    lines += [
        "", "## Calibration 중복 확인", "",
        "camera name이 아닌 10개 calibration 값의 exact signature가 겹치는 개수입니다.",
        "", "| Participant A | Participant B | 동일 Calibration |", "|---|---|---:|",
    ]
    for pair, count in report["participant_calibration_overlap"].items():
        left, right = pair.split("-", 1)
        lines.append(f"| {left} | {right} | {count} |")
    lines += [
        "", "## Frame 수", "",
        f"- 전체: {report['frame_counts']['total']}",
        f"- Train: {report['frame_counts']['train']} ({report['frame_proportions']['train']:.4%})",
        f"- Validation: {report['frame_counts']['validation']} ({report['frame_proportions']['validation']:.4%})",
        f"- Test: {report['frame_counts']['test']} ({report['frame_proportions']['test']:.4%})",
        "",
        "Camera condition 기준 비율과 실제 frame 비율은 영상 길이가 달라 다를 수 있습니다. "
        "이는 split 오류가 아니며, 학습 시 frame sampling에서 별도로 다룹니다.",
    ]
    lines += [
        "",
        "## Split 기준",
        "",
        "- 같은 participant를 서로 다른 split에 넣지 않습니다.",
        "- frame/video random split을 사용하지 않습니다.",
        "- 245개의 unique sequence-camera 기준으로 focal 분포를 비교합니다.",
        "- fx와 fy를 따로 비교하고 5-bin histogram을 사용합니다.",
        "",
        "## 분포 비교 수식",
        "",
        "D(S) = mean(|p_S - g|)이며, Train/Validation/Test와 fx/fy의 "
        "6개 D 값 평균을 후보 score로 사용합니다.",
        "", "## Histogram 비교", "",
    ]
    for field in ("fx", "fy"):
        field_edges = report["histogram"][f"{field}_bin_edges"]
        field_histograms = report["histogram"]["global"][field]
        lines += [
            f"### {field}", "",
            "| Bin | Global | Train | Validation | Test |",
            "|---|---:|---:|---:|---:|",
        ]
        for index in range(N_BINS):
            proportions = [
                field_histograms[name]["proportions"][index]
                for name in ("global", "train", "validation", "test")
            ]
            lines.append(
                f"| {field_edges[index]:.3f} - {field_edges[index + 1]:.3f} | "
                + " | ".join(f"{value:.2%}" for value in proportions) + " |"
            )
    lines += [
        "",
        "## 후보 비교",
        "",
        "| Train | Validation | Test | Train cameras | Val cameras | Test cameras | Score |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        lines.append(
            f"| {','.join(candidate['train_participants'])} | "
            f"{candidate['validation_participant']} | {candidate['test_participant']} | "
            f"{candidate['camera_counts']['train']} | "
            f"{candidate['camera_counts']['validation']} | "
            f"{candidate['camera_counts']['test']} | {candidate['score']:.8f} |"
        )
    lines += [
        "",
        "## 확정 Split",
        "",
        f"- Train: {', '.join(selected['train_participants'])}",
        f"- Validation: {selected['validation_participant']}",
        f"- Test: {selected['test_participant']}",
        "",
        "## 선택 이유",
        "",
        report["selection_reason"],
        f"histogram score={selected['score']:.8f}, participant overlap=0, "
        f"Train-Test calibration overlap={selected['calibration_overlap']['train_test']}입니다.",
        "",
        "## 중요한 한계",
        "",
        "participant는 Train / Validation / Test 사이에 겹치지 않습니다. "
        "다만 GigaHands demo에서는 같은 camera identifier가 여러 sequence에 반복됩니다. "
        "따라서 이 split은 unseen-camera 평가를 의미하지 않습니다.",
        "",
        "현재 demo는 participant 수가 적기 때문에 정확한 80/10/10보다 "
        "participant leakage 방지를 우선했습니다.",
        "Validation의 p36은 Train의 p41과 동일한 calibration parameter가 반복될 수 있어 "
        "unseen-calibration 평가가 아닙니다. Test는 Train과 exact calibration overlap이 없는 구성을 우선했습니다.",
        "",
    ]
    (output / "split_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build the locked participant-level GigaHands split.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if len(rows) != 248 or len({row["camera_key"] for row in rows}) != 245:
        raise SystemExit("unexpected manifest counts")
    report = build_report(rows)
    selected = report["selected_candidate"]
    selected_split = {
        "train": tuple(selected["train_participants"]),
        "val": (selected["validation_participant"],),
        "test": (selected["test_participant"],),
    }
    if selected_split != LOCKED_SPLIT:
        raise RuntimeError(f"calculated split differs from locked split: {selected_split}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, participants in (
        ("train", selected["train_participants"]),
        ("val", [selected["validation_participant"]]),
        ("test", [selected["test_participant"]]),
    ):
        (args.output_dir / f"{name}.txt").write_text(
            "\n".join(participants) + "\n", encoding="utf-8"
        )
    write_report(report, args.output_dir)
    print(
        f"status={report['status']} candidates={len(report['candidates'])} "
        f"score={selected['score']:.8f}"
    )
    print(
        f"train={','.join(selected['train_participants'])} "
        f"val={selected['validation_participant']} "
        f"test={selected['test_participant']}"
    )


if __name__ == "__main__":
    main()
