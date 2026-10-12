from calibfreehands.pipeline.provenance import runtime_metadata


def test_runtime_metadata_contains_python_and_platform():
    metadata = runtime_metadata()
    assert metadata["python"]
    assert metadata["platform"]
