import pytest

from calibfreehands.visualization.workspace import SOURCE_PANEL_ID, WorkspaceModel


def test_workspace_starts_with_one_owner_per_source_and_result():
    workspace = WorkspaceModel(("cfg-a", "cfg-b", "cfg-c"))

    assert workspace.owner(SOURCE_PANEL_ID).members == (
        SOURCE_PANEL_ID, "result:cfg-a", "result:cfg-b", "result:cfg-c")
    assert workspace.validate()


def test_detach_visible_panels_keeps_hidden_results_attached_and_hidden():
    workspace = WorkspaceModel(("cfg-a", "cfg-b", "cfg-c"))

    detached = workspace.detach_visible(("cfg-a", "cfg-c"))

    assert [workspace.group(group_id).members for group_id in detached] == [
        (SOURCE_PANEL_ID,), ("result:cfg-a",), ("result:cfg-c",)]
    assert workspace.owner("result:cfg-b").host == "main"
    assert workspace.validate()


@pytest.mark.parametrize("source_first", [True, False])
def test_merge_groups_is_order_independent_and_composes_source_with_results(source_first):
    workspace = WorkspaceModel(("cfg-a", "cfg-b", "cfg-c"))
    source_group, result_a, result_b = workspace.detach_visible(("cfg-a", "cfg-b"))

    left, right = ((source_group, result_a) if source_first else (result_a, source_group))
    merged = workspace.merge(left, right)
    merged = workspace.merge(result_b, merged)

    assert workspace.group(merged).members == (
        SOURCE_PANEL_ID, "result:cfg-a", "result:cfg-b")
    assert workspace.composition(merged) == "source-split"
    assert workspace.owner("result:cfg-c").host == "main"
    assert workspace.validate()


def test_three_result_groups_merge_to_overlay_and_extract_without_loss():
    workspace = WorkspaceModel(("cfg-a", "cfg-b", "cfg-c"))
    _, group_a, group_b, group_c = workspace.detach_visible(("cfg-a", "cfg-b", "cfg-c"))

    merged = workspace.merge(group_a, group_b)
    merged = workspace.merge(merged, group_c)
    extracted = workspace.extract("result:cfg-b", merged)

    assert workspace.group(merged).members == ("result:cfg-a", "result:cfg-c")
    assert workspace.group(extracted).members == ("result:cfg-b",)
    assert workspace.composition(merged) == "overlay-3d"
    assert workspace.validate()


def test_reattach_one_all_and_repeated_operations_preserve_panel_identity():
    panel_ids = ("cfg-a", "cfg-b", "cfg-c")
    workspace = WorkspaceModel(panel_ids)
    detached = workspace.detach_visible(panel_ids)
    workspace.reattach(detached[1])
    workspace.reattach_all()

    assert workspace.owner(SOURCE_PANEL_ID).host == "main"
    assert all(workspace.owner(f"result:{panel_id}").host == "main" for panel_id in panel_ids)
    assert workspace.validate()

    with pytest.raises(ValueError):
        workspace.merge(detached[0], detached[0])


def test_invalid_membership_transitions_are_atomic():
    workspace = WorkspaceModel(("cfg-a", "cfg-b"))
    before = workspace.snapshot()

    with pytest.raises(ValueError):
        workspace.detach_panel("result:unknown")

    assert workspace.snapshot() == before
    assert workspace.validate()
