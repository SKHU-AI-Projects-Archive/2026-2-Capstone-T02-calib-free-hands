"""Logical panel ownership for a Comparison Viewer workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SOURCE_PANEL_ID = "source-media"
MAIN_GROUP_ID = "main"
WorkspaceHost = Literal["main", "detached"]


@dataclass(frozen=True)
class WorkspaceGroup:
    group_id: str
    members: tuple[str, ...]
    host: WorkspaceHost

    @property
    def source_panel_id(self) -> str | None:
        return SOURCE_PANEL_ID if SOURCE_PANEL_ID in self.members else None

    @property
    def result_panel_ids(self) -> tuple[str, ...]:
        return tuple(member for member in self.members if member.startswith("result:"))


class WorkspaceModel:
    """Tracks each source/result panel in exactly one main or detached group."""

    def __init__(self, configuration_ids: tuple[str, ...] | list[str]) -> None:
        if len(configuration_ids) != len(set(configuration_ids)):
            raise ValueError("configuration IDs must be unique")
        self._configuration_ids = tuple(configuration_ids)
        self._panel_order = (SOURCE_PANEL_ID, *(f"result:{item}" for item in configuration_ids))
        self._groups: dict[str, WorkspaceGroup] = {
            MAIN_GROUP_ID: WorkspaceGroup(MAIN_GROUP_ID, self._panel_order, "main")
        }
        self._next_group_number = 1

    def snapshot(self) -> tuple[WorkspaceGroup, ...]:
        return tuple(self._groups.values())

    def group(self, group_id: str) -> WorkspaceGroup:
        try:
            return self._groups[group_id]
        except KeyError as exc:
            raise ValueError(f"unknown workspace group: {group_id}") from exc

    def owner(self, panel_id: str) -> WorkspaceGroup:
        self._validate_panel(panel_id)
        return next(group for group in self._groups.values() if panel_id in group.members)

    def groups(self, *, host: WorkspaceHost | None = None) -> tuple[WorkspaceGroup, ...]:
        return tuple(group for group in self._groups.values() if host is None or group.host == host)

    def composition(self, group_id: str) -> str:
        group = self.group(group_id)
        if group.source_panel_id is not None:
            return "source-split" if group.result_panel_ids else "source-only"
        return "overlay-3d"

    def detach_visible(self, visible_configuration_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Detach Source Media and visible results; hidden results stay in the main group."""
        visible = set(visible_configuration_ids)
        unknown = visible.difference(self._configuration_ids)
        if unknown:
            raise ValueError(f"unknown configuration IDs: {sorted(unknown)}")
        result = []
        for panel_id in self._panel_order:
            if panel_id == SOURCE_PANEL_ID or panel_id.removeprefix("result:") in visible:
                result.append(self.detach_panel(panel_id))
        return tuple(result)

    def detach_panel(self, panel_id: str) -> str:
        self._validate_panel(panel_id)
        current = self.owner(panel_id)
        if current.host == "detached" and current.members == (panel_id,):
            return current.group_id
        group_id = self._new_group_id()
        remaining = tuple(member for member in current.members if member != panel_id)
        self._replace_group(current, remaining)
        self._groups[group_id] = WorkspaceGroup(group_id, (panel_id,), "detached")
        self._validate_invariants()
        return group_id

    def extract(self, panel_id: str, group_id: str) -> str:
        self._validate_panel(panel_id)
        current = self.group(group_id)
        if panel_id not in current.members:
            raise ValueError(f"panel {panel_id} is not in group {group_id}")
        if current.members == (panel_id,):
            return group_id
        new_group_id = self._new_group_id()
        self._replace_group(current, tuple(member for member in current.members if member != panel_id))
        self._groups[new_group_id] = WorkspaceGroup(new_group_id, (panel_id,), "detached")
        self._validate_invariants()
        return new_group_id

    def merge(self, source_group_id: str, target_group_id: str) -> str:
        if source_group_id == target_group_id:
            raise ValueError("a workspace group cannot be merged into itself")
        source = self.group(source_group_id)
        target = self.group(target_group_id)
        members = self._ordered_members((*target.members, *source.members))
        del self._groups[source_group_id]
        self._groups[target_group_id] = WorkspaceGroup(target_group_id, members, target.host)
        self._validate_invariants()
        return target_group_id

    def reorder_results(self, group_id: str, result_panel_ids: tuple[str, ...] | list[str]) -> None:
        group = self.group(group_id)
        expected = set(group.result_panel_ids)
        if len(result_panel_ids) != len(expected) or set(result_panel_ids) != expected:
            raise ValueError("reordered result IDs must match the group's current results")
        members = ([SOURCE_PANEL_ID] if group.source_panel_id else []) + list(result_panel_ids)
        self._groups[group_id] = WorkspaceGroup(group_id, tuple(members), group.host)
        self._validate_invariants()

    def reattach(self, group_id: str) -> str:
        group = self.group(group_id)
        if group.host == "main":
            return group_id
        main = next((candidate for candidate in self._groups.values() if candidate.host == "main"), None)
        if main is None:
            self._groups[group_id] = WorkspaceGroup(group_id, group.members, "main")
            main_id = group_id
        else:
            main_id = main.group_id
            self._groups[main_id] = WorkspaceGroup(
                main_id, self._ordered_members((*main.members, *group.members)), "main")
            del self._groups[group_id]
        self._validate_invariants()
        return main_id

    def reattach_all(self) -> str:
        groups = tuple(self._groups.values())
        main = next((group for group in groups if group.host == "main"), None)
        if main is None:
            main_id = MAIN_GROUP_ID
            members: tuple[str, ...] = ()
        else:
            main_id = main.group_id
            members = main.members
        for group in groups:
            if group.host == "detached":
                members = (*members, *group.members)
        self._groups = {main_id: WorkspaceGroup(main_id, self._ordered_members(members), "main")}
        self._validate_invariants()
        return main_id

    def validate(self) -> bool:
        self._validate_invariants()
        return True

    def _replace_group(self, group: WorkspaceGroup, members: tuple[str, ...]) -> None:
        if members:
            self._groups[group.group_id] = WorkspaceGroup(
                group.group_id, self._ordered_members(members), group.host)
        else:
            del self._groups[group.group_id]

    def _ordered_members(self, members: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        unique = set(members)
        return tuple(panel_id for panel_id in self._panel_order if panel_id in unique)

    def _new_group_id(self) -> str:
        while True:
            group_id = f"detached-{self._next_group_number}"
            self._next_group_number += 1
            if group_id not in self._groups:
                return group_id

    def _validate_panel(self, panel_id: str) -> None:
        if panel_id != SOURCE_PANEL_ID and panel_id not in {
            f"result:{configuration_id}" for configuration_id in self._configuration_ids
        }:
            raise ValueError(f"unknown workspace panel: {panel_id}")

    def _validate_invariants(self) -> None:
        members = [panel_id for group in self._groups.values() for panel_id in group.members]
        if len(members) != len(set(members)):
            raise RuntimeError("workspace panel has more than one owner")
        if set(members) != set(self._panel_order):
            raise RuntimeError("workspace is missing or contains an unknown panel")
        if sum(group.source_panel_id is not None for group in self._groups.values()) != 1:
            raise RuntimeError("workspace must contain exactly one Source Media panel")
        if any(not group.members for group in self._groups.values()):
            raise RuntimeError("empty workspace groups are not allowed")
