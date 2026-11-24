from pathlib import Path

from calibfreehands.app import native_media_dialog


def test_selected_shell_items_use_get_count_and_get_item_at_vtable_slots():
    calls = []
    items = [object(), object()]

    def method(_interface, slot, _result_type, *_argument_types):
        def invoke(_this, *args):
            calls.append(slot)
            if slot == 7:
                args[0]._obj.value = len(items)
                return 0
            if slot == 8:
                args[1]._obj.value = id(items[args[0].value])
                return 0
            raise AssertionError(f"Unexpected IShellItemArray vtable slot: {slot}")
        return invoke

    released = []
    def shell_path(item, _method):
        return f"C:/media/{next(i for i, value in enumerate(items) if id(value) == item.value)}.mp4"

    helper = getattr(native_media_dialog, "_selected_shell_item_paths", None)
    assert callable(helper), "native result enumeration helper is missing"
    paths = helper(
        object(), method, shell_path=shell_path, release=lambda item: released.append(item.value)
    )

    assert paths == [Path("C:/media/0.mp4"), Path("C:/media/1.mp4")]
    assert calls == [7, 8, 8]
    assert released == [id(items[0]), id(items[1])]
