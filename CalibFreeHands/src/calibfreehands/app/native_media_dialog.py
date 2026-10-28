from __future__ import annotations

import ctypes
import os
from pathlib import Path

from calibfreehands.media.discovery import SUPPORTED_EXTENSIONS


_CLSID_FILE_OPEN_DIALOG = "DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7"
_IID_FILE_OPEN_DIALOG = "D57C7288-D4AD-4768-BE02-9D969532D960"
_IID_FILE_DIALOG_CUSTOMIZE = "E6FDD21A-163F-4975-9C8C-A69F1BA37034"
_IID_FILE_DIALOG_EVENTS = "973510DB-7D7F-452B-8975-74A85828D354"
_IID_FILE_DIALOG_CONTROL_EVENTS = "36116642-D713-4B97-9B83-7484A9D00433"
_IID_SHELL_ITEM = "43826D1E-E718-42EE-BC55-A1E261C37BFE"
_CUSTOM_FOLDER_BUTTON = 0xCF01


class NativeMediaDialogError(RuntimeError):
    pass


def select_media_paths(parent, start_directory: Path) -> list[Path]:
    if os.name != "nt":
        raise NativeMediaDialogError("The Windows Explorer media dialog is available only on Windows.")
    return _select_windows_media_paths(parent, Path(start_directory))


def _select_windows_media_paths(parent, start_directory: Path) -> list[Path]:
    """Use the Windows Common Item Dialog with a native current-folder action."""
    if not hasattr(ctypes, "WINFUNCTYPE"):
        raise NativeMediaDialogError("This Python runtime cannot call Windows COM interfaces.")

    HRESULT = ctypes.c_long
    ULONG = ctypes.c_ulong
    DWORD = ctypes.c_uint32
    PVOID = ctypes.c_void_p
    WCHAR_P = ctypes.c_wchar_p
    WINFUNCTYPE = ctypes.WINFUNCTYPE

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                    ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

        @classmethod
        def parse(cls, value: str):
            import uuid
            raw = uuid.UUID(value).bytes_le
            return cls.from_buffer_copy(raw)

    def check_hresult(hr, operation: str) -> None:
        if hr < 0:
            raise OSError(f"{operation} failed (HRESULT 0x{hr & 0xFFFFFFFF:08X})")

    def method(interface, index, result_type, *argument_types):
        vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(PVOID))).contents
        return WINFUNCTYPE(result_type, PVOID, *argument_types)(vtable[index])

    class EventSink:
        def __init__(self):
            self.dialog = None
            self.folder_selected = False
            self.references = 1
            self.events = EventInterface()
            self.controls = ControlInterface()
            self.events.lpVtbl = ctypes.pointer(_EVENT_VTABLE)
            self.events.owner = self
            self.controls.lpVtbl = ctypes.pointer(_CONTROL_VTABLE)
            self.controls.owner = self

    class EventVTable(ctypes.Structure):
        pass

    class ControlVTable(ctypes.Structure):
        pass

    class EventInterface(ctypes.Structure):
        _fields_ = [("lpVtbl", ctypes.POINTER(EventVTable)), ("owner", ctypes.py_object)]

    class ControlInterface(ctypes.Structure):
        _fields_ = [("lpVtbl", ctypes.POINTER(ControlVTable)), ("owner", ctypes.py_object)]

    def interface_owner(this, interface_type):
        return ctypes.cast(this, ctypes.POINTER(interface_type)).contents.owner

    IID_IUNKNOWN = GUID.parse("00000000-0000-0000-C000-000000000046")
    IID_EVENTS = GUID.parse(_IID_FILE_DIALOG_EVENTS)
    IID_CONTROLS = GUID.parse(_IID_FILE_DIALOG_CONTROL_EVENTS)

    def query_interface(this, iid, output):
        owner = interface_owner(this, EventInterface)
        requested = ctypes.string_at(iid, ctypes.sizeof(GUID))
        if requested == bytes(IID_IUNKNOWN) or requested == bytes(IID_EVENTS):
            selected = ctypes.addressof(owner.events)
        elif requested == bytes(IID_CONTROLS):
            selected = ctypes.addressof(owner.controls)
        else:
            output[0] = None
            return -2147467262  # E_NOINTERFACE
        output[0] = PVOID(selected)
        owner.references += 1
        return 0

    def add_ref(this):
        owner = interface_owner(this, EventInterface)
        owner.references += 1
        return owner.references

    def release(this):
        owner = interface_owner(this, EventInterface)
        owner.references = max(0, owner.references - 1)
        return owner.references

    def close_for_folder(this, customize, control_id):
        owner = interface_owner(this, ControlInterface)
        if control_id == _CUSTOM_FOLDER_BUTTON:
            owner.folder_selected = True
            hr = method(owner.dialog, 23, HRESULT, HRESULT)(owner.dialog, HRESULT(0))
            return hr
        return 0

    def no_op(*_args):
        return 0

    def no_op_with_response(_this, _dialog, _item, response):
        response[0] = 0
        return 0

    def no_op_folder_changing(_this, _dialog, _folder):
        return 0

    QueryInterface = WINFUNCTYPE(HRESULT, PVOID, ctypes.POINTER(GUID), ctypes.POINTER(PVOID))(query_interface)
    AddRef = WINFUNCTYPE(ULONG, PVOID)(add_ref)
    Release = WINFUNCTYPE(ULONG, PVOID)(release)
    EventMethod = WINFUNCTYPE(HRESULT, PVOID, PVOID)
    EventMethodWithItem = WINFUNCTYPE(HRESULT, PVOID, PVOID, PVOID)
    ShareOrOverwrite = WINFUNCTYPE(HRESULT, PVOID, PVOID, PVOID, ctypes.POINTER(ctypes.c_int))
    _EVENT_CALLBACKS = []
    EventVTable._fields_ = [
        ("QueryInterface", type(QueryInterface)), ("AddRef", type(AddRef)), ("Release", type(Release)),
        ("OnFileOk", EventMethod), ("OnFolderChanging", EventMethodWithItem),
        ("OnFolderChange", EventMethod), ("OnSelectionChange", EventMethod),
        ("OnShareViolation", ShareOrOverwrite), ("OnTypeChange", EventMethod),
        ("OnOverwrite", ShareOrOverwrite),
    ]
    _EVENT_VTABLE = EventVTable()
    _EVENT_CALLBACKS.extend([QueryInterface, AddRef, Release])
    _EVENT_VTABLE.QueryInterface = QueryInterface
    _EVENT_VTABLE.AddRef = AddRef
    _EVENT_VTABLE.Release = Release
    for field, prototype, callback in (
        ("OnFileOk", EventMethod, no_op), ("OnFolderChanging", EventMethodWithItem, no_op_folder_changing),
        ("OnFolderChange", EventMethod, no_op), ("OnSelectionChange", EventMethod, no_op),
        ("OnShareViolation", ShareOrOverwrite, no_op_with_response),
        ("OnTypeChange", EventMethod, no_op), ("OnOverwrite", ShareOrOverwrite, no_op_with_response),
    ):
        handler = prototype(callback)
        setattr(_EVENT_VTABLE, field, handler)
        _EVENT_CALLBACKS.append(handler)

    ControlMethod = WINFUNCTYPE(HRESULT, PVOID)
    ControlButton = WINFUNCTYPE(HRESULT, PVOID, PVOID, DWORD)
    ControlItem = WINFUNCTYPE(HRESULT, PVOID, PVOID, DWORD, DWORD)
    ControlCheck = WINFUNCTYPE(HRESULT, PVOID, PVOID, DWORD, ctypes.c_int)
    ControlVTable._fields_ = [
        ("QueryInterface", type(QueryInterface)), ("AddRef", type(AddRef)), ("Release", type(Release)),
        ("OnItemSelected", ControlItem), ("OnButtonClicked", ControlButton),
        ("OnCheckButtonToggled", ControlCheck), ("OnControlActivating", ControlButton),
    ]
    _CONTROL_VTABLE = ControlVTable()
    _CONTROL_VTABLE.QueryInterface = QueryInterface
    _CONTROL_VTABLE.AddRef = AddRef
    _CONTROL_VTABLE.Release = Release
    for field, prototype, callback in (
        ("OnItemSelected", ControlItem, no_op), ("OnButtonClicked", ControlButton, close_for_folder),
        ("OnCheckButtonToggled", ControlCheck, no_op), ("OnControlActivating", ControlButton, no_op),
    ):
        handler = prototype(callback)
        setattr(_CONTROL_VTABLE, field, handler)
        _EVENT_CALLBACKS.append(handler)

    # The interface structures retain the event callbacks for the duration of Show().
    sink = EventSink()
    dialog = PVOID()
    customize = PVOID()
    shell_folder = PVOID()
    results = PVOID()
    cookie = DWORD()
    ole32 = ctypes.WinDLL("ole32")
    shell32 = ctypes.WinDLL("shell32")
    initialized = False
    advised = False
    try:
        ole32.CoInitializeEx.argtypes = [PVOID, DWORD]
        ole32.CoInitializeEx.restype = HRESULT
        init_hr = ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED
        if init_hr in (0, 1):
            initialized = True
        elif init_hr == -2147417850:  # RPC_E_CHANGED_MODE
            raise NativeMediaDialogError("The current thread is not in a Windows UI COM apartment.")
        else:
            check_hresult(init_hr, "CoInitializeEx")

        clsid = GUID.parse(_CLSID_FILE_OPEN_DIALOG)
        iid_open = GUID.parse(_IID_FILE_OPEN_DIALOG)
        ole32.CoCreateInstance.argtypes = [ctypes.POINTER(GUID), PVOID, DWORD, ctypes.POINTER(GUID), ctypes.POINTER(PVOID)]
        ole32.CoCreateInstance.restype = HRESULT
        hr = ole32.CoCreateInstance(ctypes.byref(clsid), None, 1, ctypes.byref(iid_open), ctypes.byref(dialog))
        check_hresult(hr, "Creating Windows media dialog")
        sink.dialog = dialog

        check_hresult(method(dialog, 7, HRESULT, PVOID, ctypes.POINTER(DWORD))(dialog, ctypes.byref(sink.events), ctypes.byref(cookie)), "Registering Windows dialog events")
        advised = True
        iid_customize = GUID.parse(_IID_FILE_DIALOG_CUSTOMIZE)
        check_hresult(method(dialog, 0, HRESULT, ctypes.POINTER(GUID), ctypes.POINTER(PVOID))(dialog, ctypes.byref(iid_customize), ctypes.byref(customize)), "Enabling Windows dialog customization")
        check_hresult(method(customize, 5, HRESULT, DWORD, WCHAR_P)(customize, DWORD(_CUSTOM_FOLDER_BUTTON), "Select Current Folder"), "Adding folder selection action")

        options = DWORD()
        check_hresult(method(dialog, 10, HRESULT, ctypes.POINTER(DWORD))(dialog, ctypes.byref(options)), "Reading Windows dialog options")
        options.value |= 0x8 | 0x40 | 0x200 | 0x800 | 0x1000
        check_hresult(method(dialog, 9, HRESULT, DWORD)(dialog, options), "Setting Windows dialog options")

        class FilterSpec(ctypes.Structure):
            _fields_ = [("name", WCHAR_P), ("pattern", WCHAR_P)]

        pattern = ";".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        filters = (FilterSpec * 1)(FilterSpec("Supported media", pattern))
        check_hresult(method(dialog, 4, HRESULT, DWORD, ctypes.POINTER(FilterSpec))(dialog, 1, filters), "Setting media file filter")
        check_hresult(method(dialog, 17, HRESULT, WCHAR_P)(dialog, "Select media files or choose the current folder"), "Setting dialog title")

        shell32.SHCreateItemFromParsingName.argtypes = [WCHAR_P, PVOID, ctypes.POINTER(GUID), ctypes.POINTER(PVOID)]
        shell32.SHCreateItemFromParsingName.restype = HRESULT
        shell_iid = GUID.parse(_IID_SHELL_ITEM)
        if start_directory.is_dir():
            hr = shell32.SHCreateItemFromParsingName(str(start_directory.resolve()), None, ctypes.byref(shell_iid), ctypes.byref(shell_folder))
            check_hresult(hr, "Opening the last media directory")
            check_hresult(method(dialog, 12, HRESULT, PVOID)(dialog, shell_folder), "Selecting the last media directory")

        owner = int(parent.winId()) if parent is not None else None
        show_hr = method(dialog, 3, HRESULT, PVOID)(dialog, owner)
        if show_hr == -2147023673:  # HRESULT_FROM_WIN32(ERROR_CANCELLED)
            return []
        check_hresult(show_hr, "Displaying Windows media dialog")

        if sink.folder_selected:
            folder = PVOID()
            check_hresult(method(dialog, 13, HRESULT, ctypes.POINTER(PVOID))(dialog, ctypes.byref(folder)), "Reading selected folder")
            try:
                return [_shell_item_path(folder, method)]
            finally:
                method(folder, 2, ULONG)(folder)

        get_results = method(dialog, 27, HRESULT, ctypes.POINTER(PVOID))
        check_hresult(get_results(dialog, ctypes.byref(results)), "Reading selected media")
        count = DWORD()
        check_hresult(method(results, 3, HRESULT, ctypes.POINTER(DWORD))(results, ctypes.byref(count)), "Counting selected media")
        selected = []
        for index in range(count.value):
            item = PVOID()
            check_hresult(method(results, 4, HRESULT, DWORD, ctypes.POINTER(PVOID))(results, index, ctypes.byref(item)), "Reading selected media item")
            try:
                selected.append(Path(_shell_item_path(item, method)))
            finally:
                method(item, 2, ULONG)(item)
        return selected
    except NativeMediaDialogError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise NativeMediaDialogError(f"Could not open the Windows Explorer media dialog: {exc}") from exc
    finally:
        if advised and dialog:
            method(dialog, 8, HRESULT, DWORD)(dialog, cookie)
        for interface in (results, shell_folder, customize, dialog):
            if interface:
                method(interface, 2, ULONG)(interface)
        if initialized:
            ole32.CoUninitialize()
        sink.events.owner = None
        sink.controls.owner = None


def _shell_item_path(item, method) -> str:
    display_name = ctypes.c_void_p()
    # SIGDN_FILESYSPATH returns a CoTaskMemAlloc string owned by the caller.
    ctypes_method = method(item, 5, ctypes.c_long, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))
    hr = ctypes_method(item, 0x80058000, ctypes.byref(display_name))
    if hr < 0:
        raise OSError(f"Reading the selected filesystem path failed (HRESULT 0x{hr & 0xFFFFFFFF:08X})")
    try:
        return ctypes.wstring_at(display_name)
    finally:
        ctypes.WinDLL("ole32").CoTaskMemFree(display_name)
