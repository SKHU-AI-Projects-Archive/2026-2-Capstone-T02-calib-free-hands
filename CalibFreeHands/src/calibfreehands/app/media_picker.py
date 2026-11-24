from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QDir
from PySide6.QtWidgets import (
    QDialog, QFileSystemModel, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTreeView, QVBoxLayout,
)

from calibfreehands.media.discovery import SUPPORTED_EXTENSIONS


class MediaOpenDialog(QDialog):
    """Browse supported files or add the current folder's direct media files."""

    def __init__(self, parent=None, start_directory: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open Media")
        self.resize(760, 520)
        self._selected_paths: list[Path] = []
        directory = Path(start_directory or Path.home()).expanduser()
        if not directory.is_dir():
            directory = Path.home()
        self.path_edit = QLineEdit(str(directory.resolve()))
        self.path_edit.setReadOnly(True)
        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self.model.setNameFilters([f"*{extension}" for extension in sorted(SUPPORTED_EXTENSIONS)])
        self.model.setNameFilterDisables(False)
        self.model.setRootPath(str(directory.resolve()))
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(str(directory.resolve())))
        self.tree.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)
        for column in range(1, 4):
            self.tree.hideColumn(column)
        self.tree.doubleClicked.connect(self._navigate)
        self.up_button = QPushButton("Up")
        self.up_button.clicked.connect(self._up)
        self.add_files_button = QPushButton("Add selected files")
        self.add_files_button.clicked.connect(self._accept_files)
        self.add_folder_button = QPushButton("Add this folder")
        self.add_folder_button.clicked.connect(self._accept_folder)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        navigation = QHBoxLayout(); navigation.addWidget(self.up_button); navigation.addWidget(self.path_edit, 1)
        actions = QHBoxLayout(); actions.addWidget(self.add_files_button); actions.addWidget(self.add_folder_button); actions.addStretch(1); actions.addWidget(cancel)
        layout = QVBoxLayout(self); layout.addWidget(QLabel("Select supported media files, or add the current folder.")); layout.addLayout(navigation); layout.addWidget(self.tree, 1); layout.addLayout(actions)

    def _current_directory(self) -> Path:
        return Path(self.path_edit.text())

    def _navigate(self, index) -> None:
        path = Path(self.model.filePath(index))
        if path.is_dir():
            self.path_edit.setText(str(path.resolve()))
            self.tree.setRootIndex(self.model.index(str(path.resolve())))

    def _up(self) -> None:
        parent = self._current_directory().parent
        if parent != self._current_directory():
            self.path_edit.setText(str(parent))
            self.tree.setRootIndex(self.model.index(str(parent)))

    def _accept_files(self) -> None:
        paths = [Path(self.model.filePath(index)).resolve()
                 for index in self.tree.selectionModel().selectedRows(0)
                 if not self.model.isDir(index)]
        if paths:
            self._selected_paths = sorted(set(paths), key=lambda path: str(path).casefold())
            self.accept()

    def _accept_folder(self) -> None:
        self._selected_paths = [self._current_directory().resolve()]
        self.accept()

    def selected_paths(self) -> list[Path]:
        return list(self._selected_paths)
