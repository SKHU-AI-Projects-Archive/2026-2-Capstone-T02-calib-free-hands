from PySide6.QtWidgets import QLabel

from .i18n import tr


class TrLabel(QLabel):
    """A label for status lines that the code sets again and again: every text is translated when it is set
    (see app/i18n.py; in English ``tr`` returns the text unchanged and costs nothing)."""

    def setText(self, text: str) -> None:  # noqa: N802 - Qt name
        super().setText(tr(text))
