from pathlib import Path


class GigaHandsDataset:
    """Minimal placeholder for a future GigaHands dataset loader."""

    def __init__(self, root, split_file=None, transform=None):
        self.root = Path(root)
        self.split_file = Path(split_file) if split_file else None
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"GigaHands root does not exist: {self.root}")

        # TODO: Implement sample indexing after the annotation and image layout
        # are confirmed from the official GigaHands documentation/protocol.
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise NotImplementedError(
            "GigaHands sample loading is not implemented yet. "
            "Confirm the annotation/image structure before adding loader logic."
        )
