class AnyCalibAdapter:
    """Adapter placeholder for connecting the official AnyCalib implementation."""

    def __init__(self, checkpoint=None, pretrained=True):
        self.checkpoint = checkpoint
        self.pretrained = pretrained
        self.model = None

    def build(self):
        raise NotImplementedError(
            "Connect the official AnyCalib model here after its code structure is confirmed."
        )
