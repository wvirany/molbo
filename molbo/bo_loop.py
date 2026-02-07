# Currently adding checkpointing via a model state dict, and will add a place to dump observations to save checkpoints


class BOLoop:

    def __init__(self, checkpoints=False):
        self.state_dict = None
        self.checkpoints = checkpoints

    def initialize_bo(self):
        # Load state dict for model and checkpoint if given
        if self.checkpoints:
            if self.state_dict is not None:
                pass
            self.load_checkpoint()
            # or just load_checkpoint, and read state dict there?

    def load_checkpoint(self):
        pass
