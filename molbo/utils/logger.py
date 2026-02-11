import wandb


class WandBLogger:
    """Wrapper for WandB logging."""

    def __init__(
        self,
        project_name: str,
        run_name: str = None,
        group_name: str = None,
        tags: list = None,
        config: dict = None,
        mode: str = "online",
        silent: bool = True,
    ):
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            group=group_name,
            tags=tags,
            config=config,
            mode=mode,
            settings=wandb.Settings(silent=silent),
        )

    def log(self, data: dict):
        self.run.log(data)

    def finish(self):
        self.run.finish()
