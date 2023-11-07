
import rich.syntax
import rich.tree
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf



@rank_zero_only
def print_config(
    config: DictConfig,
    resolve: bool = True,
    save_cfg=True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_cfg:
        with open("config_tree.txt", "w") as fp:
            rich.print(tree, file=fp)