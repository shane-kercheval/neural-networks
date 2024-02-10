"""Helper functions for data management."""


def get_names() -> list[str]:
    """Get names from file (https://github.com/karpathy/makemore)."""
    with open('/code/data/names.txt') as f:
        names = f.readlines()
    return [name.strip() for name in names]
