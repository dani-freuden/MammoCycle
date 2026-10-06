from pathlib import Path
from typing import Optional, Iterable, Any

import torch



def write_lines(filename: Path, lines: Iterable[Any], sort: bool = False) -> None:
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    lines = (f'{line}\n' for line in lines)

    if sort:
        lines = sorted(lines)

    with open(filename, 'wt') as f:
        f.writelines(lines)
