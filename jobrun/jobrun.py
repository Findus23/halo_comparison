import random
from pathlib import Path
from subprocess import run
from time import sleep
from typing import List, Union

import yaml

config_file = Path("~/.config/jobrun.yaml").expanduser()


def jobrun(
        args: List[Union[str, Path]],
        pwd: Path = None,
        time: str = "05:10:00",
        tasks: int = None,
        nodes: int = None,
        mem: int = None,
        qos: str = None,
        account: str = None,
        source: Path = None,
        sbatch: bool = True
):
    with config_file.open() as f:
        config = yaml.safe_load(f)
    jobscript_dir = Path(config["jobscript_dir"]).expanduser().absolute()
    name = random.randint(0, 100000)
    jobscript_file = jobscript_dir / f"{name}.sh"
    jobscript_lines = ["#!/bin/bash"]
    sbatch_options = {
        "mail-type": "ALL",
        "mail-user": config["mail"],
        "time": time,
        "nodes": nodes,
        "tasks": tasks,
        "mem": mem,
        "qos": qos,
        "account": account,
        "job-name": name,
        "output": jobscript_dir / f"out-{name}-%a_%a.txt"
    }
    for key, value in sbatch_options.items():
        if value is not None:
            jobscript_lines.append(f"#SBATCH --{key}={value}")

    if source is not None:
        jobscript_lines.append(f"source {source.absolute()}")

    if pwd is not None:
        assert pwd.exists()
        jobscript_lines.append(f"cd {pwd.absolute()}")

    strargs: List[str] = []
    for a in args:
        if isinstance(a, Path):
            strargs.append(str(a.absolute()))
        else:
            strargs.append(a)
    jobscript_lines.append(" ".join(strargs))
    jobscript_lines.append("")
    jobscript_file.write_text("\n".join(jobscript_lines))

    print(f"created {jobscript_file.name}")

    if sbatch:
        run([
            "sbatch",
            str(jobscript_file)
        ], check=True)
        sleep(2)
