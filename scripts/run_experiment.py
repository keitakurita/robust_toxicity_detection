"""Runs experiment on AWS or GCP instance"""
import papermill as pm
import yaml
import argparse
from pathlib import Path

NB_ROOT = Path("../notebooks")

parser = argparse.ArgumentParser(description='Run experiment on AWS/GCP instance')
parser.add_argument("--file", "-f", type=str,
                    help="Configuration file to reference")
args = parser.parse_args()

def get_config(fname: str) -> dict:
    return yaml.load(open(fname, "rt"))

def _config_rep(d: dict, sep="_") -> str:
    rep_strs = []
    for k, v in sorted(d.items()):
        if isinstance(v, bool):
            if v: rep_strs.append(k)
        else:
            rep_strs.append(f"{k}={v}")
    return sep.join(rep_strs)

def run(config: dict):
    in_nb = NB_ROOT / config["notebook"]
    if config["id"] is None:
        run_id = _config_rep(config["parameters"])
    else:
        run_id = config["id"]
    out_nb = "s3://nnfornlp/notebooks/{in_nb.stem}_{run_id}.ipynb"
    # register so that notebook can reference
    config["parameters"]["run_id"] = run_id

    print(f"Executing notebook with {config['parameters']}")

    pm.execute_notebook(
        str(in_nb), out_nb,
        parameters=config["parameters"],
    )

    print("Finished executing notebook")

if __name__ == "__main__":
    config = get_config(args.file)
    print(f"Read config: {config}")
    run(config)
