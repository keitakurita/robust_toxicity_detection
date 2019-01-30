"""Deploys spot instance and runs experiment according to config file"""
from python_terraform import *
import sys
import argparse
import yaml
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Deploy spot instance and run experiment')
parser.add_argument('--file', '-f', type=str,
                    help='Configuration file to reference',
                    required=True)
parser.add_argument('--terraform', '-t', type=str,
                    default=".",
                    help="Directory to target with terraform")
parser.add_argument('--dryrun', action="store_true",
                    help='Run terraform plan, not apply')
parser.add_argument('--duration', "-d", type=int, default=3600,
                    help='Duration to wait for experiment to finish (secs.)')

args = parser.parse_args()


def get_config(fname: str) -> dict:
    return yaml.load(open(fname, "rt"))


def deploy_spot_instance(config: dict) -> None:
    t = Terraform()
    options = {
        "ami": "ami-012b19f1736b6aae8", # Deep Learning Base AMI (Ubuntu) Version 14.0
        "config_file": args.file,
        "notebook": "../notebooks/" + config["notebook"],
    }
    if "terraform_vars" in config: options.update(config["terraform_vars"])

    if args.dryrun:
        return_code, stdout, stderr = t.plan(args.terraform, var=options)
    else:
        return_code, stdout, stderr = t.plan(args.terraform, var=options)
        print(stdout)
    try:
        t.apply(args.terraform,
                skip_plan=True,
                var=options,
                capture_output=False,
                **{"auto-approve": True})

        if args.duration > 0:
            print(f"Waiting for {args.duration} secs. for experiment to finish")
            for i in tqdm(range(args.duration)): time.sleep(1)
    finally:
        # destroy resources
        if args.duration > 0:
            print("Destroying resources")
            t.destroy(args.terraform,
                      var=options,
                      capture_output=False,
                      **{"auto-approve": True})
        else:
            print("Warning: Your resources will not be automatically terminated!!")

if __name__ == "__main__":
    config = get_config(args.file)
    print(f"Read config: {config}")
    deploy_spot_instance(config)
