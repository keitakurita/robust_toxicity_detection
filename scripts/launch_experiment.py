"""Deploys spot instance and runs experiment according to config file"""
from python_terraform import *
import sys
import argparse
import yaml

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dryrun', action="store_true",
                    help='Run terraform plan, not apply')
parser.add_argument('--file', '-f', type=str,
                    help='Configuration file to reference')

args = parser.parse_args()


def get_config(fname: str) -> dict:
    return yaml.load(open(fname, "rt"))


def deploy_spot_instance(config: dict) -> None:
    t = Terraform()
    options = {
        "ami": "ami-012b19f1736b6aae8", # Deep Learning Base AMI (Ubuntu) Version 14.0
        "config_file": args.file,
    }
    if "terraform_vars" in config: options.update(config["terraform_vars"])

    if args.dryrun:
        return_code, stdout, stderr = t.plan(".", var=options)
    else:
        return_code, stdout, stderr = t.apply(".", var=options)

    print(stdout)
    print(stderr, file=sys.stderr)

if __name__ == "__main__":
    config = get_config(args.file)
    print(f"Read config: {config}")
    deploy_spot_instance(config)
