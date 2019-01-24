"""Deploys spot instance and runs experiment according to config file"""
from python_terraform import *
import sys
import argparse
import yaml

parser = argparse.ArgumentParser(description='Deploy spot instance and run experiment')
parser.add_argument('--file', '-f', type=str,
                    help='Configuration file to reference',
                    required=True)
parser.add_argument('--terraform', '-t', type=str,
                    default=".",
                    help="Directory to target with terraform")
parser.add_argument('--dryrun', action="store_true",
                    help='Run terraform plan, not apply')

args = parser.parse_args()


def get_config(fname: str) -> dict:
    return yaml.load(open(fname, "rt"))


def deploy_spot_instance(config: dict) -> None:
    t = Terraform()
    options = {
        "ami": "ami-012b19f1736b6aae8", # Deep Learning Base AMI (Ubuntu) Version 14.0
        "config_file": args.file,
        "notebook": config["notebook"],
    }
    if "terraform_vars" in config: options.update(config["terraform_vars"])

    if args.dryrun:
        return_code, stdout, stderr = t.plan(args.terraform, var=options)
    else:
        print(t.plan(args.terraform, var=options))
        t.apply(args.terraform,
                skip_plan=True,
                var=options,
                capture_output=False,
                **{"auto-approve": True})

    print(stdout)
    print(stderr, file=sys.stderr)

if __name__ == "__main__":
    config = get_config(args.file)
    print(f"Read config: {config}")
    deploy_spot_instance(config)
