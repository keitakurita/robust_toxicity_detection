from python_terraform import *
import sys
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dryrun', action="store_true",
                    help='Run terraform plan, not apply')

args = parser.parse_args()

if __name__ == "__main__":
    t = Terraform()
    options = {
        "ami": "ami-012b19f1736b6aae8", # Deep Learning Base AMI (Ubuntu) Version 14.0
    }

    if args.dryrun:
        return_code, stdout, stderr = t.plan(".", var=options)
    else:
        return_code, stdout, stderr = t.apply(".", var=options)

    print(stdout)
    print(stderr, file=sys.stderr)
