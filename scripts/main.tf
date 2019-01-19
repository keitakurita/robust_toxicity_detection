#########################
# Obtain spot instance  #
#########################
provider "aws" {
  region = "us-east-1"
}

resource "aws_security_group" "sg" {
  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"

    cidr_blocks = [
      "0.0.0.0/0",
    ]
  }

  ingress {
    from_port = 80
    to_port   = 80
    protocol  = "tcp"

    cidr_blocks = [
      "0.0.0.0/0",
    ]
  }

  # outbound internet access
  # allowed: any traffic from anywhere
  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = [
      "0.0.0.0/0",
    ]
  }
}

data "template_file" "user_data" {
  template = "${file("./setup.sh")}"
}

resource "aws_spot_instance_request" "experiment" {
  # spot_price      = "${var.spot_price}"

  ami                  = "${var.ami}"
  instance_type        = "${var.instance_type}"
  key_name             = "${var.key_name}"
  availability_zone    = "us-east-1a"
  wait_for_fulfillment = true
  spot_type            = "one-time"

  tags {
    Name = "experiment"
  }

  security_groups = [
    "${aws_security_group.sg.name}",
  ]

  connection {
    user        = "ubuntu"
    private_key = "${file("~/.ssh/cmu_cc_default.pem")}"
    host        = "${aws_spot_instance_request.experiment.public_ip}"
  }

  # this is needed to add tags
  provisioner "local-exec" {
    # copied from https://github.com/terraform-providers/terraform-provider-aws/issues/32
    command = "./add_tags.sh ${aws_spot_instance_request.experiment.id} ${aws_spot_instance_request.experiment.spot_instance_id}"
  }

  provisioner "file" {
    source      = "./setup.sh"
    destination = "/tmp/setup.sh"
  }

  provisioner "file" {
    source      = "${var.config_file}"
    destination = "/tmp/${var.config_file}"
  }

  provisioner "remote-exec" {
    inline = [
      # TODO Add appropriate commands
      # see https://akomljen.com/terraform-and-aws-spot-instances/ 
      # for example of how to use bash scripts
      "chmod +x /tmp/setup.sh",

      "/tmp/setup.sh",
    ]
  }

  root_block_device {
    volume_type = "gp2"
  }
}
