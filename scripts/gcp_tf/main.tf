#########################
# Obtain spot instance  #
#########################
provider "gcp" {
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

resource "google_compute_instance" "experiment" {
  machine_type = "${var.instance_type}"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "${var.image}"
    }
  }

  scheduling {
    preemptible = true
  }

  metadata_startup_script = "${file("./setup.sh")}"

  tags = ["experiment"]
}
