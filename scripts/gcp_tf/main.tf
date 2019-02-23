################################
# Obtain preemptible instance  #
################################
provider "google" {
  region      = "us-east1"
  project     = "${var.project}"
  credentials = "${file("~/.gcp/account.json")}"
}

resource "google_compute_firewall" "default" {
  name    = "experiment-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8888", "6000"]
  }

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_instance" "experiment" {
  name         = "experiment"
  machine_type = "${var.instance_type}"
  zone         = "${var.zone}"

  boot_disk {
    initialize_params {
      image = "${var.image}"
      size  = 120
    }
  }

  guest_accelerator {
    type  = "${var.gpu_type}"
    count = 1
  }

  network_interface {
    network = "default"

    access_config {}
  }

  scheduling {
    preemptible       = true
    automatic_restart = false
  }

  metadata {
    install-nvidia-driver = true
    sshKeys               = "gcpuser:${file("~/.ssh/id_rsa.pub")}"
  }

  metadata_startup_script = "${file("./setup.sh")}"

  tags = ["experiment"]
}
