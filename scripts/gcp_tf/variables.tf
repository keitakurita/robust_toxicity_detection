# n1-highmem-2:  13 GB, 2 CPU, $0.0250 preemptible
# n1-standard-4: 15 GB, 4 CPU, $0.0400 preemptible
variable instance_type {
  default = "n1-highmem-2"
}

variable project {
  default = "nnfornlp"
}

variable image {
  default = "deeplearning-platform-release/pytorch-latest-cu92"
}

variable zone {
  default = "us-east1-b"
}

variable config_file {
  default = "foo"
}

variable gpu_type {
  default = "nvidia-tesla-p100"
}
