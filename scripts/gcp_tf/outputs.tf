output ip {
  value = "${google_compute_instance.experiment.network_interface.0.access_config.0.nat_ip}"
}
