output instance_ip {
  value = "ssh into instance with $ ssh -i ~/.ssh/cmu_cc_default.pem ubuntu@${aws_spot_instance_request.experiment.public_ip}"
}
