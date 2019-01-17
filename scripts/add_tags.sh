#!/usr/bin/env bash

#$1 = ${aws_spot_instance_request.platform.id}
#$2 = ${aws_spot_instance_request.platform.spot_instance_id}

aws --region us-east-1 ec2 describe-spot-instance-requests --spot-instance-request-ids $1 --query 'SpotInstanceRequests[0].Tags' > tags.json
aws ec2 create-tags --resources $2 --tags file://tags.json
rm -f tags.json
