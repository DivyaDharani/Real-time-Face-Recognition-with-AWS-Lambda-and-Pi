#!/bin/sh
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
  echo "Inside AWS LAMBDA RUNTIME API if condition in entry.sh"
	exec /usr/bin/aws-lambda-rie /usr/local/bin/python3.8 -m awslambdaric "$1"
else
  echo "Inside AWS LAMBDA RUNTIME API else condition in entry.sh"
	exec /usr/local/bin/python3.8 -m awslambdaric "$1"
fi