"""
Launcher Script for benchmarking AWS Service(gpu) on:
Env: TF 1.3 - Keras 2.0.6
(Must be logged in on FH to run this)

- MLP on MNIST
- ConvNet on MNIST
- ConvNet on CIFAR-10
- Bi-LSTM on IDBM
- FastText on IDBM
"""

import os
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser(description='Benchmark script')
parser.add_argument("--aws", help="use aws",
                    action="store_true")
parser.add_argument("--gce", help="use gce",
                    action="store_true")
parser.add_argument("--runs",
                        dest="runs", default=1, type=int,
                        help="numbers of benchmark runs")
args = parser.parse_args()

# Runs parameter
RUNS = args.runs

# Platform Available on AWS|GCE
# TODO Add Platform parameter
PLATFORM_TYPE = ["gpu"]  # ["cpu", "gpu", "cpu2", "gpu2"]

if args.aws and args.gce:
	print ("provide only one platform")
	exit(-1)
elif args.aws:
	print ("using AWS instance")
	cloud_service_name = "aws"
elif args.gce:
	print ("using GCE instance")
	cloud_service_name = "gce"
else:
	print ("You have to specify --aws or --gce")
	exit(-1)

# Test file script list
# TODO Add File parameter
test_files = [f for f in os.listdir("test_files") if f.endswith('.py')]
test_files.remove('CustomCallback.py')  # exclude the Logger Class

# Test Launcher
for test_file in test_files:
	# Run the same script for different platform
	for platform in PLATFORM_TYPE:
		# Set number of Runs in bath executions
		for i in range(RUNS):
			# Command Template
			# "python test/script (cpu|gpu) [aws|gce] <current_run>"
			command = "python test_files/{} {} {} {}".format(test_file, platform, cloud_service_name, i+1) # i starts from zero

			# Command to run
			print("CMD => ", command)
			# Wait until the end of the subprocess
			subprocess.call(command.split(), shell=False)  # Safe
