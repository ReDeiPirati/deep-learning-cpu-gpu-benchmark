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

# Platform Available on AWS
PLATFORM_TYPE = ["gpu"]

cloud_service_name = sys.argv[1]
if sys.argv[1] != "aws" and sys.argv[1] != "gce":
	print ("Provide aws or gce as param.")
	exit(-1)

# Test file script list
test_files = [f for f in os.listdir("test_files") if f.endswith('.py')]
test_files.remove('CustomCallback.py')  # exclude the Logger Class

# Test Launcher
for test_file in test_files:
	# Run the same script for different platform
	for platform in PLATFORM_TYPE:
		# Command Template
		# floyd run --env .. --(gpu|cpu) --message testfile_plat
		# "python test/script (cpu|gpu) aws"
		command = "python test_files/{} {} {}".format(test_file, platform, cloud_service_name)

		# Command to run
		print(command)
		# Wait until the end of the subprocess
		subprocess.call(command.split(), shell=False)  # Safe
