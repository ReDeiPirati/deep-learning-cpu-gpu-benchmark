"""
Launcher Script for benchmarking AWS Service(gpu) on:

(Must be logged in on FH to run this)

- MLP on MNIST
- ConvNet on MNIST
- ConvNet on CIFAR-10
- Bi-LSTM on IDBM
- FastText on IDBM
"""

import os
import subprocess

# Platform Available on AWS
PLATFORM_TYPE = ["gpu"]

# Test file script list
test_files = [f for f in os.listdir("test_files") if f.endswith('.py')]
test_files.remove('CustomCallback.py')  # exclude the Logger Class

# Test Launcher
for test_file in test_files:
	# Run the same script for different platform
	for platform in PLATFORM_TYPE:
		# Command Template
		# floyd run --env .. --(gpu|cpu) --message testfile_plat
		# "python test/script (cpu|gpu) fh"
		command = "python test_files/{} {} aws".format(test_file, platform)

		# Command to run
		print(command)
		#subprocess.call(command.split(), shell=False)  # Safe
		# Wait ending then lunch next
		# TODO: auto csv-log retrieval after jobs ending.
