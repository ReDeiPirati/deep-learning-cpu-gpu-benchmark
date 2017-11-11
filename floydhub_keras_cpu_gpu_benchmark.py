"""
Launcher Script for benchmarking FloydHub Service(gpu and cpu) on:
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

# Platform Available on FH
PLATFORM_TYPE = ["cpu", "gpu"]

# Test file script list
test_files = [f for f in os.listdir("test_files") if f.endswith('.py')]
test_files.remove('CustomCallback.py')  # exclude the Logger Class

# Prepare Floyd commands
floyd_cmd = "floyd run --env tensorflow-1.3"

# Test Launcher
for test_file in test_files:
	# Run the same script for different platform
	for platform in PLATFORM_TYPE:
		# Help in Web dashboard visualization
		message = "{}_{}".format(test_file[:-3], platform)
		# Command Template
		# floyd run --env .. --(gpu|cpu) --message testfile_plat
		# "python test/script (cpu|gpu) fh"
		command = floyd_cmd + \
			" --{p} --message {m} ".format(p=platform, m=message) + \
			"python test_files/{} {} {}".format(test_file, platform, "fh")

		# Command to run
		print(command)
		subprocess.call(command.split(), shell=False)  # Safe
		# TODO: auto csv-log retrieval after jobs ending.
