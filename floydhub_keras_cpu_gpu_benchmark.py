"""
Launcher Script for benchmarking FloydHub Service(gpu and cpu) on:
Env: TF 1.4 - Keras 2.0.6
(Must be logged in on FH to run this)

- MLP on MNIST
- ConvNet on MNIST
- ConvNet on CIFAR-10
- Bi-LSTM on IDBM
- LSTM Text Generation
- FastText on IDBM
"""

import os
import subprocess

# TODO Add Runs parameter
RUNS = 2

# Platform Available on FH
# TODO Add Platform parameter
PLATFORM_TYPE = ["cpu", "gpu", "cpu2", "gpu2"]
#PLATFORM_TYPE = ["cpu"]


# Test file script list
# TODO Add File parameter
test_files = [f for f in os.listdir("test_files") if f.endswith('.py')]
test_files.remove('CustomCallback.py')  # exclude the Logger Class

# Prepare Floyd commands
# TODO Add floyd-dev  parameter
floyd_cmd = "floyd-dev run --env tensorflow-1.4 "
#floyd_cmd = "floyd run --env tensorflow-1.4 "

# Test Launcher
for test_file in test_files:
	if test_file.startswith("mnist_mlp"):
		# Run the same script for different platform
		for platform in PLATFORM_TYPE:
			# Help in Web dashboard visualization
			message = "{}_{}_runs_{}".format(test_file[:-3], platform, RUNS)
			# Multiple RUNS
			loop_cmd = "for r in $(seq {});do ".format(RUNS)
			# Command Template
			# floyd run --env .. --(gpu|cpu) --message testfile_plat
			# 'for r in $(seq RUNS); do
			# python test/script (cpu|gpu) fh <current_run>; done'
			command = floyd_cmd + \
				"--{p} --message {m} ".format(p=platform, m=message) + \
				loop_cmd + \
				"python test_files/{} {} {} {}; done".format(test_file, platform, "fh", "$r")

			# Command to run
			print("CMD => ", command)
			subprocess.call(command.split(), shell=False)  # Safe

			# TODO: Monitoring status to launch new concurrent jobs and keep the queue full
			# TODO: auto csv-log retrieval after jobs ending.
