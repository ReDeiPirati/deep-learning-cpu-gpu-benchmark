"""
Class Logger during training.
Export stats in a csv log file.
"""

import sys
import csv
import os
import time
from keras import backend as K
from keras.callbacks import Callback


class EpochStatsLogger(Callback):
	def __init__(self, cloud_service):
		self.cloud_service = cloud_service
		# Log Contains the performance output (CSV) for each test.
		# Create Log folder whit platforms subfolders, if not exist
		if self.cloud_service == "fh":
			# FH path compliant
			if not os.path.exists('/output/logs'):
				os.makedirs('/output/logs')
			# FH folder
			if not os.path.exists('/output/logs/{}'.format(self.cloud_service)):
				os.makedirs('/output/logs/{}'.format(self.cloud_service))
			# Platform [CPU, GPU]
			if not os.path.exists('/output/logs/{}/{}'.format(self.cloud_service, sys.argv[1])):
				os.makedirs('/output/logs/{}/{}'.format(self.cloud_service, sys.argv[1]))
		else:
			# GCE and AWS use relative logs
			if not os.path.exists('logs'):
				os.makedirs('logs')
			# AWS or GCE folder
			if not os.path.exists('logs/{}'.format(self.cloud_service)):
				os.makedirs('logs/{}'.format(self.cloud_service))
			# Platform [CPU, GPU]
			if not os.path.exists('logs/{}/{}'.format(self.cloud_service, sys.argv[1])):
				os.makedirs('logs/{}/{}'.format(self.cloud_service, sys.argv[1]))

			logdir = 'logs/{}/{}'.format(self.cloud_service, sys.argv[1])
			# Delete existing csv
			filelist = [f for f in os.listdir(logdir) if f.endswith(".csv")]
			for f in filelist:
				os.remove(os.path.join(logdir, f))

	"""Log stats during training"""
	def on_train_begin(self, logs={}):
		"""Create Log file, set Keras backend(default: TF) and prepare
		Log file columns name(stats)"""
		filename = os.path.basename(sys.argv[0])[:-3]
		backend = K.backend()
		if self.cloud_service == "fh":
			self.f = open('/output/logs/{}/{}/{}_{}.csv'.format(self.cloud_service,
																sys.argv[1],
																filename,
																backend), 'w')
		else:  # AWS and GCE
			self.f = open('logs/{}/{}/{}_{}.csv'.format(self.cloud_service,
																sys.argv[1],
																filename,
																backend), 'w')
		self.log_writer = csv.writer(self.f)
		self.log_writer.writerow(['epoch', 'elapsed', 'loss',
								'acc', 'val_loss', 'val_acc'])

	def on_train_end(self, logs={}):
		"""Close Log file descriptor on train end"""
		self.f.close()

	def on_epoch_begin(self, epoch, logs={}):
		"""Save time on epoch begin"""
		self.start_time = time.time()

	def on_epoch_end(self, epoch, logs={}):
		"""Append a row with stats on epcoch end """
		self.log_writer.writerow([epoch, time.time() - self.start_time,
															logs.get('loss'),
															logs.get('acc'),
															logs.get('val_loss'),
															logs.get('val_acc')])
