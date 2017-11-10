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
	def __init__(self):
		# Log Contains the performance output (CSV) for each test.
		# Create Log folder whit platforms subfolders, if not exist
		#if not os.path.exists('/output/logs'):
		if not os.path.exists('logs'):
			#os.makedirs('/output/logs')
			os.makedirs('logs')

		# sys.argv[1] = platform (cpu, gpu)
		#if not os.path.exists('/output/logs/{}'.format(sys.argv[1])):
			#os.makedirs('/output/logs/{}'.format(sys.argv[1]))
		if not os.path.exists('logs/{}'.format(sys.argv[1])):
			os.makedirs('logs/{}'.format(sys.argv[1]))

		# sys.argv[2] = platform (fh, aws, gce)
		# if not os.path.exists('/output/logs/{}/{}'.format(sys.argv[2])):
		# 	os.makedirs('/output/logs/{}/{}'.format(sys.argv[2]))
		if not os.path.exists('logs/{}/{}'.format(sys.argv[2])):
			os.makedirs('logs/{}/{}'.format(sys.argv[2]))

	"""Log stats during training"""
	def on_train_begin(self, logs={}):
		"""Create Log file, set Keras backend(default: TF) and prepare
		Log file columns name(stats)"""
		filename = os.path.basename(sys.argv[0])[:-3]
		backend = K.backend()
		# Save Log in the /output folder(FH spec)
		# self.f = open('/output/logs/{}/{}/{}_{}.csv'.format(sys.argv[1],
		# 													sys.argv[2],
		# 													filename,
		# 													backend), 'w')
		self.f = open('logs/{}/{}/{}_{}.csv'.format(sys.argv[1],
															sys.argv[2],
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
