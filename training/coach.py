import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from criteria.parsing_loss import parse_loss
from utils import common, train_utils
from criteria import id_loss, w_norm
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.e2style import E2Style
from training.ranger import Ranger


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		# Initialize network
		self.net = E2Style(self.opts).to(self.device)

		# Initialize loss
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.parse_lambda > 0:
			self.parse_loss = parse_loss.ParseLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
		self.mse_loss = nn.MSELoss().to(self.device).eval()

		# Initialize optimizer
		#self.optimizer = self.configure_optimizers()
		self.optimizer = self.configure_optimizers_frozen()
		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def train(self):
		self.net.eval()
		if self.opts.training_stage == 1:
			self.net.student_encoder_firststage.train()
		else:
			self.net.encoder_refinestage_list[self.opts.training_stage-2].train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()
				x, y = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()
				y_hat, latent = self.net.forward(x, return_latents=True)
				loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)   
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 1000 and self.global_step % 500 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')   
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, y = batch
			if batch_idx % 100 == 0:
				print(f'[Validate] Now processing batch {batch_idx}...',flush=True)
			with torch.no_grad():
				x, y = x.to(self.device).float(), y.to(self.device).float()
				y_hat, latent = self.net.forward(x, return_latents=True)
				loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hat,
									  title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				if self.opts.training_stage == 1:
					self.net.student_encoder_firststage.train()
				else:
					self.net.encoder_refinestage_list[self.opts.training_stage-2].train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		if self.opts.training_stage == 1:
			self.net.student_encoder_firststage.train()
		else:
			self.net.encoder_refinestage_list[self.opts.training_stage-2].train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'latest_model.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		
		#防止覆盖
		orig = os.path.abspath(getattr(self.opts, "checkpoint_path", ""))
		dst  = os.path.abspath(checkpoint_path)
		if orig and orig == dst:
			raise RuntimeError(f"Refuse to overwrite the source checkpoint: {dst}")
		torch.save(save_dict, checkpoint_path)
		
		if (self.global_step % self.opts.ckpt_every == 0) or is_best or (self.global_step == self.opts.max_steps):
			step_name = f"checkpoint_step{self.global_step:07d}.pt"
			step_path = os.path.join(self.checkpoint_dir, step_name)
			torch.save(save_dict, step_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers_frozen(self):   #frozen optical layer & stage=1
		params_to_train = []
		if self.opts.training_stage == 1:
			print("INFO: Configuring optimizer for Stage 1 (student_encoder_firststage).")
			if self.opts.training_optical:
				print("INFO: Training ALL layers of the student encoder, including 'input_layer'.")		
				params_to_train.extend(list(self.net.student_encoder_firststage.parameters()))
			else:
				print("INFO: The 'input_layer' will be FROZEN.")
				for name, param in self.net.student_encoder_firststage.named_parameters():
					if "input_layer" in name:
						param.requires_grad = False
					else:
						params_to_train.append(param)
						
		if self.opts.train_decoder:
			print("INFO: Decoder parameters will also be trained.")
			params_to_train.extend(list(self.net.decoder.parameters()))
		else:
			for param in self.net.decoder.parameters():
				param.requires_grad = False
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params_to_train, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params_to_train, lr=self.opts.learning_rate,weight_decay=1e-4)

		return optimizer
					
	def configure_optimizers(self):
		params = list(self.net.teacher_encoder_firststage.parameters()) if self.opts.training_stage == 1 else list(self.net.encoder_refinestage_list[self.opts.training_stage-2].parameters())
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		else:
			for param in self.net.decoder.parameters():
				param.requires_grad = False
		if self.opts.training_stage > 1:
			for param in self.net.teacher_encoder_firststage.parameters():
				param.requires_grad = False
		if self.opts.training_stage > 2:
			for idx in range(self.opts.training_stage-2):
				for param in self.net.encoder_refinestage_list[idx].parameters():
					param.requires_grad = False			
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
		print('Loading dataset for {}'.format(self.opts.dataset_type))
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset_celeba = ImagesDataset(source_root=dataset_args['train_source_root'],
		                                     target_root=dataset_args['train_target_root'],
		                                     source_transform=transforms_dict['transform_source'],
		                                     target_transform=transforms_dict['transform_gt_train'],
		                                     opts=self.opts)
		test_dataset_celeba = ImagesDataset(source_root=dataset_args['test_source_root'],
		                                    target_root=dataset_args['test_target_root'],
		                                    source_transform=transforms_dict['transform_source'],
		                                    target_transform=transforms_dict['transform_test'],
		                                    opts=self.opts)
		train_dataset = train_dataset_celeba
		test_dataset = test_dataset_celeba
		print("Number of training samples: {}".format(len(train_dataset)))
		print("Number of test samples: {}".format(len(test_dataset)))
		return train_dataset, test_dataset

	def calc_loss(self, x, y, y_hat, latent):
		loss_dict = {}
		loss = 0.0
		id_logs = None

		if self.opts.id_lambda > 0:
			loss_id = self.id_loss(y_hat, y)
			loss_dict['loss_id'] = float(loss_id)
			loss = loss_id * self.opts.id_lambda
		if self.opts.parse_lambda > 0:
			loss_parse = self.parse_loss(y_hat, y)
			loss_dict['loss_parse'] = float(loss_parse)
			loss += loss_parse * self.opts.parse_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print('Metrics for {}, step {}'.format(prefix, self.global_step))
		for key, value in metrics_dict.items():
			print('\t{} = '.format(key), value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.log_input_image(x[i], self.opts),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):    #warning:training distilling 
		full_state_dict = self.net.state_dict()   #global
		'''
		filtered_state_dict = {    #防止将其他名称保存下来
			k: v for k, v in full_state_dict.items()
			if k.startswith("student_encoder_firststage") or
			k.startswith("decoder")
		}
		'''
		save_dict = {
			'state_dict': full_state_dict,
			'opts': vars(self.opts)
		}
		
		# 保存 latent_avg（无前缀，另存）
		if self.opts.start_from_latent_avg and hasattr(self.net, 'latent_avg'):
			save_dict['latent_avg'] = self.net.latent_avg

		return save_dict
