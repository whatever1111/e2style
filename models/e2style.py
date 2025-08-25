import torch
from torch import nn
from models.encoders import backbone_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
import torch.nn.functional as F
from mobilenetv3 import mobilenetv3
import numpy as np




def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if (k[:len(name)] == name) and (k[len(name)] != '_')}
	return d_filt





class E2Style(nn.Module):

	def __init__(self, opts):
		super(E2Style, self).__init__()
		self.set_opts(opts)
		self.stage = self.opts.training_stage if self.opts.is_training is True else self.opts.stage
		#self.teacher_encoder_firststage = backbone_encoders.BackboneEncoderFirstStage(50, 'ir_se', self.opts)
		self.student_encoder_firststage = backbone_encoders.Backbone_MOBILE_EncoderFirstStage()
		if self.stage > 1:
			self.encoder_refinestage_list = nn.ModuleList([backbone_encoders.BackboneEncoderRefineStage(50, 'ir_se', self.opts) for i in range(self.stage-1)])

		self.decoder = Generator(1024, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		self.load_weights()
		#direction
		
		self.register_buffer('direction_table',self.calc_direction_split(self.decoder, self.opts)) 


	
	#cal_direction:offsets
	@staticmethod
	def calc_direction_split(model, args):#TODO:需要搞清楚

		vectors = []
		for i in range(max(args.mimic_layer)):
			w1 = model.convs[2*i].conv.modulation.weight.data.cpu().numpy()
			w2 = model.convs[2*i+1].conv.modulation.weight.data.cpu().numpy()
			w = np.concatenate((w1,w2), axis=0).T
			w /= np.linalg.norm(w, axis=0, keepdims=True)
			_, eigen_vectors = np.linalg.eig(w.dot(w.T))
			vectors.append(torch.from_numpy(eigen_vectors[:,:5].T))
		return torch.cat(vectors, dim=0)   # (5*L) * 512		

	def load_weights(self):
		if (self.opts.checkpoint_path is not None) and (not self.opts.is_training):   #inference专用
			if self.stage > self.opts.training_stage:
				raise ValueError(f'The stage must be no greater than {self.opts.training_stage} when testing!')
			print(f'Inference: Results are from Stage{self.stage}.', flush=True)
			print('Loading E2Style from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			#self.student_encoder_firststage.load_state_dict(ckpt['state_dict'], strict=True) 
			self.student_encoder_firststage.load_state_dict(get_keys(ckpt, 'student_encoder_firststage'), strict=True)   #注意
			if self.stage > 1:
				for i in range(self.stage-1):
					self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'), strict=True)
			#self.decoder.load_state_dict(ckpt['decoder'], strict=True) 
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		elif (self.opts.checkpoint_path is not None) and self.opts.is_training and (self.opts.distill_mode is False) :  # training distilling pt  or   training refinestage
			print(f'Train: The {self.stage}-th encoder of E2Style is to be trained.', flush=True)
			print('Loading previous encoders and decoder from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.student_encoder_firststage.load_state_dict(ckpt['state_dict'], strict=True)            #student
			if self.stage > 1:
				if self.stage > 2:
					for i in range(self.stage-2):
						self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'), strict=False)
			#ckpt = torch.load(self.opts.stylegan_weights)
			#self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			
			self.decoder.load_state_dict(ckpt['decoder'], strict=True)    
			for i, k in enumerate(ckpt['decoder'].keys(), 1):
				print(f"[DEC {i:03d}] decoder.{k}")
			print(f'Loading the {self.stage}-th encoder weights from irse50!')
			#encoder_ckpt = torch.load(model_paths['ir_se50'])
			#encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			#self.encoder_refinestage_list[self.stage-2].load_state_dict(encoder_ckpt, strict=False)
			self.__load_latent_avg(ckpt)
		elif (self.opts.checkpoint_path is None) and (self.stage==1) and self.opts.is_training:    #training the first_encoder_stage
			print(f'Train: The 1-th encoder of E2Style is to be trained.', flush=True)
			print('no Loading encoders weights from irse50!')
		#	encoder_ckpt = torch.load(model_paths['ir_se50'])
			# if input to encoder is not an RGB image, do not load the input layer weights
		#	if self.opts.label_nc != 0:
		#		encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
		#	self.student_encoder_firststage.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=18)		
		
		elif (self.opts.distill_mode is True) and (self.opts.checkpoint_path is not None):    #Distilling mode
			print("======== DISTILLATION MODE DETECTED ========")	
			print(f"Loading FULL TEACHER MODEL from checkpoint: {self.opts.checkpoint_path}")
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			#first teacher_encoder_stage
			self.teacher_encoder_firststage.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			#loading refinement_stage[0]&refinement_stage[1]
			if self.stage > 1:   #self.stage=3
				for i in range(self.stage-1):
					self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		
		
	def forward(self, x, resize=True, input_code=False, randomize_noise=True, return_latents=False):  #input_code=False 不能遍

		stage_output_list = []
		if input_code:   
			codes = x   
		else:
			codes = self.student_encoder_firststage(x)  


			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)   #W space
				else: 
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)   
		input_is_latent = not input_code  
		first_stage_output, result_latent = self.decoder([codes],input_is_latent=input_is_latent,randomize_noise=randomize_noise,return_latents=return_latents)
		stage_output_list.append(first_stage_output)
		
		if self.stage > 1:
			for i in range(self.stage-1):
				codes = codes + self.encoder_refinestage_list[i](x, self.face_pool(stage_output_list[i]))
				refine_stage_output, result_latent = self.decoder([codes],input_is_latent=input_is_latent,randomize_noise=randomize_noise,return_latents=return_latents)
				stage_output_list.append(refine_stage_output)

		if resize: 
			images = self.face_pool(stage_output_list[-1])

		if return_latents:
			return images, result_latent
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None): 
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None



	def forward_distill(self, x, resize=True, input_code=False, randomize_noise=True, return_latents=False):  #初始版本，只包括 first_stage
		#蒸馏专用
		batch = x.size(0)
		dim = self.direction_table.size(1)
		
		if self.opts.add_offset==True:
			
			if self.opts.offset_mode == 'random':
				offsets = torch.randn(batch, dim, device=x.device)
			else:
				num_direction = self.direction_table.size(0)
				idx = torch.from_numpy(np.random.choice(np.arange(num_direction), size=(batch,), replace=True)).to(x.device)
				offsets = self.direction_table[idx]
				offsets = F.normalize(offsets, dim=1)
				weight  = torch.randn(batch, 1, device=x.device) * self.opts.offset_weight
				offsets = (offsets * weight).unsqueeze(1)  # (B,1,512)
		else:
			offsets = None
		#1.teacher model forward
		with torch.no_grad(): 
			
			if input_code:
				teacher_codes = x
			else:
				first_stage_results = self.teacher_encoder_firststage(x)
				teacher_codes = first_stage_results['latents']
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					teacher_codes = teacher_codes + self.latent_avg.repeat(teacher_codes.shape[0], 1)
				else:
					teacher_codes = teacher_codes + self.latent_avg.repeat(teacher_codes.shape[0], 1, 1)
			input_is_latent = not input_code		
			stage_output_list = []
			first_stage_output,t_rgb_list, t_feat_list,t_latent = self.decoder([teacher_codes], input_is_latent=input_is_latent, randomize_noise=randomize_noise, return_latents=return_latents)
			stage_output_list.append(first_stage_output)
			if self.stage == 1:
				t_FPN = first_stage_results['features']

			elif self.stage > 1:
				B = x.shape[0]
				for i in range(self.stage - 1):
					previous_stage_image = self.face_pool(stage_output_list[i])[:B]
					refinement_results = self.encoder_refinestage_list[i](x, previous_stage_image)
					
					teacher_codes = teacher_codes + refinement_results['latents']
					t_FPN = refinement_results['features']
					refine_stage_output,t_rgb_list,t_feat_list,t_latent = self.decoder([teacher_codes], input_is_latent=input_is_latent,offsets=offsets, randomize_noise=randomize_noise,return_latents=return_latents)
					stage_output_list.append(refine_stage_output)
			

			t_images = stage_output_list[-1]

			if resize: 
				t_images = self.face_pool(t_images)

		
		#2.student model forward
		student_results = self.student_encoder_firststage(x)
		student_codes = student_results['latents']
		s_FPN = student_results['features']
		
		#w+ space
		if self.opts.start_from_latent_avg:
			if self.opts.learn_in_w:
				avg = self.latent_avg.repeat(student_codes.shape[0], 1)
				student_codes = student_codes + avg
			else:
				avg = self.latent_avg.repeat(student_codes.shape[0], 1, 1)
				student_codes = student_codes + avg

		
		#生成image
		s_images,s_rgb_list,s_feat_list,s_latent = self.decoder([student_codes],
									  input_is_latent=True, 
									  offsets=offsets,
									  randomize_noise=randomize_noise,
									  return_latents=return_latents)


		
		if resize: 
			s_images = self.face_pool(s_images)

		
		results = {
            "y_hat": s_images,
            "teacher_images":t_images,
			"teacher_ws": teacher_codes,
            "student_ws": student_codes,
            "teacher_features": t_feat_list,
            "student_features": s_feat_list,
			'teacher_rgb_list' : t_rgb_list,
			'student_rgb_list' : s_rgb_list,
			'teacher_style':t_latent,
			'student_style':s_latent,
			's_FPN':s_FPN,
			't_FPN':t_FPN
        }
		
		return results