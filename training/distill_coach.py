import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from utils.content_aware_pruning import Get_Parsing_Net, Batch_Img_Parsing, Get_Masked_Tensor
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from criteria.parsing_loss import parse_loss
from utils import common, train_utils,network_util
from criteria import id_loss, w_norm
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.e2style import E2Style
from training.ranger import Ranger
import torchvision.transforms as transforms
from criteria.lpips.lpips import LPIPS


class DistillCoach:
    """
    蒸馏
    - 优化器只针对学生和适配层。
    - 损失函数只包含 latent loss & feature loss。
    - 不使用任何 ground truth (y)。
    """
    def __init__(self, opts):
        self.opts = opts
        self.warmup_steps = getattr(self.opts, 'warmup_steps', 5000)
        self.global_step = 0
        self.device = 'cuda:0'
        self.opts.device = self.device
    #初始化网络 
        self.net = E2Style(self.opts).to(self.device)


    #优化器
        self.optimizer = self.configure_optimizers()   
    #数据集    
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
    #log&checkpoint
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
    

    def configure_optimizers(self):
        
        params_to_train = list(self.net.student_encoder_firststage.parameters()) 
                     
        self.net.teacher_encoder_firststage.requires_grad_(False)
        self.net.decoder.requires_grad_(False)
        if hasattr(self.net, 'encoder_refinestage_list'):
            for refine_encoder in self.net.encoder_refinestage_list:
                refine_encoder.requires_grad_(False)
        
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params_to_train, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params_to_train, lr=self.opts.learning_rate ,weight_decay=1e-4)
        
        return optimizer
    
    def train(self):
        # 明确分开控制三部分
        self.net.teacher_encoder_firststage.eval() 
        self.net.decoder.eval()          
        if hasattr(self.net, 'encoder_refinestage_list'):
            self.net.encoder_refinestage_list.eval()
        self.net.student_encoder_firststage.train()   
                
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x, _ = batch 
                x = x.to(self.device).float()
                #teacher&student output
                results_dict = self.net.forward_distill(x, return_latents=True)
                #loss_function
                loss, loss_dict = self.calc_loss(results_dict) 
                #backward
                loss.backward()
                self.optimizer.step()

                if self.global_step % self.opts.board_interval == 0:
                    if self.global_step < self.warmup_steps:
                        self.print_metrics(loss_dict, prefix=f'train (Warm-up, {self.global_step}/{self.warmup_steps} steps)')
                    else:
                        self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')
                
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 500 == 0):  
                    self.parse_and_log_images(None, x, results_dict['teacher_images'], results_dict['y_hat'], title='images/train/faces')
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    self.validate()
                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    self.checkpoint_me(loss_dict, is_best=False)  
                if self.global_step == self.opts.max_steps:
                    print("蒸馏完成goooooo!")
                    break
                self.global_step += 1
    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, (x, _) in enumerate(self.test_dataloader):
            if batch_idx % 100 == 0:
                print(f'[Validate] Now processing batch {batch_idx}...',flush=True)
            with torch.no_grad():
                x = x.to(self.device).float()
                results_dict = self.net.forward_distill(x, return_latents=True)
                loss, cur_loss_dict  = self.calc_loss(results_dict)
            agg_loss_dict.append(cur_loss_dict)
            self.parse_and_log_images(None, x, results_dict['teacher_images'],
                                       results_dict['y_hat'],
                                     title='images/test/faces')
        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')
			
            


			# For first step just do sanity test on small amount of data

        
        self.net.student_encoder_firststage.train()

        return loss_dict  # Do not log, inaccurate in first batch

    def calc_loss(self, results_dict: dict):
        #(Latent Loss + Feature Loss)
        teacher_ws = results_dict['teacher_ws']              #encoder_latent loss
        student_ws = results_dict['student_ws']
        t_features = results_dict['teacher_features']    #relation style loss
        s_features = results_dict['student_features']
        t_rgb_list = results_dict['teacher_rgb_list']       #kd_l1_loss
        s_rgb_list = results_dict['student_rgb_list']
        s_images = results_dict["y_hat"]
        t_images = results_dict["teacher_images"]
        s_FPN = results_dict['s_FPN']
        t_FPN = results_dict['t_FPN']
        
        
        loss_dict = {}
        total_loss = torch.tensor(0., device=self.device)
        is_warmup_phase = self.global_step < self.warmup_steps
        # KDloss1: Latent Loss
               
        if self.opts.latent_distill_lambda > 0:
          
            loss_latent =  F.l1_loss(student_ws, teacher_ws)  # 在传入之前已经concat对齐
            loss_dict['loss_latent'] = float(loss_latent)    
            total_loss += loss_latent * self.opts.latent_distill_lambda
        
            # loss2: Feature Loss 
        if self.opts.feature_distill_lambda > 0 and len(s_FPN):
            loss_feat = torch.tensor(0., device=self.device)
            for s, t in zip(s_FPN, t_FPN):
                    #print(f"student_features[{i}]:", student_features[i].shape)
                    #print(f"teacher_features[{i}]:", teacher_features[i].shape)
                    #assert student_features[i].shape == teacher_features[i].shape, f"Feature shape mismatch at {i}"
                loss_feat += F.mse_loss(s, t)
            loss_dict['loss_feature'] = float(loss_feat)
            total_loss += loss_feat * self.opts.feature_distill_lambda
        
        
        
        # KDloss2:content_aware_l1loss    for fake_img and fake_img_teacher
        # fake_img_teacher = t_rgb_list    
        B = student_ws.shape[0]    

        if not is_warmup_phase and self.opts.kd_l1_lambda > 0:
            fake_img_teacher = t_rgb_list[-1][:B]
            fake_img = s_rgb_list[-1][:B]
            parsing_net, _ = Get_Parsing_Net(self.device)
            if parsing_net is not None: 
                with torch.no_grad():
                    teacher_img_parsing = Batch_Img_Parsing(fake_img_teacher, parsing_net, self.device)
                fake_img_teacher = Get_Masked_Tensor(fake_img_teacher, teacher_img_parsing, self.device, mask_grad=False)
                fake_img = Get_Masked_Tensor(fake_img, teacher_img_parsing, self.device, mask_grad=True)
            if self.opts.kd_mode == 'Output_Only':
                kd_l1_loss = torch.mean(torch.abs(fake_img_teacher - fake_img))
            elif self.opts.kd_mode == 'Intermediate':
                loss_list = [torch.mean(torch.abs(fake_img_teacher[:B] - fake_img[:B])) for (fake_img_teacher, fake_img) in zip(t_rgb_list, s_rgb_list)]
                kd_l1_loss = sum(loss_list)
            loss_dict['kd_l1_loss'] = float(kd_l1_loss)
            total_loss += kd_l1_loss*self.opts.kd_l1_lambda

            #KDloss3:lpips
        if not is_warmup_phase and self.opts.kd_lpips_lambda > 0:
            lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
            kd_loss_lpips = lpips_loss(s_images[:B], t_images[:B])
            loss_dict['loss_lpips'] = float(kd_loss_lpips)
            total_loss += kd_loss_lpips * self.opts.kd_lpips_lambda

            #kdloss4:relation loss
        if not is_warmup_phase and self.opts.kd_simi_lambda > 0:
            kd_simi_loss = torch.tensor(0., device=self.device)
            
            for i in self.opts.mimic_layer:
                f1 = s_features[i-1][:B]
                f2 = s_features[i-1][B:] 
                s_simi = F.cosine_similarity(f1[:,None,:], f2[None,:,:], dim=2)
                f1 = t_features[i-1][:B]
                f2 = t_features[i-1][B:]
                t_simi = F.cosine_similarity(f1[:,None,:], f2[None,:,:], dim=2)
                if self.opts.simi_loss == 'mse':
                    kd_simi_loss += F.mse_loss(s_simi, t_simi)
                elif self.opts.simi_loss == 'kl':
                    s_simi = F.log_softmax(s_simi, dim=1)
                    t_simi = F.softmax(t_simi, dim=1)
                    kd_simi_loss += F.kl_div(s_simi, t_simi, reduction='batchmean')
                    loss_dict['kd_simi_loss'] = float(kd_simi_loss)
                    total_loss+=kd_simi_loss*self.opts.kd_simi_lambda
            
        loss_dict['loss'] = float(total_loss)    
        return total_loss, loss_dict
    

#以下与原来几乎一致
# In training/distill_coach.py

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS:
            raise Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        print(f'Loading dataset for {self.opts.dataset_type}')
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()

        train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                    target_root=dataset_args['train_target_root'],
                                    source_transform = transforms_dict['transform_source'],
                                    target_transform = transforms_dict['transform_gt_train'],
                                    opts=self.opts)  # Add this line

        test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                    target_root=dataset_args['test_target_root'],
                                    source_transform=transforms_dict['transform_source'],
                                    target_transform=transforms_dict['transform_test'],
                                    opts=self.opts)  # And add this line

        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset
    
    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'latest_model.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
            
    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = {value}')
            
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
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        
    def __get_save_dict(self):
        # 只保存学生网络和优化器的状态
        save_dict = {
            'state_dict': self.net.student_encoder_firststage.state_dict(),
            'decoder': self.net.decoder.state_dict(),
            'opts': vars(self.opts),
            'global_step': self.global_step,
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        return save_dict
        
        
        
        
