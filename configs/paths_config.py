dataset_paths = {
	'celeba_train': 'data/celeba_hq/train',
	'celeba_test': 'data/celeba_hq/val',     
	'celeba_train_4seg': 'data/celeba_hq/train',
	'celeba_test_4seg': 'data/celeba_hq/val',	
	'celeba_train_sketch': 'data/celeba_hq/train',
	'celeba_test_sketch': 'data/celeba_hq/val',
	'celeba_train_segmentation': 'data/CelebAMask-HQ',
	'celeba_test_segmentation': 'data/CelebAMask-HQ',
	'ffhq': 'data/ffhq',
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'parsing_net': 'pretrained_models/parsing.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat'
}
