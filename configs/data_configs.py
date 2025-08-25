from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'ffhq_colorization': {
		'transforms': transforms_config.RestorationTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],  #记得修改  全集
		'test_target_root': dataset_paths['celeba_test'],		#记得修改  全集
	},
	'ffhq_denoise': {
		'transforms': transforms_config.RestorationTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],		#记得修改  全集
		'test_target_root': dataset_paths['celeba_test'],		#记得修改  全集
	},
	'ffhq_inpainting': {
		'transforms': transforms_config.RestorationTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],    #记得修改  全集
		'test_target_root': dataset_paths['celeba_test'],    #记得修改  全集
	},
	'celebs_sketch_to_face': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_sketch'],  #celeba_hq generate sketch
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test_sketch'],
		'test_target_root': dataset_paths['celeba_test'],    #记得修改  仅val
	},
	'celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_segmentation'],    #celeba_hq_mask
		'train_target_root': dataset_paths['celeba_train_4seg'],      
		'test_source_root': dataset_paths['celeba_test_segmentation'],          
		'test_target_root': dataset_paths['celeba_test_4seg'],  
	},
	'celebs_super_resolution': {
		'transforms': transforms_config.SuperResTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
}
