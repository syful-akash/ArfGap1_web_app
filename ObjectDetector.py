import cv2 as cv
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import torch
import numpy as np
from PIL import Image
import sys; sys.path.insert(1, "detectron2_repo/projects/PointRend")
import point_rend

#Registering dataset 
from detectron2.data.datasets import register_coco_instances


class Detector:

	def __init__(self):

		register_coco_instances("ArfGAP1_val_coco", {}, "ArfGAP1_val_coco.json", "ArfGAP1")

		# set model and test set
		self.model = 'mask_rcnn_R_50_FPN_3x.yaml'

		# obtain detectron2's default config
		self.cfg = get_cfg() 
		self.cfg.INPUT.MASK_FORMAT="polygon" 

		# load values from a file
		# self.cfg.merge_from_file("test.yaml")
		self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+self.model)) 

		# set device to cpu
		self.cfg.MODEL.DEVICE = "cpu"

		# get weights 
		# self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model) 
		# self.cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"
		self.cfg.MODEL.WEIGHTS = ("best_checkpoint.pth")
	
		self.cfg.DATASETS.TEST = ("ArfGAP1_val_coco",)


		point_rend.add_pointrend_config(self.cfg)
		self.cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
		# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
		self.cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"

		# set the testing threshold for this model
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
		self.cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2  

		# build model from weights
		self.cfg.MODEL.WEIGHTS = self.convert_model_for_inference()

	# build model and convert for inference
	def convert_model_for_inference(self):

		# build model
		model = build_model(self.cfg)

		# save as checkpoint
		# torch.save(model.state_dict(), 'best_checkpoint.pth')

		# return path to inference model
		return 'best_checkpoint.pth'

	# detectron model
	# adapted from detectron2 colab notebook: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5	
	
	def inference(self, file):

		predictor = DefaultPredictor(self.cfg)
		im = cv.imread(file)
		outputs = predictor(im)
		#print(str(len(outputs["instances"])))
		
		num = str(len(outputs["instances"])) 
		

		# with open(self.curr_dir+'/data.txt', 'w') as fp:
		# 	json.dump(outputs['instances'], fp)
		# 	# json.dump(cfg.dump(), fp)

		# get metadata
		# metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
		ArfGAP1_metadata = MetadataCatalog.get("ArfGAP1_val_coco")
		dataset_dicts = DatasetCatalog.get("ArfGAP1_val_coco")

		# visualise
		v = Visualizer(im[:, :, ::-1], metadata=ArfGAP1_metadata, scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

		# get image 
		img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))

		# write to jpg
		# cv.imwrite('img.jpg',v.get_image())

		return img, num



