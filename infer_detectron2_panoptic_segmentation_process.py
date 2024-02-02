# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from infer_detectron2_panoptic_segmentation import update_path
from ikomia import utils, core, dataprocess
import copy
import os
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import numpy as np
import torch


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDetectron2PanopticSegmentationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x"
        self.conf_thres = 0.5
        self.cuda = True if torch.cuda.is_available() else False
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.conf_thres = float(param_map["conf_thres"])
        self.cuda = eval(param_map["cuda"])
        self.update = utils.strtobool(param_map["update"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = self.model_name
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["cuda"] = str(self.cuda)
        param_map["update"] = str(self.update)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDetectron2PanopticSegmentation(dataprocess.CInstanceSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)
        self.predictor = None
        self.cfg = None
        self.colors = None
        self.stuff_classes = None
        self.thing_classes = None
        self.class_names = None

        # Create parameters class
        if param is None:
            self.set_param_object(InferDetectron2PanopticSegmentationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        self.forward_input_image(0, 0)

        # Get parameters :
        param = self.get_param_object()

        # Set cache dir in the algorithm folder to simplify deployment
        os.environ["FVCORE_CACHE"] = os.path.join(os.path.dirname(__file__), "models")

        if self.predictor is None or param.update:
            np.random.seed(10)
            self.cfg = get_cfg()
            dataset_name, config_name = param.model_name.replace(os.path.sep, '/').split('/') 
            config_path = os.path.join(os.path.dirname(detectron2.__file__), 
                                       "model_zoo", "configs", dataset_name, config_name + '.yaml')
            self.cfg.merge_from_file(config_path)
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param.conf_thres
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url((param.model_name + '.yaml').replace('\\', '/'))
            self.stuff_classes = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("stuff_classes")
            self.thing_classes = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
            self.class_names = self.thing_classes + self.stuff_classes
            self.set_names(self.class_names)
            self.cfg.MODEL.DEVICE = 'cuda' if param.cuda and torch.cuda.is_available() else 'cpu'
            self.predictor = DefaultPredictor(self.cfg)
            param.update = False
            print("Inference will run on " + ('cuda' if param.cuda and torch.cuda.is_available() else 'cpu'))

        # Get input :
        img_input = self.get_input(0)
        if img_input.is_data_available():
            img = img_input.get_image()
            h, w, _ = img.shape
            # Get output :
            instance_output = self.get_output(1)
            self.infer(img, instance_output)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

    def infer(self, img, instance_output):
        import cv2
        outputs = self.predictor(img)

        if "panoptic_seg" in outputs.keys():
            masks, infos = outputs["panoptic_seg"]

            # reverse indexing to put stronger confidences foreground
            index = 0
            for info in infos:
                offset = len(self.thing_classes) if not info["isthing"] else 0
                px_value = info["id"]
                cat_value = info["category_id"] + offset
                bool_mask = (masks == px_value).cpu().numpy()
                y, x = np.median(bool_mask.nonzero(), axis=1)
                obj_type = 0 if info["isthing"] else 1
                self.add_object(index, obj_type, cat_value, 1.0, float(x), float(y), 0.0, 0.0,
                                bool_mask.astype("uint8"))
                index += 1


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDetectron2PanopticSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_panoptic_segmentation"
        self.info.short_description = "Infer Detectron2 panoptic segmentation models"
        self.info.description = "Infer Detectron2 panoptic segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.2.2"
        self.info.icon_path = "icons/detectron2.png"
        self.info.authors = "Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick"
        self.info.article = "Detectron2"
        self.info.journal = ""
        self.info.year = 2019
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://detectron2.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/facebookresearch/detectron2"
        # Keywords used for search
        self.info.keywords = "infer, detectron2, panoptic, semantic, segmentation"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "PANOPTIC_SEGMENTATION"

    def create(self, param=None):
        # Create process object
        return InferDetectron2PanopticSegmentation(self.info.name, param)
