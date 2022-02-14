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

from ikomia import core, dataprocess
import copy
# Your imports below
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


    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.conf_thres = float(param_map["conf_thres"])
        self.cuda = eval(param_map["cuda"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["model_name"] = self.model_name
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["cuda"] = str(self.cuda)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDetectron2PanopticSegmentation(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.predictor = None
        self.cfg = None
        self.colors = None
        self.setOutputDataType(core.IODataType.IMAGE_LABEL, 0)
        self.addOutput(dataprocess.CImageIO())
        self.addOutput(dataprocess.CNumericIO())

        # Create parameters class
        if param is None:
            self.setParam(InferDetectron2PanopticSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        self.forwardInputImage(0,1)

        # Get parameters :
        param = self.getParam()
        if self.predictor is None or param.update:
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(param.model_name + '.yaml'))
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param.conf_thres
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(param.model_name + '.yaml')
            self.class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
            self.colors = np.array(np.random.randint(0, 255, (len(self.class_names), 3)))
            self.colors = [[int(c[0]), int(c[1]), int(c[2])] for c in self.colors]
            self.cfg.MODEL.DEVICE = 'cuda' if param.cuda else 'cpu'
            self.predictor = DefaultPredictor(self.cfg)
            legend = self.getOutput(2)
            legend.addValueList(list(range(1,len(self.class_names)+1)), "Class value", self.class_names)
            param.update = False
            print("Inference will run on " + ('cuda' if param.cuda else 'cpu'))

        # Get input :
        input = self.getInput(0)


        if input.isDataAvailable():
            img = input.getImage()
            output_mask = self.getOutput(0)
            out = self.infer(img)
            output_mask.setImage(out)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def infer(self, img):
        outputs = self.predictor(img)
        h,w,c = np.shape(img)
        panoptic_seg = torch.full((h, w), fill_value=-1)

        if "instances" in outputs.keys():
            instances = outputs["instances"].to("cpu")
            scores = instances.scores
            classes = instances.pred_classes
            masks = instances.pred_masks
            for score, cls, mask in zip(scores, classes, masks):
                if score>= self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                    panoptic_seg[mask] = cls+1
        return panoptic_seg.cpu().numpy()



# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDetectron2PanopticSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_panoptic_segmentation"
        self.info.shortDescription = "your short description"
        self.info.description = "your description"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "algorithm author"
        self.info.article = "title of associated research article"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentationLink = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "your,keywords,here"

    def create(self, param=None):
        # Create process object
        return InferDetectron2PanopticSegmentation(self.info.name, param)
