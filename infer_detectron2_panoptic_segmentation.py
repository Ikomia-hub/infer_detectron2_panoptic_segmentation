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

from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        from infer_detectron2_panoptic_segmentation.infer_detectron2_panoptic_segmentation_process import InferDetectron2PanopticSegmentationFactory
        return InferDetectron2PanopticSegmentationFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        from infer_detectron2_panoptic_segmentation.infer_detectron2_panoptic_segmentation_widget import InferDetectron2PanopticSegmentationWidgetFactory
        return InferDetectron2PanopticSegmentationWidgetFactory()