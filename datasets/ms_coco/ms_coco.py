# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""


import json
from collections import OrderedDict
from pathlib import Path

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_LABELS_PATH = Path(__file__).parent / "coco-labels.txt"


def create_image_map_from_task(task_json_path, task_name=None):
    if task_name is None:
        task_name = "_".join(task_json_path.name.split("_")[:-1])

    task_dict = json.load(open(task_json_path))

    image_map = {}

    for image_info in task_dict["images"]:
        image_map[image_info["id"]] = {
            "image_id": image_info["id"],
            "file_name": image_info["file_name"],
            task_name: [],
        }

    for obj in task_dict["annotations"]:
        image_map[obj["image_id"]][task_name].append(obj)

    return OrderedDict(sorted(image_map.items())).values()


class MsCocoConfig(datasets.BuilderConfig):
    def __init__(
        self,
        year: int = 2014,
        with_caption=True,
    ):
        self.year = year

        _tasks_mask = zip(  # TODO: add support to more tasks
            ["captions", "instances"],
            [with_caption, True],
        )

        self.tasks = [task_name for task_name, tasks_bool in _tasks_mask if tasks_bool]

        self.name = "_".join(self.tasks[:-1]) + f"{year}"
        self.version = datasets.Version("1.1.0")
        self.description = ""  # TODO


class MsCoco(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        MsCocoConfig(2014, True),
        MsCocoConfig(2014, False),
        MsCocoConfig(2017, True),
        MsCocoConfig(2017, False),
    ]

    DEFAULT_CONFIG_NAME = "captions_2014"

    def _info(self):
        features = {
            "image_id": datasets.Value("int64"),
            "file_name": datasets.Value("string"),
        }

        if "instances" in self.config.tasks:
            features["instances"] = datasets.Sequence(
                {
                    "id": datasets.Value("int64"),
                    "bbox": datasets.Sequence(datasets.Value("float32")),
                    "category_id": datasets.ClassLabel(
                        num_classes=91,
                        names=[
                            "person",
                            "bicycle",
                            "car",
                            "motorcycle",
                            "airplane",
                            "bus",
                            "train",
                            "truck",
                            "boat",
                            "traffic light",
                            "fire hydrant",
                            "street sign",
                            "stop sign",
                            "parking meter",
                            "bench",
                            "bird",
                            "cat",
                            "dog",
                            "horse",
                            "sheep",
                            "cow",
                            "elephant",
                            "bear",
                            "zebra",
                            "giraffe",
                            "hat",
                            "backpack",
                            "umbrella",
                            "shoe",
                            "eye glasses",
                            "handbag",
                            "tie",
                            "suitcase",
                            "frisbee",
                            "skis",
                            "snowboard",
                            "sports ball",
                            "kite",
                            "baseball bat",
                            "baseball glove",
                            "skateboard",
                            "surfboard",
                            "tennis racket",
                            "bottle",
                            "plate",
                            "wine glass",
                            "cup",
                            "fork",
                            "knife",
                            "spoon",
                            "bowl",
                            "banana",
                            "apple",
                            "sandwich",
                            "orange",
                            "broccoli",
                            "carrot",
                            "hot dog",
                            "pizza",
                            "donut",
                            "cake",
                            "chair",
                            "couch",
                            "potted plant",
                            "bed",
                            "mirror",
                            "dining table",
                            "window",
                            "desk",
                            "toilet",
                            "door",
                            "tv",
                            "laptop",
                            "mouse",
                            "remote",
                            "keyboard",
                            "cell phone",
                            "microwave",
                            "oven",
                            "toaster",
                            "sink",
                            "refrigerator",
                            "blender",
                            "book",
                            "clock",
                            "vase",
                            "scissors",
                            "teddy bear",
                            "hair drier",
                            "toothbrush",
                            "hair brush",
                        ],
                    ),
                }
            )

        if "captions" in self.config.tasks:
            features["captions"] = datasets.Sequence(
                {"id": datasets.Value("int64"), "caption": datasets.Value("string")}
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = {
            "images": {
                "train": f"http://images.cocodataset.org/zips/train{self.config.year}.zip",
                "dev": f"http://images.cocodataset.org/zips/val{self.config.year}.zip",
                "test": f"http://images.cocodataset.org/zips/test{self.config.year}.zip",
            },
            "annotations": f"http://images.cocodataset.org/annotations/annotations_trainval{self.config.year}.zip",
            "test_image_info": f"http://images.cocodataset.org/annotations/image_info_test{self.config.year}.zip",
        }

        data_dir = dl_manager.download_and_extract(my_urls)

        annotation_path = Path(data_dir["annotations"]) / "annotations"
        train_annon = [annotation_path / f"{task}_train{self.config.year}.json" for task in self.config.tasks]

        image_train_path = Path(data_dir["images"]["train"]) / f"train{self.config.year}"

        val_annon = [annotation_path / f"{task}_val{self.config.year}.json" for task in self.config.tasks]

        image_val_path = Path(data_dir["images"]["dev"]) / f"val{self.config.year}"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_json": train_annon,
                    "image_path": image_train_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_json": val_annon,
                    "image_path": image_val_path,
                },
            ),
        ]

    def _generate_examples(
        self,
        source_json,
        image_path,
    ):
        """Yields examples as (key, example) tuples."""

        source = zip(*map(create_image_map_from_task, source_json))

        for _id, obj in enumerate(source):
            obj_dict = dict(zip(self.config.tasks, obj))

            output_dict = obj_dict["instances"]
            output_dict["file_name"] = str(image_path / output_dict["file_name"])

            for inst in output_dict["instances"]:
                inst.pop("iscrowd", None)
                inst.pop("segmentation", None)
                inst.pop("area", None)
                inst.pop("image_id", None)

            if "captions" in obj_dict:
                captions = obj_dict["captions"]["captions"]
                for capt in captions:
                    capt.pop("image_id")

                output_dict["captions"] = captions

            yield _id, output_dict