# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
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

import gc
import os
from pathlib import Path

import torch
import yaml


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    # mmap reduces peak RAM when reading large .pth files on Windows
    try:
        checkpoint = torch.load(model_pth, map_location='cpu', weights_only=False, mmap=True)
    except TypeError:
        checkpoint = torch.load(model_pth, map_location='cpu', weights_only=False)
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    gc.collect()
    info_path = str(Path(model_pth).with_suffix('.yaml'))
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs
