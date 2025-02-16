# Copyright 2021 Google LLC
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

import os

root = os.path.join(os.path.dirname(__file__), '..')

depth_pretrain_path = os.path.join(root, './pretrained_depth_ckpt/best_depth_Ours_Bilinear_inc_3_net_G.pth')
midas_pretrain_path = os.path.join(root, './pretrained_depth_ckpt/midas_cpkt.pt')
