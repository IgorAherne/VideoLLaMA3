# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import logging
logger = logging.getLogger(__name__)

import einops
import torch
import torch.distributed as dist
import torch.nn as nn

from ..constants import IGNORE_INDEX, MODAL_INDEX_MAP, NUM_FRAMES
from .encoder import build_vision_encoder
from .projector import build_vision_projector, load_mm_projector


def spatial_downsampling(features, grid_thws, stride=2):
    n, c = features.shape

    flatten_grid_thws = torch.cat([grid_thw for batch_grid_thws in grid_thws for grid_thw in batch_grid_thws])
    split_sizes = [grid_thw.prod() for grid_thw in flatten_grid_thws]
    features = torch.split(features, split_sizes)

    new_features = []
    for feature, grid_thw in zip(features, flatten_grid_thws):
        # NOTE: adapted for reshape in image processor 
        feature = feature.view(grid_thw[0], grid_thw[1] // stride, grid_thw[2] // stride, stride, stride,  c).permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(grid_thw[0], grid_thw[1], grid_thw[2], c).permute(0, 3, 1, 2)
        # NOTE: previous version model is align_corners=True
        new_feature = torch.nn.functional.interpolate(feature, (math.ceil(grid_thw[1] / stride), math.ceil(grid_thw[2] / stride)), mode='bilinear')
        # new_feature = nn.functional.avg_pool2d(feature, stride)
        # new_feature = nn.functional.max_pool2d(feature, stride)
        new_features.append(new_feature.permute(0, 2, 3, 1).view(-1, c))
    new_features = torch.cat(new_features)

    return new_features


class Videollama3MetaModel:

    def __init__(self, config):
        super(Videollama3MetaModel, self).__init__(config)

        if hasattr(config, "vision_encoder") or hasattr(config, "mm_vision_encoder"):
            self.vision_encoder = build_vision_encoder(config, delay_load=False)
            self.mm_projector = build_vision_projector(config, self.vision_encoder.hidden_size)

    def get_vision_encoder(self):
        vision_encoder = getattr(self, 'vision_encoder', None)
        if type(vision_encoder) is list:
            vision_encoder = vision_encoder[0]
        return vision_encoder

    def get_mm_projector(self):
        return self.mm_projector

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_encoder = model_args.vision_encoder
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_projector = model_args.pretrain_mm_projector

        self.config.mm_vision_encoder = vision_encoder

        if self.get_vision_encoder() is None:
            vision_encoder = build_vision_encoder(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_encoder = [vision_encoder]
            else:
                self.vision_encoder = vision_encoder
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_encoder = self.vision_encoder[0]
            else:
                vision_encoder = self.vision_encoder
            # NOTE: only compatible with delay_load encoder
            # vision_encoder.load_model(vision_encoder.cfg_only)

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_encoder.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_projector is not None:
            if os.path.exists(pretrain_mm_projector):
                is_local = True
                if os.path.isdir(pretrain_mm_projector):
                    mm_projector_weights = load_mm_projector(pretrain_mm_projector)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_projector, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_projector = pretrain_mm_projector.replace('mm_projector.bin', '')
                pretrain_mm_projector = pretrain_mm_projector.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_projector)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


class Videollama3MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def encode_images(
        self,
        pixel_values: torch.FloatTensor,    # Shape: (SUM_OF_ALL_PATCHES_IN_BATCH, patch_feature_dim)
        grid_sizes: torch.LongTensor,       # Shape: (B, 3), where B is num visual inputs, each row (t_frames, h_grid, w_grid)
        merge_sizes: torch.LongTensor,      # Shape: (B)
        vision_chunk_size: Optional[int] = None, # New parameter
    ) -> torch.FloatTensor:
        
        all_projected_features_for_batch_items = [] # To store final features for each item in the batch

        # num_patches_per_item[i] = t_i * h_grid_i * w_grid_i
        # This is the number of patches *before* any merging/pooling by the vision encoder's final stages.
        num_patches_per_item = grid_sizes.prod(dim=1) 
        
        patch_offsets = torch.zeros(num_patches_per_item.size(0) + 1, dtype=torch.long, device=pixel_values.device)
        patch_offsets[1:] = num_patches_per_item.cumsum(0)

        # Iterate over each visual input in the batch
        for i in range(grid_sizes.size(0)): # grid_sizes.size(0) is B (batch size of visual inputs)
            # Get patches for the current item
            item_pixel_values = pixel_values[patch_offsets[i] : patch_offsets[i+1]]
            # Get grid and merge sizes for the current item (retain batch dim of 1 for encoder/projector calls)
            item_grid_sizes = grid_sizes[i:i+1]    # Shape: (1, 3)
            item_merge_sizes = merge_sizes[i:i+1]  # Shape: (1)

            T_total_frames = item_grid_sizes[0, 0].item()
            H_grid = item_grid_sizes[0, 1].item()
            W_grid = item_grid_sizes[0, 2].item()
            patches_per_frame = H_grid * W_grid

            # To store features from processed chunks of the current item
            item_projected_features_list_of_chunks = []

            # Determine if chunking should be applied for this item
            use_chunking_for_this_item = (
                T_total_frames > 1 # Only chunk if it's a video with multiple frames
                and vision_chunk_size is not None 
                and vision_chunk_size > 0 
                and vision_chunk_size < T_total_frames
            )

            if not use_chunking_for_this_item:
                logger.debug(f"Item {i}: Processing all {T_total_frames} frames at once.")
                # Process all frames of this item at once
                encoded_features = self.get_model().get_vision_encoder()(
                    pixel_values=item_pixel_values,
                    grid_sizes=item_grid_sizes,
                    merge_sizes=item_merge_sizes,
                )
                projected_features = self.get_model().mm_projector(encoded_features)
                item_projected_features_list_of_chunks.append(projected_features)
            else:
                logger.debug(f"Item {i}: Processing {T_total_frames} frames in chunks of size {vision_chunk_size}.")
                # Process in chunks for this item
                for frame_chunk_start_idx_in_item in range(0, T_total_frames, vision_chunk_size):
                    frame_chunk_end_idx_in_item = min(frame_chunk_start_idx_in_item + vision_chunk_size, T_total_frames)
                    num_frames_in_current_chunk = frame_chunk_end_idx_in_item - frame_chunk_start_idx_in_item

                    # Determine patch indices for the current chunk of frames
                    patch_chunk_start_idx = frame_chunk_start_idx_in_item * patches_per_frame
                    patch_chunk_end_idx = frame_chunk_end_idx_in_item * patches_per_frame
                    
                    pixel_values_for_chunk = item_pixel_values[patch_chunk_start_idx : patch_chunk_end_idx]
                    
                    # grid_sizes for this chunk needs to reflect the number of frames in *this* chunk
                    grid_sizes_for_chunk = torch.tensor(
                        [[num_frames_in_current_chunk, H_grid, W_grid]], 
                        device=item_grid_sizes.device, 
                        dtype=item_grid_sizes.dtype
                    )
                    # item_merge_sizes is already (1), and applies to the whole item processing logic
                    # within the vision encoder which expects it per-item.

                    logger.debug(f"  Item {i}, Chunk: {frame_chunk_start_idx_in_item // vision_chunk_size + 1}, Frames in chunk: {num_frames_in_current_chunk}")

                    encoded_chunk_features = self.get_model().get_vision_encoder()(
                        pixel_values=pixel_values_for_chunk,
                        grid_sizes=grid_sizes_for_chunk,
                        merge_sizes=item_merge_sizes, 
                    )
                    projected_chunk_features = self.get_model().mm_projector(encoded_chunk_features)
                    item_projected_features_list_of_chunks.append(projected_chunk_features)
                    
                    # Optional: Aggressive VRAM cleanup if needed, can add latency
                    # del pixel_values_for_chunk, grid_sizes_for_chunk, encoded_chunk_features, projected_chunk_features
                    # if torch.cuda.is_available(): torch.cuda.empty_cache()
                    # import gc; gc.collect()
            
            # Concatenate features from all chunks for the current item
            # Each chunk's projected_features has shape (num_patches_in_chunk_after_merge, llm_hidden_size)
            concatenated_features_for_item = torch.cat(item_projected_features_list_of_chunks, dim=0)
            all_projected_features_for_batch_items.append(concatenated_features_for_item)

        # Concatenate features from all items in the batch
        # Final mm_features should have shape (TOTAL_PATCHES_IN_BATCH_AFTER_MERGE_AND_PROJECTION, llm_hidden_size)
        final_mm_features = torch.cat(all_projected_features_for_batch_items, dim=0)
        return final_mm_features

    def _get_valid_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
    ):
        valid_masks = []
        for num_patches, modal in zip(batched_num_patches, modals):
            valid_mask = torch.full((num_patches, ), modal != "text", dtype=torch.bool, device=mm_features.device)
            valid_masks.append(valid_mask)
        mm_features = mm_features[torch.cat(valid_masks)]
        return mm_features

    def _maybe_truncate_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        compression_mask: torch.BoolTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        if position_ids is None or mm_features.shape[0] == input_ids.eq(self.config.image_token_index).sum():
            return mm_features, compression_mask

        truncation_mask = []
        for num_patches, modal in zip(batched_num_patches, modals):
            if modal == "text":
                truncation_mask.append(torch.ones((0,), dtype=torch.bool, device=input_ids.device))
            else:
                truncation_mask.append(torch.ones((num_patches,), dtype=torch.bool, device=input_ids.device))

        seq_end_indices = torch.nonzero(position_ids == 0)[:, 0]
        seq_end_indices = seq_end_indices[seq_end_indices > 0].tolist()+ [len(input_ids)]
        seq_start_indices = [0] + seq_end_indices[:-1]
        num_visual_tokens = [
            input_ids[start:end].eq(self.config.image_token_index).sum()
            for start, end in zip(seq_start_indices, seq_end_indices)
        ]

        for n, mask in zip(num_visual_tokens, truncation_mask):
            if len(mask) > 0:
                mask[n:] = False
        truncation_mask = torch.cat(truncation_mask)

        return mm_features[truncation_mask], compression_mask[truncation_mask]

    def _get_compression_mask(
        self,
        pixel_values: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        modals: List[str],
        threshold: float = 0.1,
        min_tokens: int = 1,
    ) -> torch.BoolTensor:
        batched_images = pixel_values.split(grid_sizes.prod(dim=1).tolist(), dim=0)
        compression_masks = []

        for images, num_patches, grid_size, merge_size, modal in zip(
            batched_images, batched_num_patches, grid_sizes, merge_sizes, modals
        ):
            t, h, w = grid_size
            if modal == "image" or (modal == "video" and t == 1):
                compression_masks.append(torch.ones((num_patches,), dtype=torch.bool, device=images.device))

            elif modal == "video":
                # NOTE: video token compressor
                images = images.view(t, (h // merge_size) * (w // merge_size), -1)

                pixel_diff = images[1:] - images[:-1]
                pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                mask = pixel_diff > threshold
                padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                # mask[padding_ids, torch.randperm(min_tokens)] = 1
                mask[padding_ids, :min_tokens] = 1
                compression_masks.append(mask.flatten())

            else:
                # in case of psuedo image
                compression_masks.append(torch.ones((0,), dtype=torch.bool, device=images.device))

        return torch.cat(compression_masks)

    def _compress_visual_tokens(
        self,
        compression_mask: torch.BoolTensor,
        mm_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        mm_features = mm_features[compression_mask]
        image_selected = (input_ids == self.config.image_token_index)

        text_masks = torch.logical_not(image_selected)
        text_masks[image_selected] = compression_mask
        input_ids = input_ids[text_masks]

        if attention_mask is not None:
            attention_mask = attention_mask[text_masks]
        if labels is not None:
            labels = labels[text_masks]
        if position_ids is not None:
            # FIXME: assume the first position_id is always 0
            position_ids = position_ids[text_masks]
            pos_start = [0] + torch.nonzero(position_ids == 0)[:, 0].tolist()
            pos_end = pos_start[1:] + [len(input_ids)]
            position_ids = torch.cat([torch.arange(end - start, device=input_ids.device) for start, end in zip(pos_start, pos_end)])

        return mm_features, input_ids, attention_mask, position_ids, labels

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        vision_chunk_size: Optional[int] = None,
    ):
        vision_encoder = self.get_vision_encoder()
        # NOTE: text-only situation
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels

        # 1. flatten text inputs
        B, N = input_ids.shape
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # 2. embed visual tokens
        batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()
        mm_features = self.encode_images(pixel_values, grid_sizes, merge_sizes, vision_chunk_size=vision_chunk_size).to(input_ids.device)
        mm_features = self._get_valid_visual_tokens(mm_features, batched_num_patches, modals)

        compression_mask = self._get_compression_mask(
            pixel_values, batched_num_patches, grid_sizes, merge_sizes, modals
        )
        mm_features, compression_mask = self._maybe_truncate_visual_tokens(
            mm_features, compression_mask, batched_num_patches, modals, input_ids, position_ids
        )

        # 3. compress visual tokens
        if self.config.use_token_compression:
            assert B == 1, "Token compression is only supported for batch_size=1"
            mm_features, input_ids, attention_mask, position_ids, labels = self._compress_visual_tokens(
                compression_mask, mm_features, input_ids, attention_mask, position_ids, labels
            )

        # 4. embed text tokens
        inputs_embeds = self.get_model().embed_tokens(input_ids).clone()

        # 5. replace multimodal tokens with features
        image_selected = (input_ids == self.config.image_token_index)
        inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features   

        # 6. reshape back to batched format
        C = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B, -1)

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels
