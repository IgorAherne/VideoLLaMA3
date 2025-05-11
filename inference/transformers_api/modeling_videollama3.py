# Adopted from https://github.com/haotian-liu/LLaVA.
# Below is the original copyright:
# Copyright 2023 Haotian Liu
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
"""PyTorch VideoLLaMA3 model."""

import importlib.util
import os.path as osp
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers import AutoModel, Qwen2ForCausalLM, Qwen2Model
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_videollama3 import Videollama3Qwen2Config
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "configuration_videollama3",
        osp.join(osp.dirname(__file__), "configuration_videollama3.py"),
    )
    configuration_videollama3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration_videollama3)
    Videollama3Qwen2Config = getattr(
        configuration_videollama3,
        "Videollama3Qwen2Config",
    )


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


def build_vision_projector(config, delay_load=False, **kwargs):
    # videollama3 projector only support image-wise operation now, i.e., prohibit the temporal aggregation
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type.startswith("mlp"):
        return MlpGeluProjector(config, projector_type)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')


class MlpGeluProjector(nn.Module):

    def __init__(self, config, projector_type):
        super().__init__()

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))

        self.readout = build_mlp(mlp_depth, config.vision_encoder_config.hidden_size, config.hidden_size)

    def forward(self, x):
        x = self.readout(x)
        return x


class Videollama3MetaModel:

    def __init__(self, config):
        super(Videollama3MetaModel, self).__init__(config)
        if config.vision_encoder is not None:
            self.vision_encoder = AutoModel.from_pretrained(
                config.vision_encoder,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.dtype,
            )
            self.config.vision_encoder_config = self.vision_encoder.config
            self.config.vision_encoder = None
        elif config.vision_encoder_config is not None:
            self.vision_encoder = AutoModel.from_config(
                self.config.vision_encoder_config,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.dtype,
            )
        else:
            raise ValueError("Vision encoder is not provided in config")
        self.mm_projector = build_vision_projector(config)

    def get_vision_encoder(self):
        return self.vision_encoder

    def get_mm_projector(self):
        return self.mm_projector


class Videollama3Qwen2Model(Videollama3MetaModel, Qwen2Model):

    config_class = Videollama3Qwen2Config

    def __init__(self, config: Videollama3Qwen2Config):
        super(Videollama3Qwen2Model, self).__init__(config)


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
        grid_sizes: Optional[torch.LongTensor] = None, # Shape (B_vis, 3) where B_vis is num visual inputs; each row (t, h_patches, w_patches)
        merge_sizes: Optional[torch.LongTensor] = None, # Shape (B_vis), linear merge factor for H and W patch dimensions
        modals: Optional[List[str]] = None,
        vision_chunk_size: Optional[int] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        vision_encoder = self.get_vision_encoder()
        if vision_encoder is None or pixel_values is None or (input_ids is not None and input_ids.shape[1] == 1):
            if inputs_embeds is None and input_ids is not None:
                 inputs_embeds = self.get_model().embed_tokens(input_ids)
            return input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels

        # 1. Flatten text inputs (assuming B_text is the batch dim of input_ids, often 1 for prefill)
        B_text, N_text = input_ids.shape
        input_ids_flat = input_ids.view(B_text * N_text)
        
        attention_mask_flat = attention_mask.view(B_text * N_text) if attention_mask is not None else None
        position_ids_flat = position_ids.view(B_text * N_text) if position_ids is not None else None
        labels_flat = labels.view(B_text * N_text) if labels is not None else None

        # 2. Calculate batched_num_patches (post-merge) for helper functions.
        # This should be a tensor of shape (B_vis), where each element is the number of 
        # visual tokens output by the vision_encoder (and thus projector) for that visual item.
        # grid_sizes[:, 0] is T (num frames)
        # grid_sizes[:, 1] is H_patches (num patches in height before merge)
        # grid_sizes[:, 2] is W_patches (num patches in width before merge)
        # merge_sizes is the linear factor by which H_patches and W_patches are reduced.
        if grid_sizes.shape[0] != merge_sizes.shape[0]:
            raise ValueError(f"Batch dimension mismatch between grid_sizes ({grid_sizes.shape[0]}) and merge_sizes ({merge_sizes.shape[0]})")

        # Calculate effective number of patches in H and W dimensions after merging
        eff_H_patches = grid_sizes[:, 1].float() / merge_sizes.float()
        eff_W_patches = grid_sizes[:, 2].float() / merge_sizes.float()

        # Ensure these are integers, rounding up (ceil) if vision encoder effectively pads,
        # or floor if it truncates. The image_processor.py uses implicit floor via integer division
        # when creating patch grids, and `Videollama3ImageProcessor.process_text` uses `/ merge_size`
        # which implies float division then product.
        # The most robust is to match how image_processor.py counts tokens for DEFAULT_IMAGE_TOKEN.
        # Videollama3ImageProcessor.process_text -> grid_sizes_for_text.append(grid_size[1:] / merge_size)
        # and then thw.prod().long(). So, direct float division is fine here.
        
        # Total number of visual tokens per item after merging in vision encoder
        batched_num_patches_post_merge = (grid_sizes[:, 0].float() * eff_H_patches * eff_W_patches).long()


        # Embed visual tokens using the (potentially chunked) encode_images
        mm_features = self.encode_images(
            pixel_values, 
            grid_sizes, 
            merge_sizes, 
            vision_chunk_size=vision_chunk_size
        ).to(input_ids.device)

        # Filter and process visual features using helper methods
        # These helpers now receive the post-merge patch counts per item.
        mm_features_valid = self._get_valid_visual_tokens(mm_features, batched_num_patches_post_merge, modals)

        compression_mask = self._get_compression_mask(
            pixel_values, batched_num_patches_post_merge, grid_sizes, merge_sizes, modals
        ) # pixel_values is pre-encoder, grid_sizes is pre-merge patch counts
          # batched_num_patches_post_merge is post-merge counts.
          # Ensure _get_compression_mask correctly uses these. The original _get_compression_mask
          # re-calculates `images.view(t, (h // merge_size) * (w // merge_size), -1)`,
          # which is consistent with `batched_num_patches_post_merge`.

        mm_features_truncated, compression_mask_truncated = self._maybe_truncate_visual_tokens(
            mm_features_valid, compression_mask, batched_num_patches_post_merge, modals, input_ids_flat, position_ids_flat
        )

        # 3. Compress visual tokens if enabled
        if self.config.use_token_compression:
            if B_text != 1:
                logger.warning("Token compression is typically designed for a text batch size of 1 during multimodal prefill. Proceeding, but verify behavior for B_text > 1.")
            
            mm_features_final, input_ids_final, attention_mask_final, position_ids_final, labels_final = self._compress_visual_tokens(
                compression_mask_truncated, mm_features_truncated, input_ids_flat, attention_mask_flat, position_ids_flat, labels_flat
            )
        else:
            mm_features_final = mm_features_truncated
            input_ids_final = input_ids_flat
            attention_mask_final = attention_mask_flat
            position_ids_final = position_ids_flat
            labels_final = labels_flat

        # 4. Embed text tokens
        current_inputs_embeds = self.get_model().embed_tokens(input_ids_final).clone()

        # 5. Replace multimodal placeholder tokens with actual visual features
        image_selected_final = (input_ids_final == self.config.image_token_index)
        
        if image_selected_final.sum() != mm_features_final.shape[0]:
            # This error check is crucial.
            # For debugging, one might want to log the counts from different stages.
            # E.g., total expected image tokens from batched_num_patches_post_merge.sum()
            # vs mm_features.shape[0] (after encode_images)
            # vs image_selected_final.sum() (after input_ids_final is determined)
            # vs mm_features_final.shape[0] (after all mm_features processing)
            expected_total_image_tokens = batched_num_patches_post_merge[torch.tensor([m != "text" for m in modals])].sum() # Sum only for actual visual items
            logger.error(
                f"Mismatch details: \n"
                f"  - Expected total image tokens from batched_num_patches_post_merge (for non-text modals): {expected_total_image_tokens.item()}\n"
                f"  - mm_features shape[0] (output of encode_images): {mm_features.shape[0]}\n"
                f"  - mm_features_valid shape[0] (after _get_valid_visual_tokens): {mm_features_valid.shape[0]}\n"
                f"  - mm_features_truncated shape[0] (after _maybe_truncate_visual_tokens): {mm_features_truncated.shape[0]}\n"
                f"  - mm_features_final shape[0] (after potential compression): {mm_features_final.shape[0]}\n"
                f"  - Number of <IMAGE> tokens in input_ids_flat: {input_ids_flat.eq(self.config.image_token_index).sum().item()}\n"
                f"  - Number of <IMAGE> tokens in input_ids_final (to be replaced): {image_selected_final.sum().item()}"
            )
            raise ValueError(
                f"Mismatch between number of image placeholder tokens ({image_selected_final.sum()}) "
                f"and actual multimodal features ({mm_features_final.shape[0]}) after all processing steps."
            )
        current_inputs_embeds[image_selected_final] = mm_features_final.to(current_inputs_embeds.dtype)

        # 6. Reshape back to batched format
        C_embed = current_inputs_embeds.shape[-1]
        current_seq_len = current_inputs_embeds.shape[0] // B_text 
        if current_inputs_embeds.shape[0] % B_text != 0:
            raise ValueError(f"Total number of embeddings ({current_inputs_embeds.shape[0]}) is not divisible by the original text batch size ({B_text}).")

        inputs_embeds_batched = current_inputs_embeds.view(B_text, current_seq_len, C_embed)
        
        attention_mask_batched = attention_mask_final.view(B_text, current_seq_len) if attention_mask_final is not None else None
        labels_batched = labels_final.view(B_text, current_seq_len) if labels_final is not None else None
        position_ids_batched = position_ids_final.view(B_text, current_seq_len) if position_ids_final is not None else None
        
        return None, attention_mask_batched, position_ids_batched, past_key_values, inputs_embeds_batched, labels_batched


class Videollama3Qwen2ForCausalLM(Qwen2ForCausalLM, Videollama3MetaForCausalLM):

    config_class = Videollama3Qwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Videollama3Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def get_model(self):
        return self.model


    # NOTE: arguments are copied from transformers==4.46.3
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None, # Should match Qwen2: Optional[Union[Cache, List[torch.FloatTensor]]]
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0, # <--- RENAMED and type matched
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        vision_chunk_size: Optional[int] = None,
        **loss_kwargs, # Corresponds to **kwargs: Unpack[KwargsForCausalLM] in base
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None and pixel_values is not None: 
            (
                _, 
                attention_mask_multimodal,
                position_ids_multimodal,
                _, 
                inputs_embeds_multimodal, 
                labels_multimodal,
            ) = self.prepare_inputs_labels_for_multimodal( # This is your custom method
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
                vision_chunk_size=vision_chunk_size,
                inputs_embeds=inputs_embeds 
            )
            inputs_embeds = inputs_embeds_multimodal
            if attention_mask_multimodal is not None:
                attention_mask = attention_mask_multimodal
            if position_ids_multimodal is not None:
                position_ids = position_ids_multimodal
            if labels_multimodal is not None:
                labels = labels_multimodal
        
        # All other kwargs not explicitly listed in your forward but accepted by Qwen2ForCausalLM.forward
        # (like those in KwargsForCausalLM if any, beyond loss_kwargs) would be passed via **loss_kwargs here.
        # It's generally fine if loss_kwargs is primarily for loss function parameters.
        return super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, 
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep, # <--- PASSING THE CORRECTLY NAMED PARAMETER
            **loss_kwargs, # Pass along any other relevant kwargs
        )


    @torch.no_grad()
    def generate(
        self,
        # multimodal inputs (for the first pass if not already embedded)
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        vision_chunk_size: Optional[int] = None, # <--- NEW PARAMETER for initial processing
        **kwargs, # May contain input_ids, attention_mask etc.
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        # If pixel_values are provided and inputs_embeds are not already in kwargs,
        # it means this is the first call and we need to process visual inputs.
        if pixel_values is not None and "inputs_embeds" not in kwargs:
            # Pop relevant arguments for prepare_inputs_labels_for_multimodal from kwargs
            # These will be replaced by their multimodal versions
            current_input_ids = kwargs.pop("input_ids", None)
            current_attention_mask = kwargs.pop("attention_mask", None)
            current_position_ids = kwargs.pop("position_ids", None)
            # past_key_values usually None for the first step of generation
            current_past_key_values = kwargs.pop("past_key_values", None) 

            if current_input_ids is None:
                # This case should be handled by HF generate if bos_token_id is set
                raise ValueError("input_ids must be provided to generate if pixel_values are present.")

            (
                _, # multimodal_input_ids (None from prepare_inputs_labels_for_multimodal)
                multimodal_attention_mask,
                multimodal_position_ids,
                _, # past_key_values (not modified for the first token by this func)
                multimodal_inputs_embeds,
                _, # labels (None for generation)
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                position_ids=current_position_ids,
                past_key_values=current_past_key_values, # Should be None for first pass
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
                vision_chunk_size=vision_chunk_size, # <--- PASS vision_chunk_size HERE
            )
            # Update kwargs for the call to super().generate()
            kwargs["inputs_embeds"] = multimodal_inputs_embeds
            if multimodal_attention_mask is not None:
                kwargs["attention_mask"] = multimodal_attention_mask
            if multimodal_position_ids is not None:
                kwargs["position_ids"] = multimodal_position_ids
            # Pass the original current_input_ids because `generate` needs it for
            # determining sequence length, eos criteria, etc., even if embeddings are provided.
            kwargs["input_ids"] = current_input_ids

        # vision_chunk_size has been consumed by prepare_inputs_labels_for_multimodal if it was used.
        # It's not needed by super().generate() or subsequent forward calls during autoregression.
        return super().generate(**kwargs)



    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
