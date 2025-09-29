from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from deepseek_vl.models import VLChatProcessor,MultiModalityCausalLM,VLChatProcessorOutput
from deepseek_vl.utils.io import load_pil_images

import numpy as np
from PIL import Image

def load_model_and_processor(model_name, model_path):
    if model_name == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_name == 'phi3':
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype="auto")
        model.use_cache = False
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif model_name == 'qwen':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_name == 'deepseek_vl':
        processor:VLChatProcessor=VLChatProcessor.from_pretrained(model_path)
        model:MultiModalityCausalLM=AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True, device_map='auto',torch_dtype=torch.bfloat16)
        model = model.eval()
    else:
        raise NotImplementedError(f"Unsupported model_name = {model_name}")
    
    return model, processor

def reverse_sigmoid(x):
    return torch.log(x/(1-x+1e-3))

def get_inputs(processor, prompt, image, target_output, model_name='llava', eval=False):
    if model_name == 'llava':
        conversation = [{"role": "user", "content":[{'type': "image"}, {"type": "text", "text": prompt}]}]
        conv_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=conv_prompt, return_tensors="pt")
        start = len(inputs.input_ids[0])
        
        conv_prompt = f'{conv_prompt}{target_output}'
        inputs = processor(images=image, text=conv_prompt, return_tensors="pt")
        target_slice = slice(start, len(inputs.input_ids[0]))
        loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    elif model_name == 'phi3':
        conversation = [{"role": "user", "content": "<|image_1|>\n"}, {"role": "user", "content": prompt}]
        conv_prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(conv_prompt, [image], return_tensors="pt")
        start = len(inputs.input_ids[0])
        
        conv_prompt = f'{conv_prompt}{target_output}'
        inputs = processor(conv_prompt, [image], return_tensors="pt")
        target_slice = slice(start, len(inputs.input_ids[0]))
        loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    elif model_name == 'qwen':
        conversation = [{"role": "user", "content":[{'type': "image"}, {"type": "text", "text": prompt}]}]        
        conv_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[conv_prompt], images=[image], videos=None, padding=True, return_tensors="pt")
        start = len(inputs.input_ids[0])

        conv_prompt = f'{conv_prompt}{target_output}'
        inputs = processor(text=[conv_prompt], images=[image], videos=None, padding=True, return_tensors="pt")
        target_slice = slice(start, len(inputs.input_ids[0]))  
        loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    elif model_name == 'deepseek_vl':
        conversation = [
            {
                "role":"User",
                "content":f"<image_placeholder>{prompt}",
                "image":['']
            },
            {
                "role":"Assistant",
                "content":""
            }
        ]
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations = conversation,
            sft_format = processor.sft_format,
            system_prompt=processor.system_prompt,
        )
        sft_input_ids = processor.tokenizer.encode(sft_format)
        start = len(sft_input_ids)
        if eval:
            conv_prompt = sft_format
        else:
            conv_prompt = f'{sft_format} {target_output}'
        combo_input_ids = processor.tokenizer.encode(conv_prompt)
        combo_target_slice = slice(start,len(combo_input_ids))
        combo_loss_slice = slice(combo_target_slice.start-1,combo_target_slice.stop-1)

        input_ids = processor.tokenizer.encode(conv_prompt)
        input_ids = torch.LongTensor(input_ids)

        image_token_mask: torch.BoolTensor = input_ids == processor.image_id
        image_indices = image_token_mask.nonzero()
        input_ids,num_image_tokens = processor.add_image_token(
            image_indices=image_indices,
            input_ids = input_ids,
        )
        image_outputs = processor.image_processor([image],return_tensor='pt')
        prepare = VLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=input_ids,
            pixel_values=image_outputs.pixel_values,
            num_image_tokens=num_image_tokens,
        )
        inputs = processor.batchify([prepare])

        image_tokens = int(num_image_tokens)
        
        if not eval:
            target_slice = slice(combo_target_slice.start+image_tokens-1,combo_target_slice.stop+image_tokens-1)
            loss_slice = slice(target_slice.start-1, target_slice.stop-1)
        else:
            target_slice = combo_target_slice
            loss_slice = combo_loss_slice
    else:
        raise NotImplementedError(f"Unsupported model_name = {model_name}")
    
    return inputs, conv_prompt, target_slice, loss_slice

def exact_math(processor, input_ids, generate_ids, target_output, model_name):
    if model_name == 'llava':
        response = processor.batch_decode(generate_ids[:, len(input_ids[0]):])[0]
    elif model_name == 'phi3':
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    else:
        raise NotImplementedError(f"Unsupported model_name = {model_name}")
    if target_output.endswith(processor.tokenizer.eos_token):
        return response == target_output
    return response.startswith(target_output)



class Process:
    def __init__(self, upper_bound=None, lower_bound=None):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        
    def convert_to_image(self, image, upper_bound=None, lower_bound=None):
        cur_image = F.sigmoid(image)
        upper_bound = upper_bound if upper_bound is not None else self.upper_bound
        lower_bound = lower_bound if lower_bound is not None else self.lower_bound
        if upper_bound is not None and lower_bound is not None:
            cur_image = cur_image * (upper_bound - lower_bound) + lower_bound
        return cur_image

class Llava_Process(Process):
    def __init__(self, processor, upper_bound=None, lower_bound=None):
        self.normalize = transforms.Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std)
        super().__init__(upper_bound, lower_bound)
    
    def preprocess(self, images, norm=True):
        return self.normalize(self.convert_to_image(images)) if norm else self.convert_to_image(images)
    
    def __call__(self, images, inputs, norm=True, *args, **kwargs):
        model_kwargs = {
            'input_ids': inputs.input_ids,
            'pixel_values': self.preprocess(images, norm),
            'attention_mask': inputs.attention_mask
        }
        return model_kwargs

class Phi3_Process(Process):
    def __init__(self, processor, upper_bound=None, lower_bound=None):
        self.processor = processor
        super().__init__(upper_bound, lower_bound)

    def pad_to_max_num_crops_tensor(self, images, max_crops=5):
        """
        images: B x 3 x H x W, B<=max_crops
        """
        B, _, H, W = images.shape
        if B < max_crops:
            pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
            images = torch.cat([images, pad], dim=0)
        return images

    def preprocess(self, images):
        # the input image must be 336*336
        shape = images[0].shape
        assert len(images) != 1 or shape[0] != 336 or shape[1] != 336, 'Current imaplements only support single image with the size of 336*336'
        new_h, new_w = 1344, 1344
        num_crops = 4
        elems = [transforms.functional.resize(img, [new_h, new_w],) for img in images]
        img_processor = transforms.Compose([
                transforms.Normalize(mean=self.processor.image_processor.image_mean, std=self.processor.image_processor.image_std)
            ])
        hd_images = [img_processor(im) for im in elems]
        global_image = [torch.nn.functional.interpolate(im.unsqueeze(0).float(), size=(336, 336), mode='bicubic',).to(im.dtype) for im in hd_images]
        shapes = [[im.size(1), im.size(2)] for im in hd_images]
        num_img_tokens = [int(((h//336)*(w//336)+1)*144 + 1 + (h//336+1)*12) for h, w in shapes]
        # reshape to channel dimension -> (num_images, num_crops, 3, 336, 336)
        # (1, 3, h//336, 336, w//336, 336) -> (1, h//336, w//336, 3, 336, 336) -> (h//336*w//336, 3, 336, 336)
        hd_images_reshape = [im.reshape(1, 3, h//336, 336, w//336, 336).permute(0,2,4,1,3,5).reshape(-1, 3, 336, 336).contiguous() for im, (h, w) in zip(hd_images, shapes)]
        # concat global image and local image
        hd_images_reshape = [torch.cat([_global_image] + [_im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)]

        # pad to max_num_crops
        image_transformed = [self.pad_to_max_num_crops_tensor(im, num_crops+1) for im in hd_images_reshape]
        image_transformed = torch.stack(image_transformed, dim=0)
        
        return image_transformed

    def __call__(self, images, inputs, *args, **kwargs):
        model_kwargs = {
            'input_ids': inputs.input_ids,
            'pixel_values': self.preprocess([self.convert_to_image(images)]),
            'attention_mask': inputs.attention_mask,
            'image_sizes': inputs.image_sizes,
            # 'use_cache': False
        }
        return model_kwargs

class Qwen_Process(Process):
    def __init__(self, processor, upper_bound=None, lower_bound=None):
        self.normalize = transforms.Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std)
        self.processor = processor
        super().__init__(upper_bound, lower_bound)
    
    def preprocess(self, images):
        shape = images[0].shape
        assert len(images) != 1 or shape[0] != 336 or shape[1] != 336, 'Current imaplements only support single image with the size of 336*336'
        resized_height, resized_width = 336, 336
        
        images = torch.tile(images, (self.processor.image_processor.temporal_patch_size, 1, 1, 1))
        channel = images.shape[1]
        grid_t = images.shape[0] // self.processor.image_processor.temporal_patch_size
        grid_h, grid_w = resized_height // self.processor.image_processor.patch_size, resized_width // self.processor.image_processor.patch_size
        images = images.reshape(
                grid_t,
                self.processor.image_processor.temporal_patch_size,
                channel,
                grid_h // self.processor.image_processor.merge_size,
                self.processor.image_processor.merge_size,
                self.processor.image_processor.patch_size,
                grid_w // self.processor.image_processor.merge_size,
                self.processor.image_processor.merge_size,
                self.processor.image_processor.patch_size,
            )
        images = images.permute((0, 3, 6, 4, 7, 2, 1, 5, 8))
        flatten_patches = images.reshape(
            grid_t * grid_h * grid_w, channel * self.processor.image_processor.temporal_patch_size * self.processor.image_processor.patch_size * self.processor.image_processor.patch_size
        )
        
        return flatten_patches
    
    def __call__(self, images, inputs, model, *args, **kwargs):
        images = self.normalize(self.convert_to_image(images))
        model_kwargs = {
            'pixel_values': self.preprocess(images),
            'attention_mask': inputs.attention_mask,
            'image_grid_thw': inputs.image_grid_thw,
            'use_cache': True,
            'past_key_values': DynamicCache()
        }
        model_kwargs = model._get_initial_cache_position(inputs.input_ids, model_kwargs)
        model_kwargs = model.prepare_inputs_for_generation(inputs.input_ids, **model_kwargs)
        
        return model_kwargs

class DeepSeek_Process(Process):
    def __init__(self, processor,upper_bound=None, lower_bound=None):
        self.normalize = transforms.Normalize(
            mean = processor.image_processor.image_mean,
            std = processor.image_processor.image_std
        )
        self.processor = processor
        super().__init__(upper_bound,lower_bound)
    def preprocess(self,image,conv_prompt,norm=False):
        input_ids = self.processor.tokenizer.encode(conv_prompt)
        input_ids = torch.LongTensor(input_ids)

        image_token_mask: torch.BoolTensor = input_ids==self.processor.image_id
        image_indices = image_token_mask.nonzero()
        input_ids,num_image_tokens = self.processor.add_image_token(
            image_indices = image_indices,
            input_ids = input_ids
        )
        image = self.convert_to_image(image)
        images = [image]

        if norm:
            image = [self.normalize(im) for im in images]
        images_tensor = torch.stack(images,dim=0)
        images_outputs = {"pixel_values":images_tensor}

        prepare = VLChatProcessorOutput(
            sft_format=conv_prompt,
            input_ids = input_ids,
            pixel_values = images_outputs["pixel_values"],
            num_image_tokens = num_image_tokens
        )
        prepare = self.processor.batchify([prepare])
        prepare = prepare.to("cuda")
        return prepare
    def __call__(self, image,norm = False):
        return self.preprocess(image,norm)