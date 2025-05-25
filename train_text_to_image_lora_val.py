import argparse
import logging
import math
import os
import random
import shutil
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    clip_model,      # Thêm
    clip_processor,
    is_final_validation=False,
):
    era = args.training_era
    validation_prompts = [
        f"a person wearing clothes inspired by early {era} rock icons",
        f"a person dressed in a {era}-inspired urban outfit",
        f"a person in light summer clothes like in {era} ads",
        f"a person wearing a winter outfit like {era} fashion magazines",
    ]
    
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images for each of {len(validation_prompts)} prompts."
    )
    
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    
    images = []
    captions = []
    clip_scores = []

    autocast_ctx = torch.autocast(accelerator.device.type) if not torch.backends.mps.is_available() else nullcontext()
    with autocast_ctx:
        for prompt_idx, prompt in enumerate(tqdm(validation_prompts, desc="Validation Prompts", disable=not accelerator.is_local_main_process)):
            prompt_images = []
            for _ in range(args.num_validation_images):
                image = pipeline(prompt, num_inference_steps=30, generator=generator).images[0]
                prompt_images.append(image)
                images.append(image) # Lưu PIL image
                captions.append(prompt)

            if args.calculate_clip_score and clip_model is not None and clip_processor is not None:
                with torch.no_grad(): # Quan trọng: không cần tính gradient cho CLIP model
                    # `prompt_images` là list các PIL.Image, `prompt` là string
                    inputs = clip_processor(
                        text=[prompt] * len(prompt_images), images=prompt_images, return_tensors="pt", padding=True
                    )
                    # Chuyển inputs lên device của clip_model (hoặc accelerator.device nếu giống nhau)
                    inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}

                    outputs = clip_model(**inputs)
                    # logits_per_image: [num_images, num_texts]
                    # logits_per_text: [num_texts, num_images]
                    # Chúng ta cần so sánh mỗi ảnh với prompt tương ứng của nó.
                    # Vì ta đưa [prompt]*N và N ảnh, ta lấy đường chéo của logits_per_image
                    # (hoặc tính cosine similarity trực tiếp từ embeddings)

                    # Cách 1: Dùng logits (đã được chuẩn hóa và nhân với logit_scale)
                    # logits = outputs.logits_per_image
                    # diag = torch.diag(logits) # Lấy đường chéo
                    # current_prompt_scores = diag.cpu().tolist()

                    # Cách 2: Tính cosine similarity từ embeddings (linh hoạt hơn)
                    image_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds
                    # Chuẩn hóa embeddings
                    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                    # Cosine similarity
                    # Vì mỗi image_embed chỉ cần so với text_embed tương ứng (cùng prompt)
                    # Và text_embeds có thể chỉ có 1 vector nếu tất cả ảnh dùng chung 1 prompt
                    # Hoặc N vector nếu mỗi ảnh có 1 prompt (nhưng ở đây là nhiều ảnh cho 1 prompt)
                    # Nên ta có thể lặp qua
                    similarity = (image_embeds * text_embeds[0]).sum(dim=-1) # text_embeds[0] vì cùng prompt
                    current_prompt_scores = (similarity * 100.0).cpu().tolist() # Nhân 100 là quy ước của CLIP Score

                    clip_scores.extend(current_prompt_scores)
                    if accelerator.is_local_main_process:
                        logger.info(f"CLIP scores for prompt '{prompt}': {['{:.2f}'.format(s) for s in current_prompt_scores]}")


    wandb_log_data = {} # Tạo dict để log một lần
    
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            logger.info(f"Logging {len(images)} images to W&B for {phase_name}")
            wandb_log_data[phase_name] = [
                wandb.Image(image, caption=f"{i}: {caption}")
                for i, (image, caption) in enumerate(zip(images, captions))
            ]

            if args.calculate_clip_score and clip_scores:
                avg_clip_score = sum(clip_scores) / len(clip_scores)
                logger.info(f"Average CLIP Score for {phase_name} ({len(clip_scores)} images): {avg_clip_score:.4f}")
                wandb_log_data[f"{phase_name}_avg_clip_score"] = avg_clip_score
                # Log scores chi tiết hơn cho từng prompt nếu muốn
                scores_by_prompt = {}
                current_idx = 0
                for p_idx, p_text in enumerate(validation_prompts):
                    num_img_for_prompt = args.num_validation_images
                    scores_for_this_prompt = clip_scores[current_idx : current_idx + num_img_for_prompt]
                    if scores_for_this_prompt:
                       scores_by_prompt[f"{phase_name}_prompt_{p_idx+1}_avg_clip_score"] = sum(scores_for_this_prompt) / len(scores_for_this_prompt)
                    current_idx += num_img_for_prompt
                wandb_log_data.update(scores_by_prompt)


    if wandb_log_data and accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                 tracker.log(wandb_log_data) # Log tất cả một lần

    return images # Trả về PIL images

def save_lora_weights(save_path, unet, text_encoder=None, safe_serialization=True):
    """Save LoRA weights separately for easy sharing and integration"""
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save UNet LoRA weights
    unet_lora_state_dict = get_peft_model_state_dict(unet)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=save_path,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=None,
        safe_serialization=safe_serialization,
    )
    
    logger.info(f"Saved LoRA weights to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop the input images.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate to use.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help='The scheduler type to use.',
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--save_lora_steps",
        type=int,
        default=None,
        help="Save standalone LoRA weights every X steps (in addition to checkpoints).",
    )
    parser.add_argument("--seed", type=int, default=1337, help="A seed for reproducible training.")
    parser.add_argument(
        "--caption_column",
        type=str,
        default="additional_feature",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--training_era",
        type=str,
        default="90s",
        help=(
            "The era of the training data. This is used to determine the style of the generated images."
            " Options are: '90s', '2000s', '2020s'."
        ),
    )
    parser.add_argument(
        "--clip_model_name_or_path",
        type=str,
        default="openai/clip-vit-large-patch14", # Model CLIP phổ biến
        help="Pretrained CLIP model name or path for validation scoring.",
    )
    parser.add_argument(
        "--calculate_clip_score",
        action="store_true",
        help="Whether to calculate CLIP score during validation.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("Need a training folder.")

    # Set default save_lora_steps to match checkpointing_steps if not provided
    if args.save_lora_steps is None:
        args.save_lora_steps = args.checkpointing_steps

    return args

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    clip_model = None
    clip_processor = None
    if args.calculate_clip_score and accelerator.is_main_process: # Chỉ main process mới load và tính
        try:
            logger.info(f"Loading CLIP model: {args.clip_model_name_or_path}")
            clip_model = CLIPModel.from_pretrained(args.clip_model_name_or_path).to(accelerator.device)
            clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name_or_path)
            clip_model.eval() # Đặt ở chế độ evaluation
            logger.info("CLIP model and processor loaded successfully.")
        except Exception as e:
            logger.error(f"Could not load CLIP model or processor: {e}. CLIP score calculation will be skipped.")
            clip_model = None
            clip_processor = None
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet_lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )

    data_files = {}
    if args.train_data_dir is not None:
        data_files["train"] = os.path.join(args.train_data_dir, "**")
    dataset = load_dataset("imagefolder", data_files=data_files)

    column_names = dataset["train"].column_names
    caption_column = args.caption_column
    if caption_column not in column_names:
        raise ValueError(f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}")

    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}")

    interpolation = transforms.InterpolationMode.BILINEAR
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=interpolation),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def tokenize_captions(examples):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption))
            else:
                raise ValueError(f"Caption column `{caption_column}` should contain either strings or lists of strings.")
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, math.ceil(args.max_train_steps / num_update_steps_per_epoch)):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                # Save checkpoint
                # if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                #     checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                #     accelerator.save_state(checkpoint_path)
                #     logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Save standalone LoRA weights
                if global_step % args.save_lora_steps == 0 and accelerator.is_main_process:
                    logger.info(f"Running validation at step {global_step}")
                    if args.calculate_clip_score and clip_model is None:
                        logger.warning("CLIP model not loaded, skipping CLIP score calculation.")
                    pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    torch_dtype=weight_dtype,
                    )
                    images = log_validation(
                        pipeline, args, accelerator, epoch,
                        clip_model, clip_processor # Truyền vào
                    )
                    lora_path = os.path.join(args.output_dir, f"lora-weights-{global_step}")
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    save_lora_weights(lora_path, unwrapped_unet)
                    del pipeline
                    torch.cuda.empty_cache()
            if global_step >= args.max_train_steps:
                break

        # Validation
        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(f"Running validation... \n Generating {args.num_validation_images} images.")
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    torch_dtype=weight_dtype,
                )
                if args.calculate_clip_score and clip_model is None:
                    logger.warning("CLIP model not available, skipping CLIP score calculation for this validation.")
                images = log_validation(
                    pipeline, args, accelerator, epoch,
                    clip_model, clip_processor # Truyền vào
                )

                del pipeline
                torch.cuda.empty_cache()

    # Final save
    if accelerator.is_main_process:
        # Save final checkpoint
        # final_checkpoint_path = os.path.join(args.output_dir, "final-checkpoint")
        # accelerator.save_state(final_checkpoint_path)
        # logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
        
        # Save final LoRA weights
        final_lora_path = os.path.join(args.output_dir, "final-lora-weights")
        unwrapped_unet = accelerator.unwrap_model(unet)
        save_lora_weights(final_lora_path, unwrapped_unet)
        
        # Run final validation if specified
        if args.validation_prompt is not None:
            logger.info(f"Running final validation...")
            if args.calculate_clip_score and clip_model is None:
                logger.warning("CLIP model not available, skipping CLIP score calculation for final validation.")
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                torch_dtype=weight_dtype,
            )
            
            images = log_validation(
            pipeline, args, accelerator, epoch, # epoch ở đây sẽ là epoch cuối
            clip_model, clip_processor, # Truyền vào
            is_final_validation=True
            )
            
            del pipeline
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
