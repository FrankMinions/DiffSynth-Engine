from typing import Literal, TypeAlias


preset_models_on_huggingface = {
    "HunyuanDiT": [
        ("Tencent-Hunyuan/HunyuanDiT", "t2i/clip_text_encoder/pytorch_model.bin", "models/HunyuanDiT/t2i/clip_text_encoder"),
        ("Tencent-Hunyuan/HunyuanDiT", "t2i/mt5/pytorch_model.bin", "models/HunyuanDiT/t2i/mt5"),
        ("Tencent-Hunyuan/HunyuanDiT", "t2i/model/pytorch_model_ema.pt", "models/HunyuanDiT/t2i/model"),
        ("Tencent-Hunyuan/HunyuanDiT", "t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin", "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix"),
    ],
    "stable-video-diffusion-img2vid-xt": [
        ("stabilityai/stable-video-diffusion-img2vid-xt", "svd_xt.safetensors", "models/stable_video_diffusion"),
    ],
    "ExVideo-SVD-128f-v1": [
        ("ECNU-CILab/ExVideo-SVD-128f-v1", "model.fp16.safetensors", "models/stable_video_diffusion"),
    ],
    # Stable Diffusion
    "StableDiffusion_v15": [
        ("benjamin-paine/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors", "models/stable_diffusion"),
    ],
    "DreamShaper_8": [
        ("Yntec/Dreamshaper8", "dreamshaper_8.safetensors", "models/stable_diffusion"),
    ],
    # Textual Inversion
    "TextualInversion_VeryBadImageNegative_v1.3": [
        ("gemasai/verybadimagenegative_v1.3", "verybadimagenegative_v1.3.pt", "models/textual_inversion"),
    ],
    # Stable Diffusion XL
    "StableDiffusionXL_v1": [
        ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", "models/stable_diffusion_xl"),
    ],
    "BluePencilXL_v200": [
        ("frankjoshua/bluePencilXL_v200", "bluePencilXL_v200.safetensors", "models/stable_diffusion_xl"),
    ],
    "StableDiffusionXL_Turbo": [
        ("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors", "models/stable_diffusion_xl_turbo"),
    ],
    # Stable Diffusion 3
    "StableDiffusion3": [
        ("stabilityai/stable-diffusion-3-medium", "sd3_medium_incl_clips_t5xxlfp16.safetensors", "models/stable_diffusion_3"),
    ],
    "StableDiffusion3_without_T5": [
        ("stabilityai/stable-diffusion-3-medium", "sd3_medium_incl_clips.safetensors", "models/stable_diffusion_3"),
    ],
    # ControlNet
    "ControlNet_v11f1p_sd15_depth": [
        ("lllyasviel/ControlNet-v1-1", "control_v11f1p_sd15_depth.pth", "models/ControlNet"),
        ("lllyasviel/Annotators", "dpt_hybrid-midas-501f0c75.pt", "models/Annotators")
    ],
    "ControlNet_v11p_sd15_softedge": [
        ("lllyasviel/ControlNet-v1-1", "control_v11p_sd15_softedge.pth", "models/ControlNet"),
        ("lllyasviel/Annotators", "ControlNetHED.pth", "models/Annotators")
    ],
    "ControlNet_v11f1e_sd15_tile": [
        ("lllyasviel/ControlNet-v1-1", "control_v11f1e_sd15_tile.pth", "models/ControlNet")
    ],
    "ControlNet_v11p_sd15_lineart": [
        ("lllyasviel/ControlNet-v1-1", "control_v11p_sd15_lineart.pth", "models/ControlNet"),
        ("lllyasviel/Annotators", "sk_model.pth", "models/Annotators"),
        ("lllyasviel/Annotators", "sk_model2.pth", "models/Annotators")
    ],
    "ControlNet_union_sdxl_promax": [
        ("xinsir/controlnet-union-sdxl-1.0", "diffusion_pytorch_model_promax.safetensors", "models/ControlNet/controlnet_union"),
        ("lllyasviel/Annotators", "dpt_hybrid-midas-501f0c75.pt", "models/Annotators")
    ],
    # AnimateDiff
    "AnimateDiff_v2": [
        ("guoyww/animatediff", "mm_sd_v15_v2.ckpt", "models/AnimateDiff"),
    ],
    "AnimateDiff_xl_beta": [
        ("guoyww/animatediff", "mm_sdxl_v10_beta.ckpt", "models/AnimateDiff"),
    ],

    # Qwen Prompt
    "QwenPrompt": [
        ("Qwen/Qwen2-1.5B-Instruct", "config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "generation_config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "model.safetensors", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "special_tokens_map.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "tokenizer.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "tokenizer_config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "merges.txt", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "vocab.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
    ],
    # Beautiful Prompt
    "BeautifulPrompt": [
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "generation_config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "model.safetensors", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "special_tokens_map.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "tokenizer.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "tokenizer_config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
    ],
    # Omost prompt
    "OmostPrompt":[
        ("lllyasviel/omost-llama-3-8b-4bits", "model-00001-of-00002.safetensors", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "model-00002-of-00002.safetensors", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "tokenizer.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "tokenizer_config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "generation_config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "model.safetensors.index.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "special_tokens_map.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
    ],
    # Translator
    "opus-mt-zh-en": [
        ("Helsinki-NLP/opus-mt-zh-en", "config.json", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "generation_config.json", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "metadata.json", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "pytorch_model.bin", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "source.spm", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "target.spm", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "tokenizer_config.json", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "vocab.json", "models/translator/opus-mt-zh-en"),
    ],
    # IP-Adapter
    "IP-Adapter-SD": [
        ("h94/IP-Adapter", "models/image_encoder/model.safetensors", "models/IpAdapter/stable_diffusion/image_encoder"),
        ("h94/IP-Adapter", "models/ip-adapter_sd15.bin", "models/IpAdapter/stable_diffusion"),
    ],
    "IP-Adapter-SDXL": [
        ("h94/IP-Adapter", "sdxl_models/image_encoder/model.safetensors", "models/IpAdapter/stable_diffusion_xl/image_encoder"),
        ("h94/IP-Adapter", "sdxl_models/ip-adapter_sdxl.bin", "models/IpAdapter/stable_diffusion_xl"),
    ],
    "SDXL-vae-fp16-fix": [
        ("madebyollin/sdxl-vae-fp16-fix", "diffusion_pytorch_model.safetensors", "models/sdxl-vae-fp16-fix")
    ],
    # Kolors
    "Kolors": [
        ("Kwai-Kolors/Kolors", "text_encoder/config.json", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model.bin.index.json", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00001-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00002-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00003-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00004-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00005-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00006-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00007-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "unet/diffusion_pytorch_model.safetensors", "models/kolors/Kolors/unet"),
        ("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.safetensors", "models/kolors/Kolors/vae"),
    ],
    # FLUX
    "FLUX.1-dev": [
        ("black-forest-labs/FLUX.1-dev", "text_encoder/model.safetensors", "models/FLUX/FLUX.1-dev/text_encoder"),
        ("black-forest-labs/FLUX.1-dev", "text_encoder_2/config.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("black-forest-labs/FLUX.1-dev", "text_encoder_2/model-00001-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("black-forest-labs/FLUX.1-dev", "text_encoder_2/model-00002-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("black-forest-labs/FLUX.1-dev", "text_encoder_2/model.safetensors.index.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("black-forest-labs/FLUX.1-dev", "ae.safetensors", "models/FLUX/FLUX.1-dev"),
        ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors", "models/FLUX/FLUX.1-dev"),
    ],
    # RIFE
    "RIFE": [
        ("AlexWortega/RIFE", "flownet.pkl", "models/RIFE"),
    ],
    # CogVideo
    "CogVideoX-5B": [
        ("THUDM/CogVideoX-5b", "text_encoder/config.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "text_encoder/model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "text_encoder/model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "text_encoder/model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "transformer/config.json", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "transformer/diffusion_pytorch_model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "vae/diffusion_pytorch_model.safetensors", "models/CogVideo/CogVideoX-5b/vae"),
    ],
}
preset_models_on_modelscope = {
    # Hunyuan DiT
    "HunyuanDiT": [
        ("modelscope/HunyuanDiT", "t2i/clip_text_encoder/pytorch_model.bin", "models/HunyuanDiT/t2i/clip_text_encoder"),
        ("modelscope/HunyuanDiT", "t2i/mt5/pytorch_model.bin", "models/HunyuanDiT/t2i/mt5"),
        ("modelscope/HunyuanDiT", "t2i/model/pytorch_model_ema.pt", "models/HunyuanDiT/t2i/model"),
        ("modelscope/HunyuanDiT", "t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin", "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix"),
    ],
    # Stable Video Diffusion
    "stable-video-diffusion-img2vid-xt": [
        ("AI-ModelScope/stable-video-diffusion-img2vid-xt", "svd_xt.safetensors", "models/stable_video_diffusion"),
    ],
    # ExVideo
    "ExVideo-SVD-128f-v1": [
        ("ECNU-CILab/ExVideo-SVD-128f-v1", "model.fp16.safetensors", "models/stable_video_diffusion"),
    ],
    "ExVideo-CogVideoX-LoRA-129f-v1": [
        ("ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1", "ExVideo-CogVideoX-LoRA-129f-v1.safetensors", "models/lora"),
    ],
    # Stable Diffusion
    "StableDiffusion_v15": [
        ("AI-ModelScope/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors", "models/stable_diffusion"),
    ],
    "DreamShaper_8": [
        ("sd_lora/dreamshaper_8", "dreamshaper_8.safetensors", "models/stable_diffusion"),
    ],
    "AingDiffusion_v12": [
        ("sd_lora/aingdiffusion_v12", "aingdiffusion_v12.safetensors", "models/stable_diffusion"),
    ],
    "Flat2DAnimerge_v45Sharp": [
        ("sd_lora/Flat-2D-Animerge", "flat2DAnimerge_v45Sharp.safetensors", "models/stable_diffusion"),
    ],
    # Textual Inversion
    "TextualInversion_VeryBadImageNegative_v1.3": [
        ("sd_lora/verybadimagenegative_v1.3", "verybadimagenegative_v1.3.pt", "models/textual_inversion"),
    ],
    # Stable Diffusion XL
    "StableDiffusionXL_v1": [
        ("AI-ModelScope/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", "models/stable_diffusion_xl"),
    ],
    "BluePencilXL_v200": [
        ("sd_lora/bluePencilXL_v200", "bluePencilXL_v200.safetensors", "models/stable_diffusion_xl"),
    ],
    "StableDiffusionXL_Turbo": [
        ("AI-ModelScope/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors", "models/stable_diffusion_xl_turbo"),
    ],
    "SDXL_lora_zyd232_ChineseInkStyle_SDXL_v1_0": [
        ("sd_lora/zyd232_ChineseInkStyle_SDXL_v1_0", "zyd232_ChineseInkStyle_SDXL_v1_0.safetensors", "models/lora"),
    ],
    # Stable Diffusion 3
    "StableDiffusion3": [
        ("AI-ModelScope/stable-diffusion-3-medium", "sd3_medium_incl_clips_t5xxlfp16.safetensors", "models/stable_diffusion_3"),
    ],
    "StableDiffusion3_without_T5": [
        ("AI-ModelScope/stable-diffusion-3-medium", "sd3_medium_incl_clips.safetensors", "models/stable_diffusion_3"),
    ],
    # ControlNet
    "ControlNet_v11f1p_sd15_depth": [
        ("AI-ModelScope/ControlNet-v1-1", "control_v11f1p_sd15_depth.pth", "models/ControlNet"),
        ("sd_lora/Annotators", "dpt_hybrid-midas-501f0c75.pt", "models/Annotators")
    ],
    "ControlNet_v11p_sd15_softedge": [
        ("AI-ModelScope/ControlNet-v1-1", "control_v11p_sd15_softedge.pth", "models/ControlNet"),
        ("sd_lora/Annotators", "ControlNetHED.pth", "models/Annotators")
    ],
    "ControlNet_v11f1e_sd15_tile": [
        ("AI-ModelScope/ControlNet-v1-1", "control_v11f1e_sd15_tile.pth", "models/ControlNet")
    ],
    "ControlNet_v11p_sd15_lineart": [
        ("AI-ModelScope/ControlNet-v1-1", "control_v11p_sd15_lineart.pth", "models/ControlNet"),
        ("sd_lora/Annotators", "sk_model.pth", "models/Annotators"),
        ("sd_lora/Annotators", "sk_model2.pth", "models/Annotators")
    ],
    "ControlNet_union_sdxl_promax": [
        ("AI-ModelScope/controlnet-union-sdxl-1.0", "diffusion_pytorch_model_promax.safetensors", "models/ControlNet/controlnet_union"),
        ("sd_lora/Annotators", "dpt_hybrid-midas-501f0c75.pt", "models/Annotators")
    ],
    "Annotators:Depth": [
        ("sd_lora/Annotators", "dpt_hybrid-midas-501f0c75.pt", "models/Annotators"),
    ],
    "Annotators:Softedge": [
        ("sd_lora/Annotators", "ControlNetHED.pth", "models/Annotators"),
    ],
    "Annotators:Lineart": [
        ("sd_lora/Annotators", "sk_model.pth", "models/Annotators"),
        ("sd_lora/Annotators", "sk_model2.pth", "models/Annotators"),
    ],
    "Annotators:Normal": [
        ("sd_lora/Annotators", "scannet.pt", "models/Annotators"),
    ],
    "Annotators:Openpose": [
        ("sd_lora/Annotators", "body_pose_model.pth", "models/Annotators"),
        ("sd_lora/Annotators", "facenet.pth", "models/Annotators"),
        ("sd_lora/Annotators", "hand_pose_model.pth", "models/Annotators"),
    ],
    # AnimateDiff
    "AnimateDiff_v2": [
        ("Shanghai_AI_Laboratory/animatediff", "mm_sd_v15_v2.ckpt", "models/AnimateDiff"),
    ],
    "AnimateDiff_xl_beta": [
        ("Shanghai_AI_Laboratory/animatediff", "mm_sdxl_v10_beta.ckpt", "models/AnimateDiff"),
    ],
    # RIFE
    "RIFE": [
        ("Damo_XR_Lab/cv_rife_video-frame-interpolation", "flownet.pkl", "models/RIFE"),
    ],
    # Qwen Prompt
    "QwenPrompt": {
        "file_list": [
            ("qwen/Qwen2-1.5B-Instruct", "config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "generation_config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "model.safetensors", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "special_tokens_map.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "tokenizer.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "tokenizer_config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "merges.txt", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "vocab.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ],
        "load_path": [
            "models/QwenPrompt/qwen2-1.5b-instruct",
        ],
    },
    # Beautiful Prompt
    "BeautifulPrompt": {
        "file_list": [
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "generation_config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "model.safetensors", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "special_tokens_map.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "tokenizer.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "tokenizer_config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ],
        "load_path": [
            "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd",
        ],
    },
    # Omost prompt
    "OmostPrompt": {
        "file_list": [
            ("Omost/omost-llama-3-8b-4bits", "model-00001-of-00002.safetensors", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "model-00002-of-00002.safetensors", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "tokenizer.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "tokenizer_config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "generation_config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "model.safetensors.index.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "special_tokens_map.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ],
        "load_path": [
            "models/OmostPrompt/omost-llama-3-8b-4bits",
        ],
    },
    # Translator
    "opus-mt-zh-en": {
        "file_list": [
            ("moxying/opus-mt-zh-en", "config.json", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "generation_config.json", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "metadata.json", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "pytorch_model.bin", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "source.spm", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "target.spm", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "tokenizer_config.json", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "vocab.json", "models/translator/opus-mt-zh-en"),
        ],
        "load_path": [
            "models/translator/opus-mt-zh-en",
        ],
    },
    # IP-Adapter
    "IP-Adapter-SD": [
        ("AI-ModelScope/IP-Adapter", "models/image_encoder/model.safetensors", "models/IpAdapter/stable_diffusion/image_encoder"),
        ("AI-ModelScope/IP-Adapter", "models/ip-adapter_sd15.bin", "models/IpAdapter/stable_diffusion"),
    ],
    "IP-Adapter-SDXL": [
        ("AI-ModelScope/IP-Adapter", "sdxl_models/image_encoder/model.safetensors", "models/IpAdapter/stable_diffusion_xl/image_encoder"),
        ("AI-ModelScope/IP-Adapter", "sdxl_models/ip-adapter_sdxl.bin", "models/IpAdapter/stable_diffusion_xl"),
    ],
    # Kolors
    "Kolors": {
        "file_list": [
            ("Kwai-Kolors/Kolors", "text_encoder/config.json", "models/kolors/Kolors/text_encoder"),
            ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model.bin.index.json", "models/kolors/Kolors/text_encoder"),
            ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00001-of-00007.bin", "models/kolors/Kolors/text_encoder"),
            ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00002-of-00007.bin", "models/kolors/Kolors/text_encoder"),
            ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00003-of-00007.bin", "models/kolors/Kolors/text_encoder"),
            ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00004-of-00007.bin", "models/kolors/Kolors/text_encoder"),
            ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00005-of-00007.bin", "models/kolors/Kolors/text_encoder"),
            ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00006-of-00007.bin", "models/kolors/Kolors/text_encoder"),
            ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00007-of-00007.bin", "models/kolors/Kolors/text_encoder"),
            ("Kwai-Kolors/Kolors", "unet/diffusion_pytorch_model.safetensors", "models/kolors/Kolors/unet"),
            ("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.safetensors", "models/kolors/Kolors/vae"),
        ],
        "load_path": [
            "models/kolors/Kolors/text_encoder",
            "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
            "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors",
        ],
    },
    "SDXL-vae-fp16-fix": [
        ("AI-ModelScope/sdxl-vae-fp16-fix", "diffusion_pytorch_model.safetensors", "models/sdxl-vae-fp16-fix")
    ],
    # FLUX
    "FLUX.1-dev": {
        "file_list": [
            ("AI-ModelScope/FLUX.1-dev", "text_encoder/model.safetensors", "models/FLUX/FLUX.1-dev/text_encoder"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/config.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model-00001-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model-00002-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model.safetensors.index.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "ae.safetensors", "models/FLUX/FLUX.1-dev"),
            ("AI-ModelScope/FLUX.1-dev", "flux1-dev.safetensors", "models/FLUX/FLUX.1-dev"),
        ],
        "load_path": [
            "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            "models/FLUX/FLUX.1-dev/text_encoder_2",
            "models/FLUX/FLUX.1-dev/ae.safetensors",
            "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
        ],
    },
    "FLUX.1-schnell": {
        "file_list": [
            ("AI-ModelScope/FLUX.1-dev", "text_encoder/model.safetensors", "models/FLUX/FLUX.1-dev/text_encoder"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/config.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model-00001-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model-00002-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model.safetensors.index.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "ae.safetensors", "models/FLUX/FLUX.1-dev"),
            ("AI-ModelScope/FLUX.1-schnell", "flux1-schnell.safetensors", "models/FLUX/FLUX.1-schnell"),
        ],
        "load_path": [
            "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            "models/FLUX/FLUX.1-dev/text_encoder_2",
            "models/FLUX/FLUX.1-dev/ae.safetensors",
            "models/FLUX/FLUX.1-schnell/flux1-schnell.safetensors"
        ],
    },
    "InstantX/FLUX.1-dev-Controlnet-Union-alpha": [
        ("InstantX/FLUX.1-dev-Controlnet-Union-alpha", "diffusion_pytorch_model.safetensors", "models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha"),
    ],
    "jasperai/Flux.1-dev-Controlnet-Depth": [
        ("jasperai/Flux.1-dev-Controlnet-Depth", "diffusion_pytorch_model.safetensors", "models/ControlNet/jasperai/Flux.1-dev-Controlnet-Depth"),
    ],
    "jasperai/Flux.1-dev-Controlnet-Surface-Normals": [
        ("jasperai/Flux.1-dev-Controlnet-Surface-Normals", "diffusion_pytorch_model.safetensors", "models/ControlNet/jasperai/Flux.1-dev-Controlnet-Surface-Normals"),
    ],
    "jasperai/Flux.1-dev-Controlnet-Upscaler": [
        ("jasperai/Flux.1-dev-Controlnet-Upscaler", "diffusion_pytorch_model.safetensors", "models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler"),
    ],
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha": [
        ("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", "diffusion_pytorch_model.safetensors", "models/ControlNet/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha"),
    ],
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta": [
        ("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", "diffusion_pytorch_model.safetensors", "models/ControlNet/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"),
    ],
    "Shakker-Labs/FLUX.1-dev-ControlNet-Depth": [
        ("Shakker-Labs/FLUX.1-dev-ControlNet-Depth", "diffusion_pytorch_model.safetensors", "models/ControlNet/Shakker-Labs/FLUX.1-dev-ControlNet-Depth"),
    ],
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro": [
        ("Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro", "diffusion_pytorch_model.safetensors", "models/ControlNet/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"),
    ],
    # ESRGAN
    "ESRGAN_x4": [
        ("AI-ModelScope/Real-ESRGAN", "RealESRGAN_x4.pth", "models/ESRGAN"),
    ],
    # RIFE
    "RIFE": [
        ("AI-ModelScope/RIFE", "flownet.pkl", "models/RIFE"),
    ],
    # CogVideo
    "CogVideoX-5B": {
        "file_list": [
            ("ZhipuAI/CogVideoX-5b", "text_encoder/config.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "text_encoder/model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "text_encoder/model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "text_encoder/model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "transformer/config.json", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "transformer/diffusion_pytorch_model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "vae/diffusion_pytorch_model.safetensors", "models/CogVideo/CogVideoX-5b/vae"),
        ],
        "load_path": [
            "models/CogVideo/CogVideoX-5b/text_encoder",
            "models/CogVideo/CogVideoX-5b/transformer",
            "models/CogVideo/CogVideoX-5b/vae/diffusion_pytorch_model.safetensors",
        ],
    },
}
Preset_model_id: TypeAlias = Literal[
    "HunyuanDiT",
    "stable-video-diffusion-img2vid-xt",
    "ExVideo-SVD-128f-v1",
    "ExVideo-CogVideoX-LoRA-129f-v1",
    "StableDiffusion_v15",
    "DreamShaper_8",
    "AingDiffusion_v12",
    "Flat2DAnimerge_v45Sharp",
    "TextualInversion_VeryBadImageNegative_v1.3",
    "StableDiffusionXL_v1",
    "BluePencilXL_v200",
    "StableDiffusionXL_Turbo",
    "ControlNet_v11f1p_sd15_depth",
    "ControlNet_v11p_sd15_softedge",
    "ControlNet_v11f1e_sd15_tile",
    "ControlNet_v11p_sd15_lineart",
    "AnimateDiff_v2",
    "AnimateDiff_xl_beta",
    "RIFE",
    "BeautifulPrompt",
    "opus-mt-zh-en",
    "IP-Adapter-SD",
    "IP-Adapter-SDXL",
    "StableDiffusion3",
    "StableDiffusion3_without_T5",
    "Kolors",
    "SDXL-vae-fp16-fix",
    "ControlNet_union_sdxl_promax",
    "FLUX.1-dev",
    "FLUX.1-schnell",
    "InstantX/FLUX.1-dev-Controlnet-Union-alpha",
    "jasperai/Flux.1-dev-Controlnet-Depth",
    "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    "SDXL_lora_zyd232_ChineseInkStyle_SDXL_v1_0",
    "QwenPrompt",
    "OmostPrompt",
    "ESRGAN_x4",
    "RIFE",
    "CogVideoX-5B",
    "Annotators:Depth",
    "Annotators:Softedge",
    "Annotators:Lineart",
    "Annotators:Normal",
    "Annotators:Openpose",
]
