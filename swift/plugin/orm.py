import os
import re
from typing import TYPE_CHECKING, Dict, List, Union

import wandb

import torch
import PIL.Image
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional, List, Dict, Union, Any

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from .utils.text_encoder import CustomTextEncoder

import json
import copy

if TYPE_CHECKING:
    from swift.llm import InferRequest


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):
    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):
    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards



# --- imports ---
import copy
import re
from typing import List, Tuple, Optional, Dict, Any
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL
from PIL import Image

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LMSDiscreteScheduler,
    DDPMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel

# --- helpers ---

def _randn_like(x: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Losowanie N(0,1) kompatybilne z wersjami PyTorch bez wsparcia generatora w randn_like."""
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)

def _pad_to_multiple_of_8(t: torch.Tensor) -> torch.Tensor:
    # t: [B, C, H, W], pad symetryczny do najbliższej wielokrotności 8
    _, _, h, w = t.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)  # (left, right, top, bottom)
    return F.pad(t, pad, mode="reflect")

# --- preprocessing ---

@torch.no_grad()
def preprocess_target_image(image: Image.Image, device: torch.device, resolution: Optional[int] = None) -> torch.Tensor:
    """
    PIL.Image -> tensor [1,3,H,W] w zakresie [-1,1].
    Jeśli resolution=None: brak resize/crop. Tylko pad do wielokrotności 8.
    Jeśli resolution=int: resize+centercrop do kwadratu 'resolution'.
    """
    if resolution is None:
        pre = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        img = image.convert("RGB")
        t = pre(img).unsqueeze(0).to(device=device, dtype=torch.float32)
        return _pad_to_multiple_of_8(t)

    pre = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = image.convert("RGB")
    t = pre(img).unsqueeze(0).to(device=device, dtype=torch.float32)
    return t


class DenoisingReward(ORM):
    """
    Reward: średnia po K krokach poprawa względem uncond:
        mean_t( MSE_uncond(t) - MSE_cond(t) )
    MSE liczone między predykcją UNet a celem (epsilon/velocity) przy DDPM noisingu.
    Wizualizacja: sampling przez oddzielny scheduler (LMS domyślnie).
    """

    def __init__(
        self,
        base_model_name: str,
        unlearned_unet_path: str,
        device: str = "cuda",
        *,
        input_resolution: Optional[int] = None,  # None => brak resize w reward
        compute_dtype: torch.dtype = torch.float16,
        reward_num_timesteps: int = 12,
        normalize_batch_reward: bool = True,  # pozostawione dla kompatybilności, nieużywane
        sampler_kind: str = "lms",
        seed: int = 42,
    ):
        self.device = torch.device(device)
        self.image_cache: Dict[str, torch.Tensor] = {}
        self.input_resolution = input_resolution
        self.compute_dtype = compute_dtype
        self.reward_num_timesteps = max(1, int(reward_num_timesteps))
        self.normalize_batch_reward = normalize_batch_reward
        self.sampler_kind = sampler_kind.lower()
        self.seed = int(seed)

        print(f"[DenoisingReward] base_model={base_model_name}")
        print(f"[DenoisingReward] unlearned_unet_path={unlearned_unet_path}")

        try:
            # --- komponenty SD ---
            self.vae = AutoencoderKL.from_pretrained(base_model_name, subfolder="vae").to(device=self.device)
            self.latent_scaling: float = float(getattr(self.vae.config, "scaling_factor", 0.18215))

            self.tokenizer = CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(base_model_name, subfolder="text_encoder").to(
                dtype=self.compute_dtype, device=self.device
            )

            unet_config = UNet2DConditionModel.load_config(base_model_name, subfolder="unet")
            with torch.no_grad():
                self.unet = UNet2DConditionModel.from_config(unet_config).to(
                    dtype=self.compute_dtype, device=self.device
                )

            # --- scheduler do NOISINGU (reward) ---
            self.train_scheduler = DDPMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                prediction_type="epsilon",
            )
            _pt = (self.train_scheduler.config.get("prediction_type", "epsilon")
                   if isinstance(self.train_scheduler.config, Mapping) else "epsilon")

            # --- scheduler do GENERACJI (wizualizacja) ---
            self.sample_scheduler = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )

            # --- wczytanie wag UNet ---
            state_dict = torch.load(unlearned_unet_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.unet.load_state_dict(state_dict, strict=True)

            # --- eval i zamrożenie ---
            self.vae.eval().requires_grad_(False)
            self.text_encoder.eval().requires_grad_(False)
            self.unet.eval().requires_grad_(False)

            print(f"[DenoisingReward] Loaded UNet. prediction_type(train_scheduler)={_pt}")
            print(f"[DenoisingReward] VAE scaling_factor={self.latent_scaling:.6f}")
        except Exception as e:
            print(f"[DenoisingReward] Error during initialization: {e}")
            raise

    # --- cache latentów ---

    def _get_cached_image_latent(self, image_path: str) -> Optional[torch.Tensor]:
        if image_path in self.image_cache:
            return self.image_cache[image_path]

        print(f"[DenoisingReward] Caching image: {image_path}")
        try:
            image_pil = PIL.Image.open(image_path)
        except Exception as e:
            print(f"[DenoisingReward] ERROR opening image {image_path}: {e}")
            return None

        try:
            target_tensor = preprocess_target_image(image_pil, self.device, resolution=self.input_resolution)
            with torch.no_grad():
                posterior = self.vae.encode(target_tensor)
                clean_latents = posterior.latent_dist.mean  # [1,4,H/8,W/8]
                clean_latents = clean_latents * self.latent_scaling
                clean_latents = clean_latents.to(device=self.device, dtype=self.compute_dtype)
            self.image_cache[image_path] = clean_latents
            return clean_latents
        except Exception as e:
            print(f"[DenoisingReward] ERROR processing image {image_path}: {e}")
            return None

    @torch.no_grad()
    def _reward_for_prompt(
        self,
        clean_latents: torch.Tensor,
        adversarial_prompt: str,
        *,
        t_list: torch.Tensor,                       # [K] long
        noise_list: List[torch.Tensor],             # length-K list of [1,4,h,w]
        uncond_losses: torch.Tensor,                # [K] FP32 MSE_uncond for this image
    ) -> float:
        """
        Reward dla pojedynczego promptu:
            mean_t( MSE_uncond(t) - MSE_cond(t) )
        Uncond_losses są wstępnie policzone per-obraz i timestep.
        """

        # tokenize & encode conditional prompt
        ids_cond = self.tokenizer(
            adversarial_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        enc_cond = self.text_encoder(input_ids=ids_cond)[0].to(dtype=self.compute_dtype)

        # scheduler prediction type
        pred_type = (
            self.train_scheduler.config.get("prediction_type", "epsilon")
            if isinstance(self.train_scheduler.config, Mapping) else "epsilon"
        )

        K = int(self.reward_num_timesteps)
        assert t_list.shape[0] == K and len(noise_list) == K
        assert uncond_losses.shape[0] == K

        improvements: List[torch.Tensor] = []
        for k in range(K):
            t = t_list[k].view(1)              # [1]
            noise = noise_list[k]              # [1,4,H/8,W/8]
            noisy = self.train_scheduler.add_noise(clean_latents, noise, t)

            # UNet forward, conditional
            pred_c = self.unet(noisy, t, encoder_hidden_states=enc_cond).sample

            # compute correct target
            if pred_type == "epsilon":
                target = noise
            elif pred_type in ("v_prediction", "v-prediction", "v"):
                target = self.train_scheduler.get_velocity(clean_latents, noise, t)
            else:
                target = noise

            # MSE w FP32
            l_c = F.mse_loss(pred_c.float(), target.float(), reduction="mean")
            # cached MSE_uncond dla tego timesteppu
            l_u = uncond_losses[k]

            improvements.append((l_u - l_c).float())          # >0 oznacza lepiej z kondycją

        mean_improvement = torch.stack(improvements).float().mean()
        return float(mean_improvement.item())

    # --- generacja do wizualizacji ---

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        input_ids = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        text_embeddings = self.text_encoder(input_ids=input_ids)[0].to(dtype=self.compute_dtype)

        uncond_input_ids = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(self.device)
        uncond_embeddings = self.text_encoder(input_ids=uncond_input_ids)[0].to(dtype=self.compute_dtype)

        scheduler = copy.deepcopy(self.sample_scheduler)
        scheduler.set_timesteps(num_inference_steps, device=self.device)

        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)

        latents = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8),
            generator=gen,
            device=self.device,
            dtype=self.compute_dtype,
        )
        latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_model_input = scheduler.scale_model_input(latents, timestep=t)

            noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=uncond_embeddings).sample
            noise_pred_text = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / self.latent_scaling
        image = self.vae.decode(latents.to(dtype=torch.float32)).sample  # decode w FP32
        image = (image / 2 + 0.5).clamp(0, 1)
        image_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        return PIL.Image.fromarray(image_np)

    # --- główne wywołanie ---

    @torch.no_grad()
    def __call__(self, completions: List[str], **kwargs) -> Tuple[List[float], Optional[List[Dict[str, Any]]]]:
        image_paths: List[str] = kwargs.get("target_img", [])
        step: int = int(kwargs.get("step", -1))
        mode = kwargs.get("mode", False)
        original_prompt: str = kwargs.get("original_prompt", "oopsie")

        if len(image_paths) != len(completions):
            print(f"[DenoisingReward] WARNING: len(image_paths)={len(image_paths)} != len(completions)={len(completions)}")

        batch_size = len(completions)
        rewards: List[float] = []
        adversarial_prompts: List[str] = []
        images: Optional[List[Dict[str, Any]]] = None

        # RNG deterministyczny per-step (jeśli chcesz różne seedy między runami, dodaj offset per-run)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed + max(0, step))

        # cache latentów dla obrazów w batchu
        latents_per_img: Dict[str, Optional[torch.Tensor]] = {}
        for img_path in image_paths:
            if img_path and img_path not in latents_per_img:
                latents_per_img[img_path] = self._get_cached_image_latent(img_path)

        # pre-losowanie wspólnej listy kroków t (K) oraz szumów per obraz i per krok
        K = int(self.reward_num_timesteps)
        T = int(self.train_scheduler.config.num_train_timesteps)
        t_list = torch.randint(low=0, high=T, size=(K,), device=self.device, generator=gen).long()

        noise_bank: Dict[str, List[torch.Tensor]] = {}
        for img_path, latents in latents_per_img.items():
            if latents is None:
                continue
            noise_bank[img_path] = [_randn_like(latents, generator=gen) for _ in range(K)]

        # --- precompute unconditional losses per image and timestep (cache) ---
        # embeddings dla uncond ("")
        uncond_ids = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(self.device)
        enc_uncond = self.text_encoder(input_ids=uncond_ids)[0].to(dtype=self.compute_dtype)

        pred_type = (
            self.train_scheduler.config.get("prediction_type", "epsilon")
            if isinstance(self.train_scheduler.config, Mapping) else "epsilon"
        )

        uncond_loss_bank: Dict[str, torch.Tensor] = {}
        for img_path, clean_latents in latents_per_img.items():
            if clean_latents is None:
                continue
            noise_list = noise_bank.get(img_path, None)
            if noise_list is None:
                continue

            per_t_losses: List[torch.Tensor] = []
            for k in range(K):
                t = t_list[k].view(1)
                noise = noise_list[k]
                noisy = self.train_scheduler.add_noise(clean_latents, noise, t)

                pred_u = self.unet(noisy, t, encoder_hidden_states=enc_uncond).sample

                if pred_type == "epsilon":
                    target = noise
                elif pred_type in ("v_prediction", "v-prediction", "v"):
                    target = self.train_scheduler.get_velocity(clean_latents, noise, t)
                else:
                    target = noise

                l_u = F.mse_loss(pred_u.float(), target.float(), reduction="mean")
                per_t_losses.append(l_u.float())

            uncond_loss_bank[img_path] = torch.stack(per_t_losses, dim=0)  # [K]

        # --- pętla po promptach ---
        for i in range(batch_size):
            generated_text = completions[i]
            img_path = image_paths[i] if i < len(image_paths) else None

            try:
                match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
                adversarial_prompt = (match.group(1) if match else generated_text).strip()[:1024]
            except Exception as e:
                print(f"[DenoisingReward] Error parsing completion[{i}]: {e}")
                adversarial_prompt = generated_text.strip()[:1024]

            adversarial_prompts.append(adversarial_prompt)

            if not img_path:
                print(f"[DenoisingReward] Missing image path for sample {i}")
                rewards.append(0.0)
                continue

            clean_latents = latents_per_img.get(img_path, None)
            if clean_latents is None:
                rewards.append(0.0)
                continue

            noise_list = noise_bank.get(img_path, None)
            if noise_list is None:
                rewards.append(0.0)
                continue

            uncond_losses = uncond_loss_bank.get(img_path, None)
            if uncond_losses is None:
                rewards.append(0.0)
                continue

            r = self._reward_for_prompt(
                clean_latents,
                adversarial_prompt,
                t_list=t_list,
                noise_list=noise_list,
                uncond_losses=uncond_losses,
            )
            rewards.append(float(r))

        # wizualizacja co 50 kroków lub w trybie eval (bez zmian)
        if ((step + 1) % 50 == 0 or mode == "eval") and adversarial_prompts:
            images = []
            if image_paths:
                try:
                    target_img_path = image_paths[0]
                    target_image = PIL.Image.open(target_img_path).convert("RGB")
                    images.append({"target": target_image})
                except Exception as e:
                    print(f"[DenoisingReward] ERROR loading target image for viz: {e}")

            print(f"[DenoisingReward] Step {step}: generating {len(adversarial_prompts)} images for visualization")

            for prompt in adversarial_prompts:
                try:
                    sample = {"prompt": prompt}
                    img = self.generate_image(prompt=prompt, height=512, width=512)
                    sample["generated"] = img
                    images.append(sample)
                except Exception as e:
                    print(f"[DenoisingReward] ERROR generating sample for prompt: {e}")

            try:
                base_sample = {"original_prompt": original_prompt}
                img = self.generate_image(prompt=original_prompt, height=512, width=512)
                base_sample["generated"] = img
                images.append(base_sample)
            except Exception as e:
                print(f"[DenoisingReward] ERROR generating original_prompt sample: {e}")

        return rewards, images






# --- rejestracja ORMów ---
orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'denoising': DenoisingReward,
}
