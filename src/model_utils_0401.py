# model_utils.py
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model
import yaml
from typing import Tuple, Optional, Dict, Any
import logging
from accelerate import init_empty_weights, infer_auto_device_map # accelerate 라이브러리 추가

logger = logging.getLogger(__name__)

class ModelUtils:
    """모델 로딩, 설정 관련 유틸리티 클래스"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path (str): 설정 파일 경로
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # GPU 정보 로깅 (스크립트 시작 시 한 번만)
        self.device_count = torch.cuda.device_count()
        logger.info(f"사용 가능한 GPU 개수: {self.device_count}")
        if self.device_count > 0:
            for i in range(self.device_count):
                gpu_props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {gpu_props.name} | Memory: {gpu_props.total_memory / 1024**3:.2f}GB | CUDA Arch: {gpu_props.major}.{gpu_props.minor}")
        else:
            logger.warning("사용 가능한 GPU가 없습니다. CPU를 사용합니다.")

        # CUDA 설정 (필요 시)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True # 고정된 입력 크기 시 성능 향상
            # torch.cuda.set_per_process_memory_fraction(0.95) # 필요 시 주석 해제 (OOM 문제 발생 시)

    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """양자화 설정 생성"""
        q_config = self.config.get('quantization', {})
        if not q_config.get('load_in_4bit') and not q_config.get('load_in_8bit'):
             return None

        compute_dtype_str = q_config.get('bnb_4bit_compute_dtype', 'float16')
        compute_dtype = getattr(torch, compute_dtype_str, torch.float16)

        return BitsAndBytesConfig(
            load_in_4bit=q_config.get('load_in_4bit', False),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=q_config.get('bnb_4bit_use_double_quant', False),
            bnb_4bit_quant_type=q_config.get('bnb_4bit_quant_type', 'nf4'),
            load_in_8bit=q_config.get('load_in_8bit', False), # 8비트 옵션도 고려
            # llm_int8_enable_fp32_cpu_offload 옵션은 4bit 로딩 시 불필요
        )

    def get_lora_config(self) -> LoraConfig:
        """LoRA 설정 생성"""
        lora_config = self.config['lora']
        return LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            task_type=lora_config['task_type']
        )

    def _get_auto_device_map(self, model_name: str, trust_remote_code: bool) -> Dict[str, Any]:
        """accelerate를 사용하여 자동 device map 추론"""
        if self.device_count <= 1:
            return "auto" # 단일 GPU는 간단히 "auto" 사용

        try:
            model_config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            # 빈 모델로 초기화하여 메모리 요구량 추정
            with init_empty_weights():
                model_empty = AutoModelForCausalLM.from_config(
                    model_config, trust_remote_code=trust_remote_code, torch_dtype=torch.float16
                )
                model_empty.tie_weights() # 가중치 공유 설정 (필요한 경우)

            # 각 GPU에 할당할 최대 메모리 (전체 메모리의 90% 정도 할당)
            max_memory = {
                i: f"{int(torch.cuda.get_device_properties(i).total_memory / 1024**3 * 0.9)}GB"
                for i in range(self.device_count)
            }
            # 모델 아키텍처에 따라 분할하지 않을 모듈 지정 (예: 'LlamaDecoderLayer')
            # 이 부분은 사용하는 모델에 맞게 조정해야 할 수 있습니다.
            no_split_module_classes = getattr(model_empty, "_no_split_modules", [])

            device_map = infer_auto_device_map(
                model_empty,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes
            )
            logger.info(f"자동 추론된 device_map: {device_map}")
            # 메모리 부족 시 'disk' offload 옵션 추가 고려 가능
            # device_map = infer_auto_device_map(..., offload_folder="offload_dir")
            del model_empty # 메모리 해제
            return device_map

        except Exception as e:
            logger.error(f"자동 device_map 추론 실패: {e}. 'auto'로 대체합니다.")
            return "auto"

    def load_model_and_tokenizer(self, use_auto_device_map: bool = True) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """모델과 토크나이저 로드"""
        model_name = self.config['model']['path']
        trust_remote_code = self.config['model'].get('trust_remote_code', True) # Config에 추가 가능

        # 토크나이저 로드
        logger.info(f"토크나이저 로드 중: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        # 패딩 토큰 설정 (Gemma 등 일부 모델은 PAD 토큰이 없을 수 있음)
        if tokenizer.pad_token is None:
            logger.warning("PAD 토큰이 없습니다. EOS 토큰을 PAD 토큰으로 사용합니다.")
            tokenizer.pad_token = tokenizer.eos_token

        # 모델 로드 설정
        quantization_config = self.get_quantization_config()
        if quantization_config:
             logger.info(f"양자화 설정 적용: {quantization_config}")

        # Device Map 결정
        if use_auto_device_map and self.device_count > 1:
            logger.info("자동 device_map 추론 시도...")
            device_map = self._get_auto_device_map(model_name, trust_remote_code)
        else:
            device_map = "auto" # 단일 GPU 또는 자동 추론 비활성화 시
            logger.info(f"device_map 설정: {device_map}")

        # 모델 로드
        logger.info(f"모델 로드 중: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map, # 결정된 device_map 사용
            trust_remote_code=trust_remote_code,
            torch_dtype=getattr(torch, self.config['quantization'].get('bnb_4bit_compute_dtype', 'float16'), torch.float16) if quantization_config else torch.float16, # 양자화 설정 따르거나 기본 float16
            # low_cpu_mem_usage=True # 매우 큰 모델 로드 시 CPU 메모리 절약 (device_map과 함께 사용)
        )

        # LoRA 적용
        if 'lora' in self.config:
            logger.info("LoRA 설정 적용 중...")
            lora_config = self.get_lora_config()
            model = get_peft_model(model, lora_config)
            logger.info("LoRA 적용 완료.")
            model.print_trainable_parameters()
        else:
            logger.info("LoRA 설정이 없어 적용하지 않습니다.")

        return model, tokenizer