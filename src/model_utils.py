import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import yaml
from typing import Tuple, Any

class ModelUtils:
    """모델 관련 유틸리티 클래스"""
    
    def __init__(self, config_path: str):
        """초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get_quantization_config(self) -> BitsAndBytesConfig:
        """양자화 설정 생성
        
        Returns:
            BitsAndBytesConfig: 양자화 설정
        """
        return BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype']),
            bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
            llm_int8_enable_fp32_cpu_offload=self.config['quantization']['llm_int8_enable_fp32_cpu_offload']
        )
    
    def get_lora_config(self) -> LoraConfig:
        """LoRA 설정 생성
        
        Returns:
            LoraConfig: LoRA 설정
        """
        return LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type']
        )
    
    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """모델과 토크나이저 로드
        
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: 로드된 모델과 토크나이저
        """
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['path'],
            trust_remote_code=True
        )
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['path'],
            quantization_config=self.get_quantization_config(),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # LoRA 적용
        model = get_peft_model(model, self.get_lora_config())
        model.print_trainable_parameters()
        
        return model, tokenizer 