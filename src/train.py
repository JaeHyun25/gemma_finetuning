import os
import yaml
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from data_processor import DataProcessor
from model_utils import ModelUtils
import logging
import torch
import torch.distributed as dist

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def setup_distributed():
    """분산 학습 설정"""
    if torch.cuda.is_available():
        # CUDA 초기화
        torch.cuda.empty_cache()
        
        # NCCL 백엔드 설정
        if torch.cuda.device_count() > 1:
            # NCCL 환경 변수 설정
            os.environ['NCCL_DEBUG'] = 'INFO'
            os.environ['NCCL_IB_DISABLE'] = '0'
            os.environ['NCCL_P2P_DISABLE'] = '0'
            
            # NVLink 사용 설정 (가능한 경우)
            if hasattr(torch.cuda, 'nccl'):
                torch.cuda.nccl.set_nccl_options(
                    nccl_use_nvlink=True,
                    nccl_use_p2p=True
                )

def main():
    """메인 함수"""
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # GPU 정보 출력
        device_count = torch.cuda.device_count()
        logger.info(f"사용 가능한 GPU 개수: {device_count}")
        for i in range(device_count):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_props.name}")
            logger.info(f"  - 메모리: {gpu_props.total_memory / 1024**3:.2f}GB")
            logger.info(f"  - CUDA 기능: {gpu_props.major}.{gpu_props.minor}")
            
            # NVLink 지원 확인
            if hasattr(gpu_props, 'multi_processor_count'):
                logger.info(f"  - 멀티 프로세서 수: {gpu_props.multi_processor_count}")
        
        # 분산 학습 설정
        setup_distributed()
        
        # 설정 파일 경로
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'config', 'config.yaml')
        
        # 모델과 토크나이저 로드
        logger.info("모델과 토크나이저 로드 중...")
        model_utils = ModelUtils(config_path)
        model, tokenizer = model_utils.load_model_and_tokenizer()
        
        # 데이터셋 준비
        logger.info("데이터셋 준비 중...")
        data_processor = DataProcessor(config_path)
        dataset = data_processor.prepare_dataset(tokenizer)
        
        # 학습 인자 설정
        logger.info("학습 인자 설정 중...")
        training_args = TrainingArguments(
            output_dir=model_utils.config['model']['output_dir'],
            num_train_epochs=model_utils.config['training']['epochs'],
            per_device_train_batch_size=model_utils.config['training']['batch_size'],
            gradient_accumulation_steps=model_utils.config['training']['gradient_accumulation_steps'],
            learning_rate=model_utils.config['training']['learning_rate'],
            weight_decay=model_utils.config['training']['weight_decay'],
            warmup_steps=model_utils.config['training']['warmup_steps'],
            logging_steps=model_utils.config['training']['logging_steps'],
            save_steps=model_utils.config['training']['save_steps'],
            fp16=model_utils.config['training']['fp16'],
            optim=model_utils.config['training']['optim'],
            ddp_find_unused_parameters=model_utils.config['training']['ddp_find_unused_parameters'],
            local_rank=model_utils.config['training']['local_rank'],
            dataloader_num_workers=model_utils.config['training']['dataloader_num_workers'],
            remove_unused_columns=model_utils.config['training']['remove_unused_columns'],
            # 멀티 GPU 설정
            parallel_mode="distributed" if device_count > 1 else "single",
            ddp_backend="nccl" if device_count > 1 else "gloo",
            gradient_checkpointing=True,  # 메모리 효율을 위해 활성화
            max_grad_norm=1.0,  # 그래디언트 클리핑
            report_to="tensorboard",  # 학습 모니터링
            # 성능 최적화
            dataloader_pin_memory=True,  # 데이터 로딩 최적화
            dataloader_prefetch_factor=2,  # 데이터 프리페칭
            gradient_checkpointing_kwargs={"use_reentrant": False},  # 그래디언트 체크포인팅 최적화
            # 메모리 최적화
            optim="paged_adamw_8bit",  # 8비트 옵티마이저
            max_grad_norm=1.0,  # 그래디언트 클리핑
            # 로깅 설정
            logging_first_step=True,
            logging_nan_inf_filter=False
        )
        
        # Trainer 초기화
        logger.info("Trainer 초기화 중...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )
        
        # 학습 실행
        logger.info("학습 시작...")
        trainer.train()
        
        # 모델 저장
        logger.info("모델 저장 중...")
        trainer.save_model(model_utils.config['model']['final_output_dir'])
        
        logger.info("학습 완료!")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
