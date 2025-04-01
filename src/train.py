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

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
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
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i} 메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f}GB")
        
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
            report_to="tensorboard"  # 학습 모니터링
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
