# train.py
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
import argparse # argparse 추가

# 로깅 설정
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# 분산 환경 설정 (NCCL 환경 변수 설정 위주)
def setup_distributed_env():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # NCCL 환경 변수 설정 (디버깅 또는 특정 네트워크 환경에 필요할 수 있음)
        # os.environ['NCCL_DEBUG'] = 'INFO' # NCCL 상세 로깅
        os.environ['NCCL_IB_DISABLE'] = '0' # InfiniBand 사용 활성화 (있을 경우)
        os.environ['NCCL_P2P_DISABLE'] = '0' # Peer-to-peer 통신 활성화
        # deprecated 된 set_nccl_options 제거
        # NVLink 등은 보통 자동으로 감지/사용됨
        logger.info("NCCL 환경 변수 설정 (IB/P2P 활성화 시도)")

def main(args):
    """메인 학습 함수"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 분산 환경 설정 (환경 변수만)
        setup_distributed_env()

        # 설정 파일 로드 (인자로 받은 경로 사용)
        config_path = args.config_path
        logger.info(f"설정 파일 로드: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 모델 유틸리티 초기화 (GPU 정보 로깅 포함)
        model_utils = ModelUtils(config_path)

        # 모델과 토크나이저 로드
        logger.info("모델과 토크나이저 로드 중...")
        # Multi-GPU 시 자동 device map 사용 여부 (config로 제어 가능하게 할 수도 있음)
        use_auto_map = config.get('training', {}).get('use_auto_device_map', True)
        model, tokenizer = model_utils.load_model_and_tokenizer(use_auto_device_map=use_auto_map)

        # 데이터셋 준비
        logger.info("데이터셋 준비 중...")
        data_processor = DataProcessor(config_path)
        # tokenizer를 prepare_dataset에 전달해야 함
        train_dataset = data_processor.prepare_dataset(tokenizer)
        # 필요 시 평가 데이터셋도 준비
        # train_dataset, eval_dataset = data_processor.prepare_dataset(tokenizer)

        # 학습 인자 설정
        logger.info("학습 인자 설정 중...")
        training_config = config['training']
        training_args = TrainingArguments(
            output_dir=config['model']['output_dir'],
            num_train_epochs=training_config['epochs'],
            per_device_train_batch_size=training_config['batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_steps=training_config['warmup_steps'],
            logging_dir=os.path.join(config['model']['output_dir'], 'logs'), # 로그 저장 경로
            logging_strategy="steps",
            logging_steps=training_config['logging_steps'],
            save_strategy="steps",
            save_steps=training_config['save_steps'],
            save_total_limit=2, # 체크포인트 최대 저장 개수
            fp16=training_config['fp16'],
            optim=training_config['optim'],
            # local_rank는 torchrun/accelerate launch가 자동으로 설정하므로 명시적 전달 X
            # local_rank=training_config['local_rank'],
            ddp_find_unused_parameters=training_config.get('ddp_find_unused_parameters'), # None이면 기본값 사용
            dataloader_num_workers=training_config['dataloader_num_workers'],
            gradient_checkpointing=training_config['gradient_checkpointing'],
            gradient_checkpointing_kwargs=training_config.get('gradient_checkpointing_kwargs', {"use_reentrant": False}),
            max_grad_norm=training_config['max_grad_norm'],
            report_to=training_config.get('report_to', "tensorboard").split(','), # 여러개 지정 가능 (쉼표 구분)
            logging_first_step=training_config.get('logging_first_step', True),
            logging_nan_inf_filter=training_config.get('logging_nan_inf_filter', True),
            # remove_unused_columns=training_config.get('remove_unused_columns', True), # 기본값 True 사용
            # Multi-GPU 설정은 Trainer가 자동으로 처리 (torchrun/accelerate launch 사용 시)
            # parallel_mode, ddp_backend 등은 보통 자동으로 설정됨
            # 성능 최적화
            dataloader_pin_memory=True, # 데이터 로딩 최적화
        )

        # Data Collator 설정 (Causal LM용)
        # labels 컬럼이 이미 preprocess_function에서 생성되었으므로 mlm=False 유지
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Trainer 초기화
        logger.info("Trainer 초기화 중...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=eval_dataset, # 평가 데이터셋이 있다면 추가
            tokenizer=tokenizer, # tokenizer 전달하면 pad 토큰 처리 등 용이
            data_collator=data_collator,
        )

        # 학습 실행
        logger.info("=== 학습 시작 ===")
        train_result = trainer.train()
        logger.info("=== 학습 완료 ===")

        # 학습 결과 저장 및 로깅
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        logger.info(f"학습 지표: {metrics}")

        # 최종 모델 저장
        final_output_dir = config['model']['final_output_dir']
        logger.info(f"최종 모델 저장 중: {final_output_dir}")
        trainer.save_model(final_output_dir) # PEFT 어댑터만 저장됨 (일반적)
        # 필요 시 전체 모델 저장 (메모리 많이 차지)
        # model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir) # 토크나이저도 저장
        logger.info("최종 모델 및 토크나이저 저장 완료.")

    except Exception as e:
        logger.exception(f"학습 중 심각한 오류 발생: {e}") # 스택 트레이스와 함께 로깅
        raise # 오류 다시 발생시켜 프로세스 종료

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 파인튜닝 스크립트")
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'), # 기본 경로 (프로젝트 구조에 맞게 수정)
        help="설정 파일(config.yaml)의 경로"
    )
    args = parser.parse_args()

    # 로거 설정 (메인 함수 호출 전에)
    # setup_logging() # 메인 함수 시작 시 호출하도록 변경
    logger = logging.getLogger(__name__) # __main__ 로거

    main(args)