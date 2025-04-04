# data_process.py
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from typing import List, Dict, Any
import yaml
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """데이터 로딩, 증강, 전처리 클래스"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path (str): 설정 파일 경로
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.data_config = self.config['data']
        self.augmentation_config = self.data_config.get('augmentation', {})

    def load_data(self) -> pd.DataFrame:
        """CSV 데이터 로드"""
        input_file = self.data_config['input_file']
        logger.info(f"데이터 로드 중: {input_file}")
        try:
            # Hugging Face datasets 라이브러리 사용 고려 (더 큰 데이터셋, 스트리밍 등 지원)
            # ds = load_dataset("csv", data_files=input_file)
            # return ds['train'].to_pandas() # 예시
            return pd.read_csv(input_file)
        except FileNotFoundError:
            logger.error(f"데이터 파일을 찾을 수 없습니다: {input_file}")
            raise
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {e}")
            raise

    def _apply_variations(self, text: str, column_name: str) -> str:
        """지정된 컬럼에 대해 정의된 변형 중 하나를 랜덤하게 적용"""
        if column_name not in self.augmentation_config.get('variations', {}):
            return text

        variations = self.augmentation_config['variations'][column_name]
        if not variations or len(variations) < 2:
            return text # 원본 또는 대체어가 없으면 변형 불가

        original_keyword = variations[0]
        replacement_keywords = variations[1:]

        # 원본 키워드가 포함된 경우에만 변형 시도
        if original_keyword in text:
            chosen_replacement = np.random.choice(replacement_keywords)
            # 첫 번째 발생만 바꾸거나, 모두 바꿀 수 있음 (여기서는 첫 번째만)
            return text.replace(original_keyword, chosen_replacement, 1)
        else:
            return text # 원본 키워드가 없으면 그대로 반환

    def augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 증강: 복제 및 키워드 변형"""
        if not self.augmentation_config.get('apply', False):
            logger.info("데이터 증강 설정이 비활성화되어 건너<0xEB><0x8A>니다.")
            return df

        repeat_count = self.augmentation_config.get('repeat_count', 1)
        if repeat_count <= 1 and not self.augmentation_config.get('variations'):
             logger.info("데이터 증강 설정(반복/변형)이 없어 원본 데이터를 사용합니다.")
             return df

        logger.info(f"데이터 증강 시작: {repeat_count}배 복제 및 변형 적용")

        # 데이터 복제
        dfs_to_concat = [df.copy()] # 원본 포함
        if repeat_count > 1:
            dfs_to_concat.extend([df.copy() for _ in range(repeat_count - 1)])

        df_augmented = pd.concat(dfs_to_concat, ignore_index=True)

        # 변형 적용
        variation_config = self.augmentation_config.get('variations', {})
        if variation_config:
            logger.info(f"키워드 변형 적용 대상: {list(variation_config.keys())}")
            for col_name in variation_config.keys():
                if col_name in df_augmented.columns:
                    df_augmented[col_name] = df_augmented[col_name].astype(str).apply(
                        lambda x: self._apply_variations(x, col_name)
                    )
                else:
                    logger.warning(f"증강 설정에 있는 컬럼 '{col_name}'이 데이터프레임에 없습니다.")

        logger.info(f"데이터 증강 완료. 원본 {len(df)}개 -> 증강 후 {len(df_augmented)}개")
        return df_augmented

    def _create_prompt(self, instruction: str, input_text: str, output: str) -> str:
         """개별 데이터 포인트에 대한 프롬프트 생성"""
         # 프롬프트 템플릿 (필요에 따라 수정)
         if input_text and pd.notna(input_text): # Input이 있는 경우
             return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
         else: # Input이 없는 경우 (Instruction-Output 만)
             return f"Instruction: {instruction}\nOutput: {output}"

    def preprocess_function(self, examples: Dict[str, List[str]], tokenizer: Any) -> Dict[str, List[int]]:
        """데이터 전처리 및 토큰화 함수 (Dataset.map 에 사용)"""
        # 프롬프트 생성
        prompts = [
            self._create_prompt(instruction, input_text, output)
            for instruction, input_text, output in zip(
                examples["instruction"],
                examples["input"],
                examples["output"]
            )
        ]

        # 토큰화 (config에서 max_length 읽기)
        max_length = self.data_config.get('max_length', 512)
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length", # 또는 "longest"
            # return_tensors="pt" # Trainer 사용 시 필요 없음, map에서 자동으로 처리
        )

        # Causal LM 파인튜닝 시에는 input_ids와 attention_mask 외에 'labels'도 필요
        # labels는 padding 토큰 위치를 -100으로 마스킹하여 손실 계산에서 제외
        tokenized["labels"] = [
             [(l if l != tokenizer.pad_token_id else -100) for l in label_ids]
             for label_ids in tokenized["input_ids"]
        ]

        return tokenized

    def prepare_dataset(self, tokenizer: Any) -> Dataset:
        """데이터 로드, 증강, 전처리 및 Dataset 객체 반환"""
        # 데이터 로드
        df = self.load_data()

        # 필수 컬럼 확인
        required_columns = ["instruction", "input", "output"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"데이터에 필수 컬럼이 없습니다: {missing}. 'instruction', 'input', 'output' 컬럼이 필요합니다.")

        # 데이터 증강
        df_processed = self.augment_data(df)

        # Dataset으로 변환
        dataset = Dataset.from_pandas(df_processed)

        # 전처리 및 토큰화 적용
        logger.info("데이터 전처리 및 토큰화 시작...")
        tokenized_dataset = dataset.map(
            lambda x: self.preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names # 원본 컬럼 제거
        )
        logger.info("데이터 전처리 및 토큰화 완료.")

        # 데이터셋 분할 (예: train/validation) - 필요 시 추가
        # tokenized_datasets = tokenized_dataset.train_test_split(test_size=0.1)
        # return tokenized_datasets['train'], tokenized_datasets['test']

        return tokenized_dataset