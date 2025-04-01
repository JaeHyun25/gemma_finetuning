import pandas as pd
import numpy as np
from datasets import Dataset
from typing import List, Dict, Any
import yaml

class DataProcessor:
    """데이터 전처리 및 증강을 위한 클래스"""
    
    def __init__(self, config_path: str):
        """초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
    def load_data(self) -> pd.DataFrame:
        """데이터 로드
        
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        return pd.read_csv(self.config['data']['input_file'])
    
    def augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 증강
        
        Args:
            df (pd.DataFrame): 원본 데이터
            
        Returns:
            pd.DataFrame: 증강된 데이터
        """
        # 데이터 복제
        df_repeated = pd.concat([df] * self.config['data']['augmentation']['repeat_count'], 
                              ignore_index=True)
        
        # 변형 적용
        variations = self.config['data']['augmentation']['variations']
        df_repeated["instruction"] = df_repeated["instruction"].apply(
            lambda x: np.random.choice([x.replace(variations[0], v) for v in variations[1:]])
        )
        df_repeated["input"] = df_repeated["input"].apply(
            lambda x: np.random.choice([x.replace(variations[2], v) for v in variations[3:]])
        )
        df_repeated["output"] = df_repeated["output"].apply(
            lambda x: np.random.choice([x.replace(variations[4], v) for v in variations[5:]])
        )
        
        return df_repeated
    
    def preprocess_function(self, examples: Dict[str, List[str]], tokenizer: Any) -> Dict[str, List[int]]:
        """데이터 전처리 함수
        
        Args:
            examples (Dict[str, List[str]]): 입력 데이터
            tokenizer: 토크나이저 객체
            
        Returns:
            Dict[str, List[int]]: 전처리된 데이터
        """
        # 프롬프트 템플릿
        prompt_template = (
            "Instruction: {instruction}\n"
            "Input: {input}\n"
            "Output: {output}"
        )
        
        # 프롬프트 생성
        prompts = [
            prompt_template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
            for instruction, input_text, output in zip(
                examples["instruction"],
                examples["input"],
                examples["output"]
            )
        ]
        
        # 토큰화
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        return tokenized
    
    def prepare_dataset(self, tokenizer: Any) -> Dataset:
        """데이터셋 준비
        
        Args:
            tokenizer: 토크나이저 객체
            
        Returns:
            Dataset: 준비된 데이터셋
        """
        # 데이터 로드
        df = self.load_data()
        
        # 데이터 증강
        df_augmented = self.augment_data(df)
        
        # Dataset으로 변환
        dataset = Dataset.from_pandas(df_augmented)
        
        # 전처리 적용
        tokenized_dataset = dataset.map(
            lambda x: self.preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset 