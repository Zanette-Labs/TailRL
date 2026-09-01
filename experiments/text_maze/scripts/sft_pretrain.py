import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, get_cosine_schedule_with_warmup
from tokenizers import Tokenizer, models as tok_models, pre_tokenizers, AddedToken
from tqdm import tqdm
import pandas as pd
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MazeSFTDataset(Dataset):
    """Maze SFT Dataset，支持parquet和json格式"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        start_time = time.time()
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # 加载数据
        if data_path.endswith('.parquet'):
            logger.info(f"Reading parquet file: {data_path}")
            df = pd.read_parquet(data_path)
            logger.info(f"Processing {len(df)} rows...")
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {os.path.basename(data_path)}"):
                # 从extra_info中提取answer作为完整序列
                extra_info = row['extra_info']
                if isinstance(extra_info, str):
                    extra_info = json.loads(extra_info)
                
                answer = extra_info.get('answer', '')
                question = extra_info.get('question', '')
                
                if answer:
                    example = self._process_sequence(answer, question)
                    if example is not None:
                        self.examples.append(example)
        else:
            # JSON格式
            logger.info(f"Reading JSON file: {data_path}")
            with open(data_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Processing {len(data)} items...")
            for item in tqdm(data, desc=f"Loading {os.path.basename(data_path)}"):
                sequence = item.get('sequence', '')
                if sequence:
                    example = self._process_sequence(sequence)
                    if example is not None:
                        self.examples.append(example)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Loaded {len(self.examples)} examples from {data_path} in {elapsed_time:.2f}s")
    
    def _process_sequence(self, sequence: str, prompt: Optional[str] = None) -> Optional[Dict]:
        """处理序列，构建input_ids和labels"""
        # 编码完整序列
        input_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # 找到PATH_START位置，只预测路径部分
        path_start_token = "PATH_START"
        path_start_id = self.tokenizer.encode(path_start_token, add_special_tokens=False)
        
        if len(path_start_id) == 0:
            return None
        
        path_start_id = path_start_id[0]
        
        try:
            path_start_idx = input_ids.index(path_start_id)
        except ValueError:
            return None
        
        # 构建labels：PATH_START之前（包括）设为-100
        labels = [-100] * (path_start_idx + 1) + input_ids[path_start_idx + 1:]
        
        # 保存prompt用于生成式评估
        if prompt is None:
            # 从sequence中提取prompt
            prompt = sequence.split("PATH_START")[0].strip() + " PATH_START"
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt": prompt,
            "full_sequence": sequence,
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "input_ids": example["input_ids"],
            "labels": example["labels"],
            "attention_mask": torch.ones_like(example["input_ids"]),
        }
    
    def get_prompt(self, idx) -> str:
        return self.examples[idx]["prompt"]
    
    def get_full_sequence(self, idx) -> str:
        return self.examples[idx]["full_sequence"]


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    max_len = max(item["input_ids"].size(0) for item in batch)
    batch_size = len(batch)
    
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class MazeValidator:
    """迷宫验证器：检查生成的路径是否正确"""
    
    def __init__(self):
        self.action_map = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1),
        }
    
    def parse_grid(self, sequence: str) -> Tuple[Optional[np.ndarray], Optional[Tuple], Optional[Tuple], Optional[int]]:
        """从序列中解析迷宫网格"""
        try:
            # 提取GRID_START和GRID_END之间的内容
            grid_start = sequence.find("GRID_START")
            grid_end = sequence.find("GRID_END")
            
            if grid_start == -1 or grid_end == -1:
                return None, None, None, None
            
            grid_content = sequence[grid_start + len("GRID_START"):grid_end].strip()
            lines = grid_content.split("NEWLINE")
            
            # 解析网格
            grid = []
            start = None
            goal = None
            
            for r, line in enumerate(lines):
                if not line.strip():
                    continue
                tokens = line.strip().split()
                row = []
                for c, token in enumerate(tokens):
                    if token == "WALL":
                        row.append(1)
                    elif token == "PATH":
                        row.append(0)
                    elif token == "START":
                        row.append(0)
                        start = (r, c)
                    elif token == "GOAL":
                        row.append(0)
                        goal = (r, c)
                    else:
                        row.append(0)
                if row:
                    grid.append(row)
            
            if not grid or start is None or goal is None:
                return None, None, None, None
            
            size = len(grid)
            return np.array(grid), start, goal, size
            
        except Exception as e:
            logger.debug(f"Error parsing grid: {e}")
            return None, None, None, None
    
    def parse_actions(self, sequence: str) -> List[str]:
        """从序列中解析动作序列"""
        try:
            path_start = sequence.find("PATH_START")
            if path_start == -1:
                return []
            
            path_content = sequence[path_start + len("PATH_START"):].strip()
            tokens = path_content.split()
            
            actions = []
            for token in tokens:
                if token in self.action_map:
                    actions.append(token)
                elif token == "DONE":
                    break
            
            return actions
        except:
            return []
    
    def validate_path(self, sequence: str) -> Tuple[bool, str]:
        """验证路径是否正确到达目标"""
        grid, start, goal, size = self.parse_grid(sequence)
        
        if grid is None:
            return False, "invalid_grid"
        
        actions = self.parse_actions(sequence)
        
        if not actions:
            return False, "no_actions"
        
        # 模拟执行路径
        current = start
        for action in actions:
            if action not in self.action_map:
                return False, "invalid_action"
            
            dr, dc = self.action_map[action]
            new_r, new_c = current[0] + dr, current[1] + dc
            
            # 检查边界
            if not (0 <= new_r < size and 0 <= new_c < size):
                return False, "out_of_bounds"
            
            # 检查是否撞墙
            if grid[new_r, new_c] == 1:
                return False, "hit_wall"
            
            current = (new_r, new_c)
        
        # 检查是否到达目标
        if current == goal:
            return True, "success"
        else:
            return False, "not_at_goal"
    
    def compute_optimal_length(self, sequence: str) -> Optional[int]:
        """使用BFS计算最优路径长度"""
        grid, start, goal, size = self.parse_grid(sequence)
        
        if grid is None:
            return None
        
        # BFS
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            (r, c), dist = queue.popleft()
            
            if (r, c) == goal:
                return dist
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if grid[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))
        
        return None


MAZE_VOCAB = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "GRID_START": 4, "GRID_END": 5, "PATH_START": 6, "DONE": 7,
    "PATH": 8, "WALL": 9, "GOAL": 10, "START": 11, "NEWLINE": 12,
    "UP": 13, "DOWN": 14, "LEFT": 15, "RIGHT": 16, "\n": 17,
}


def create_model_from_scratch(args):
    num_reserved = 14
    vocab_size = len(MAZE_VOCAB) + num_reserved

    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        dtype="bfloat16",
        use_cache=True,
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    reserved = {f"RESERVED_{i}": len(MAZE_VOCAB) + i for i in range(num_reserved)}
    full_vocab = {**MAZE_VOCAB, **reserved}
    tok = Tokenizer(tok_models.WordLevel(vocab=full_vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    for t in ["<pad>", "<bos>", "<eos>", "<unk>"]:
        tok.add_special_tokens([AddedToken(t, special=True)])

    save_dir = os.path.join(args.output_dir, "init_model")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save(os.path.join(save_dir, "tokenizer.json"))
    for fname, data in [
        ("tokenizer_config.json", {
            "bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>",
            "pad_token": "<pad>", "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": args.max_position_embeddings,
        }),
        ("special_tokens_map.json", {
            "bos_token": "<bos>", "eos_token": "<eos>",
            "unk_token": "<unk>", "pad_token": "<pad>",
        }),
    ]:
        with open(os.path.join(save_dir, fname), "w") as f:
            json.dump(data, f, indent=2)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized from scratch: {num_params:,} params, vocab_size={vocab_size}, saved to {save_dir}")
    return save_dir


class MazeSFTTrainer:
    """Maze SFT Trainer with Generative Evaluation"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_path = args.model_path
        if model_path is None:
            logger.info("No model_path provided, creating model from scratch...")
            model_path = create_model_from_scratch(args)

        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载数据集
        logger.info("Loading datasets...")
        dataset_load_start = time.time()
        self.train_dataset = MazeSFTDataset(args.train_data, self.tokenizer, args.max_length)
        self.val_dataset = MazeSFTDataset(args.val_data, self.tokenizer, args.max_length)
        dataset_load_time = time.time() - dataset_load_start
        logger.info(f"All datasets loaded in {dataset_load_time:.2f}s")
        
        # 创建DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.tokenizer.pad_token_id),
            num_workers=4,
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.micro_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.tokenizer.pad_token_id),
            num_workers=4,
            pin_memory=True,
        )
        
        # 优化器和调度器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        
        total_steps = len(self.train_loader) * args.num_epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        
        # 选择学习率调度器
        if args.lr_scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif args.lr_scheduler == "constant":
            # 常数学习率（带warmup）
            from transformers import get_constant_schedule_with_warmup
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")
        
        # 验证器
        self.validator = MazeValidator()
        
        # 日志
        self.global_step = 0
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        self.eval_schedule = self._parse_eval_schedule(args.eval_schedule)
        self.eval_metrics_path = os.path.join(args.output_dir, "eval_metrics.jsonl")
        
        # 初始化wandb
        self.use_wandb = args.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=args.project_name,
                name=args.experiment_name,
                config={
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "micro_batch_size": args.micro_batch_size,
                    "num_epochs": args.num_epochs,
                    "max_length": args.max_length,
                    "model_path": args.model_path,
                    "train_data": args.train_data,
                    "val_data": args.val_data,
                }
            )
            logger.info("Wandb initialized successfully")
        elif args.use_wandb and not WANDB_AVAILABLE:
            logger.warning("Wandb requested but not installed. Install with: pip install wandb")
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
        if self.eval_schedule:
            logger.info(f"Eval schedule: {self.eval_schedule}")

    @staticmethod
    def _parse_eval_schedule(schedule: str) -> List[Tuple[int, int]]:
        """Parse '<end_step>:<interval>,...' eval schedule segments."""
        if not schedule:
            return []

        segments = []
        for raw_segment in schedule.split(","):
            segment = raw_segment.strip()
            if not segment:
                continue
            try:
                end_step_str, interval_str = segment.split(":", 1)
                end_step = int(end_step_str)
                interval = int(interval_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid --eval_schedule segment '{segment}'. "
                    "Expected '<end_step>:<interval>', e.g. '2000:200,6000:50'."
                ) from exc
            if end_step <= 0 or interval <= 0:
                raise ValueError("--eval_schedule end_step and interval must be positive.")
            segments.append((end_step, interval))

        segments.sort(key=lambda item: item[0])
        return segments

    def should_evaluate(self, step: int) -> bool:
        if self.eval_schedule:
            for end_step, interval in self.eval_schedule:
                if step <= end_step:
                    return step % interval == 0
            return False

        return self.args.eval_steps > 0 and step % self.args.eval_steps == 0

    def write_eval_metrics(self, step: int, metrics: Dict[str, float]):
        row = {"step": step, "time": time.time()}
        row.update(metrics)
        with open(self.eval_metrics_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """单步训练"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # 梯度累积
        micro_batch_size = self.args.micro_batch_size
        batch_size = input_ids.size(0)
        accumulation_steps = max(1, batch_size // micro_batch_size)
        
        self.optimizer.zero_grad()
        total_loss = 0.0
        
        for i in range(0, batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, batch_size)
            
            outputs = self.model(
                input_ids=input_ids[i:end_idx],
                attention_mask=attention_mask[i:end_idx],
                labels=labels[i:end_idx],
            )
            
            loss = outputs.loss / accumulation_steps
            loss.backward()
            total_loss += loss.item()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return total_loss
    
    @torch.no_grad()
    def validate_loss(self) -> float:
        """计算验证集loss"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _compute_pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        计算Pass@k指标（unbiased estimator）
        
        Pass@k的定义：从n次采样中随机选择k次，这k次中至少有1次成功的概率。
        公式：Pass@k = 1 - C(n-c, k) / C(n, k)
        
        例如：n=8次采样，c=3次成功，k=2
        Pass@2 = 从8次中随机选2次，至少对1次的概率
        
        Args:
            n: 总采样次数
            c: 成功次数
            k: 随机选择的次数
        
        Returns:
            Pass@k值
        """
        if n - c < k:
            # 失败次数不足k次，随机选k次必然包含成功的
            return 1.0
        # 计算 1 - C(n-c, k) / C(n, k)
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    @torch.no_grad()
    def generative_evaluate(self, num_samples: int = 100, n_samples_per_prompt: int = 8, temperature: float = 1.0) -> Dict[str, float]:
        """
        生成式评估：每个样本采样多次，计算Pass@k指标
        
        Pass@k的含义：从n次采样中随机选k次，至少有1次成功的概率
        例如 n=8, Pass@2 = 从8次采样中随机选2次，这2次中至少1次成功的概率
        
        Args:
            num_samples: 评估的prompt数量
            n_samples_per_prompt: 每个prompt的采样次数 (n)
            temperature: 采样温度
        
        Returns:
            包含Pass@1/2/4/8等指标的字典
        """
        self.model.eval()

        # 记录每个样本的成功次数和最优次数
        all_success_counts = []  # 每个样本成功的次数
        all_optimal_counts = []  # 每个样本达到最优的次数
        all_continuous_rewards = []  # 每个 prompt 的 n_samples 平均 continuous reward
        
        # 错误统计
        error_stats = {
            "invalid_grid": 0,
            "no_actions": 0,
            "invalid_action": 0,
            "out_of_bounds": 0,
            "hit_wall": 0,
            "not_at_goal": 0,
        }
        total_generations = 0
        
        # 随机选择样本
        indices = np.random.choice(len(self.val_dataset), min(num_samples, len(self.val_dataset)), replace=False)
        
        # 获取eos_token_id
        try:
            done_token_id = self.tokenizer.encode("DONE", add_special_tokens=False)[0]
        except:
            done_token_id = self.tokenizer.eos_token_id
        
        for idx in tqdm(indices, desc="Generative Eval"):
            prompt = self.val_dataset.get_prompt(idx)
            full_sequence = self.val_dataset.get_full_sequence(idx)
            optimal_len = self.validator.compute_optimal_length(full_sequence)
            
            # 编码prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            success_count = 0
            optimal_count = 0
            cont_reward_sum = 0.0  # sum over n_samples_per_prompt of judge_maze_continuous-style reward

            # 对每个prompt采样n次
            try:
                # 批量生成n_samples_per_prompt个样本
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=n_samples_per_prompt,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=done_token_id,
                )
                
                # 验证每个生成的路径
                for i in range(n_samples_per_prompt):
                    generated = self.tokenizer.decode(output_ids[i], skip_special_tokens=False)
                    
                    success, reason = self.validator.validate_path(generated)
                    total_generations += 1
                    
                    if success:
                        success_count += 1
                        generated_actions = self.validator.parse_actions(generated)
                        L_gen = len(generated_actions)
                        if optimal_len is not None and L_gen == optimal_len:
                            optimal_count += 1
                        # judge_maze_continuous-style reward (matches verl/utils/reward_score/maze.py)
                        if optimal_len is not None and self.args.eval_ub > optimal_len:
                            if L_gen >= self.args.eval_ub:
                                cont_r = 0.0
                            else:
                                cont_r = (self.args.eval_ub - L_gen) / (self.args.eval_ub - optimal_len)
                                if cont_r < 0.0:
                                    cont_r = 0.0
                                elif cont_r > 1.0:
                                    cont_r = 1.0
                            cont_reward_sum += cont_r
                    else:
                        error_stats[reason] = error_stats.get(reason, 0) + 1
                        
            except Exception as e:
                logger.debug(f"Generation error: {e}")
                total_generations += n_samples_per_prompt
            
            all_success_counts.append(success_count)
            all_optimal_counts.append(optimal_count)
            all_continuous_rewards.append(cont_reward_sum / max(1, n_samples_per_prompt))
        
        # 计算Pass@k指标
        n = n_samples_per_prompt
        k_values = [1, 2, 4, 8]
        
        metrics = {}
        
        # Pass@k for success (到达目标)
        for k in k_values:
            if k <= n:
                pass_at_k_values = [self._compute_pass_at_k(n, c, k) for c in all_success_counts]
                metrics[f"eval/pass@{k}"] = np.mean(pass_at_k_values) if pass_at_k_values else 0.0
        
        # Pass@k for optimal (最优路径)
        for k in k_values:
            if k <= n:
                pass_at_k_values = [self._compute_pass_at_k(n, c, k) for c in all_optimal_counts]
                metrics[f"eval/optimal_pass@{k}"] = np.mean(pass_at_k_values) if pass_at_k_values else 0.0
        
        # 错误率统计
        total = max(total_generations, 1)
        metrics["eval/invalid_rate"] = (error_stats["invalid_grid"] + error_stats["no_actions"] + error_stats["invalid_action"]) / total
        metrics["eval/wall_hit_rate"] = error_stats["hit_wall"] / total
        metrics["eval/not_at_goal_rate"] = error_stats["not_at_goal"] / total
        
        # 平均成功率（所有生成的平均）
        metrics["eval/avg_success_rate"] = sum(all_success_counts) / max(total_generations, 1)
        metrics["eval/avg_optimal_rate"] = sum(all_optimal_counts) / max(total_generations, 1)
        # Continuous reward (mean over prompts of avg-over-n-samples)
        if all_continuous_rewards:
            metrics["eval/continuous_reward_mean"] = float(np.mean(all_continuous_rewards))

        return metrics
    
    def save_checkpoint(self, step: int):
        """保存checkpoint"""
        path = os.path.join(self.args.output_dir, f"ckpt-{step}")
        os.makedirs(path, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        logger.info(f"Saved checkpoint to {path}")
    
    def train(self):
        """训练主循环"""
        logger.info("Starting training...")
        
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            for batch in pbar:
                self.global_step += 1
                
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })
                
                # 记录训练loss到wandb
                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss,
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch + 1,
                    }, step=self.global_step)
                
                # 评估
                pass1_this_step = None
                did_eval = False
                if self.should_evaluate(self.global_step):
                    did_eval = True
                    val_loss = self.validate_loss()
                    logger.info(f"Step {self.global_step} - Val Loss: {val_loss:.4f}")

                    eval_metrics = {"eval/val_loss": val_loss}
                    
                    if self.args.use_generative_eval:
                        metrics = self.generative_evaluate(
                            num_samples=self.args.eval_samples,
                            n_samples_per_prompt=self.args.n_samples_per_prompt,
                            temperature=self.args.eval_temperature,
                        )
                        eval_metrics.update(metrics)
                        logger.info(
                            f"Step {self.global_step} - Pass@1={metrics.get('eval/pass@1', 0):.4f}, "
                            f"Pass@2={metrics.get('eval/pass@2', 0):.4f}, "
                            f"Pass@4={metrics.get('eval/pass@4', 0):.4f}, "
                            f"Pass@8={metrics.get('eval/pass@8', 0):.4f}"
                        )
                        logger.info(
                            f"Step {self.global_step} - Optimal Pass@1={metrics.get('eval/optimal_pass@1', 0):.4f}, "
                            f"Avg Success={metrics.get('eval/avg_success_rate', 0):.4f}"
                        )
                    
                    if self.use_wandb:
                        wandb.log(eval_metrics, step=self.global_step)
                    self.write_eval_metrics(self.global_step, eval_metrics)
                    pass1_this_step = eval_metrics.get("eval/pass@1")

                # 保存checkpoint
                if self.args.save_on_eval and did_eval:
                    self.save_checkpoint(self.global_step)
                elif self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint(self.global_step)

                # Early-exit conditions: pass@1 hit OR step cap reached.
                if (
                    self.args.early_stop_pass1 > 0
                    and pass1_this_step is not None
                    and pass1_this_step >= self.args.early_stop_pass1
                ):
                    logger.info(
                        f"Early stop: pass@1={pass1_this_step:.4f} >= {self.args.early_stop_pass1} "
                        f"at step {self.global_step}."
                    )
                    self.save_checkpoint(self.global_step)
                    if self.use_wandb:
                        wandb.finish()
                    return
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    logger.info(
                        f"Reached max_steps={self.args.max_steps} at step {self.global_step}."
                    )
                    self.save_checkpoint(self.global_step)
                    if self.use_wandb:
                        wandb.finish()
                    return

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # 保存最终模型
        self.save_checkpoint(self.global_step)

        # 关闭wandb
        if self.use_wandb:
            wandb.finish()

        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Maze SFT Trainer")
    
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model (if omitted, trains from scratch)")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    
    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Micro batch size for gradient accumulation")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "constant"], help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio for learning rate scheduler")
    
    # 评估和保存参数
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=25, help="Evaluate every N steps")
    parser.add_argument("--eval_schedule", type=str, default="", help="Comma-separated '<end_step>:<interval>' eval schedule, e.g. '2000:200,6000:50'. Overrides --eval_steps.")
    parser.add_argument("--save_on_eval", action="store_true", help="Save a checkpoint immediately after each evaluation.")
    parser.add_argument("--use_generative_eval", action="store_true", help="Use generative evaluation (RL-style)")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of prompts for generative evaluation")
    parser.add_argument("--n_samples_per_prompt", type=int, default=8, help="Number of samples per prompt for Pass@k evaluation")
    parser.add_argument("--eval_temperature", type=float, default=1.0, help="Temperature for sampling during evaluation")
    parser.add_argument("--eval_ub", type=int, default=60, help="UB for the continuous-reward eval metric (matches judge_maze_continuous).")
    parser.add_argument("--early_stop_pass1", type=float, default=0.0, help="If > 0, exit when eval/pass@1 reaches this value.")
    parser.add_argument("--max_steps", type=int, default=0, help="If > 0, exit training once global_step reaches this value.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for python/numpy/torch RNG (deterministic shuffling).")
    
    # 日志参数
    parser.add_argument("--project_name", type=str, default="maze-sft", help="Project name for logging")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Experiment name")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    
    args = parser.parse_args()

    # Determinism: same seed -> same DataLoader shuffle order across runs.
    import random as pyrandom
    pyrandom.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建训练器并开始训练
    trainer = MazeSFTTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
