import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict, Any


def _bfs_optimal_length(grid, start, goal):
    """BFS shortest path length from start to goal. Returns None if unreachable."""
    size = grid.shape[0]
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return None


def _bfs_distance(grid, pos, goal):
    """BFS distance from pos to goal on open cells. Returns None if unreachable."""
    if pos == goal:
        return 0
    size = grid.shape[0]
    queue = deque([(pos, 0)])
    visited = {pos}
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return None


def _simulate_actions(env, actions):
    """
    Simulate action sequence on a MazeEnv.
    Returns: (reached_goal, hit_wall, L, final_pos)
    L = number of actions to first goal visit (counting loops/revisits).
    On wall hit, final_pos is position BEFORE the collision.
    """
    env.reset()
    hit_wall = False
    reached_goal = False
    L = None
    final_pos = env.start
    done = False

    for i, action in enumerate(actions):
        pos, done, success = env.step(action)
        if done:
            if success:
                reached_goal = True
                L = i + 1
                final_pos = pos
            else:
                hit_wall = True
                final_pos = env.current_pos
            break
        final_pos = pos

    if not done and not hit_wall and not reached_goal:
        final_pos = env.current_pos
        if final_pos == env.goal:
            reached_goal = True
            L = len(actions)

    return reached_goal, hit_wall, L, final_pos


def _parse_actions_from_solution(solution_str):
    """Parse and validate actions from solution string. Returns list or None."""
    valid_actions = {"UP", "DOWN", "LEFT", "RIGHT"}
    tokens = solution_str.strip().split()
    try:
        done_idx = tokens.index("DONE")
    except ValueError:
        return None
    actions = tokens[:done_idx]
    for a in actions:
        if a not in valid_actions:
            return None
    return actions

class MazeEnv:
    """
    Maze environment for simulating action execution and checking success.
    
    Implementation:
    - Maintains grid, start position, goal position, and current position
    - step() executes a single action
    - check_success() simulates a sequence of actions
    """
    
    # Action mapping: action name -> (row_delta, col_delta)
    ACTION_MAP = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }
    
    def __init__(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ):
        """
        Args:
            grid: Maze grid, 1 = wall, 0 = path
            start: Start position (row, col)
            goal: Goal position (row, col)
        """
        self.grid = grid
        self.size = grid.shape[0]
        self.start = start
        self.goal = goal
        self.current_pos = start
    
    def reset(self) -> Tuple[int, int]:
        """
        Reset environment to initial state.
        
        Returns:
            Start position
        """
        self.current_pos = self.start
        return self.current_pos
    
    def step(self, action: str) -> Tuple[Tuple[int, int], bool, bool]:
        """
        Execute a single action.
        
        Args:
            action: Action name (UP/DOWN/LEFT/RIGHT)
        
        Returns:
            (new_pos, done, success)
            - new_pos: New position
            - done: Whether episode ended (reached goal or hit wall)
            - success: Whether successfully reached goal
        """
        if action not in self.ACTION_MAP:
            # Invalid action, stay in place
            return self.current_pos, True, False
        
        dr, dc = self.ACTION_MAP[action]
        new_row = self.current_pos[0] + dr
        new_col = self.current_pos[1] + dc
        
        # Check boundary
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            # Out of bounds, fail
            return self.current_pos, True, False
        
        # Check wall collision
        if self.grid[new_row, new_col] == 1:
            # Hit wall, fail
            return self.current_pos, True, False
        
        # Move to new position
        self.current_pos = (new_row, new_col)
        
        # Check if reached goal
        if self.current_pos == self.goal:
            return self.current_pos, True, True
        
        return self.current_pos, False, False
    
    def check_success(self, actions: List[str]) -> bool:
        """
        Simulate a sequence of actions and check if goal is reached.
        
        Args:
            actions: List of actions
        
        Returns:
            Whether successfully reached goal
        """
        self.reset()
        
        for action in actions:
            _, done, success = self.step(action)
            if done:
                return success
        
        # Check if at goal position after all actions
        return self.current_pos == self.goal
    
    @classmethod
    def from_sequence(cls, sequence: str) -> Optional["MazeEnv"]:
        """
        Parse and create MazeEnv from training data sequence.
        
        Implementation:
        1. Parse grid tokens between GRID_START and GRID_END
        2. Reconstruct grid from WALL/PATH/START/GOAL/NEWLINE tokens
        
        Args:
            sequence: Training data sequence string
        
        Returns:
            MazeEnv instance, or None if parsing fails
        """
        tokens = sequence.split()
        
        # Find GRID_START and GRID_END positions
        try:
            grid_start_idx = tokens.index("GRID_START")
            grid_end_idx = tokens.index("GRID_END")
        except ValueError:
            return None
        
        # Extract grid tokens
        grid_tokens = tokens[grid_start_idx + 1:grid_end_idx]
        
        # Parse grid
        rows = []
        current_row = []
        start = None
        goal = None
        
        for i, token in enumerate(grid_tokens):
            if token == "NEWLINE":
                if current_row:
                    rows.append(current_row)
                    current_row = []
            elif token in ("WALL", "PATH", "START", "GOAL"):
                row_idx = len(rows)
                col_idx = len(current_row)
                
                if token == "WALL":
                    current_row.append(1)
                else:
                    current_row.append(0)
                    if token == "START":
                        start = (row_idx, col_idx)
                    elif token == "GOAL":
                        goal = (row_idx, col_idx)
        
        # Add last row (if not ended with NEWLINE)
        if current_row:
            rows.append(current_row)
        
        if not rows or start is None or goal is None:
            return None
        
        # Convert to numpy array
        grid = np.array(rows, dtype=int)
        
        return cls(grid=grid, start=start, goal=goal)
    
    @classmethod
    def from_token_ids(
        cls,
        token_ids: List[int],
        vocab: Dict[str, int],
    ) -> Optional["MazeEnv"]:
        """
        Parse and create MazeEnv from token id sequence.
        
        Args:
            token_ids: Token id sequence
            vocab: Vocabulary dictionary
        
        Returns:
            MazeEnv instance, or None if parsing fails
        """
        # Reverse vocab dictionary
        id_to_token = {v: k for k, v in vocab.items()}
        
        # Convert to token strings
        tokens = [id_to_token.get(tid, "<unk>") for tid in token_ids]
        sequence = " ".join(tokens)
        
        return cls.from_sequence(sequence)
    
    def render_ascii(self) -> str:
        """
        Render maze as ASCII string.
        """
        chars = {1: '#', 0: ' '}
        lines = []
        lines.append("-" * (self.size + 2))
        
        for r in range(self.size):
            row_chars = []
            for c in range(self.size):
                if (r, c) == self.current_pos:
                    row_chars.append('A')  # Agent
                elif (r, c) == self.start:
                    row_chars.append('S')
                elif (r, c) == self.goal:
                    row_chars.append('G')
                else:
                    row_chars.append(chars[self.grid[r, c]])
            lines.append("|" + "".join(row_chars) + "|")
        
        lines.append("-" * (self.size + 2))
        return "\n".join(lines)


def _action_distribution(actions):
    """Return per-action counts UP/DOWN/LEFT/RIGHT for the parsed action list."""
    counts = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}
    if actions is None:
        return counts
    for a in actions:
        if a in counts:
            counts[a] += 1
    return counts


def _has_done_token(solution_str):
    return "DONE" in solution_str.split()


def _format_outcome(solution_str, env, actions):
    """Compute trajectory-level outcome flags shared across reward fns.

    Fields that are not applicable to this rollout (path_length when no DONE
    was emitted, optimal_length when the maze couldn't be parsed, etc.) are
    returned as None so that ``process_validation_metrics`` skips them via
    its built-in None handling rather than averaging in -1 sentinels.
    """
    has_done = _has_done_token(solution_str)
    valid_format = 1.0 if actions is not None else 0.0

    if env is None or actions is None:
        return {
            "goal_reached": 0.0,
            "wall_collision": 0.0,
            "valid_format": valid_format,
            "done_token_generated": 1.0 if has_done else 0.0,
            "valid_path_no_goal": 0.0,
            "is_shortest": 0.0,
            "path_length": None,
            "optimal_length": None,
            "final_pos": None,
            "reached_goal": False,
            "hit_wall": False,
            "L": None,
            "L_star": None,
        }

    reached, hit_wall, L, final_pos = _simulate_actions(env, actions)
    L_star = _bfs_optimal_length(env.grid, env.start, env.goal)
    valid_path_no_goal = 1.0 if (not reached and not hit_wall) else 0.0
    # Reward-agnostic shortest-path indicator: 1 iff agent reached goal AND
    # took a path of optimal length L*.  Computed identically here for every
    # reward function so dashboards can plot eval/<ds>/pct/reached_shortest
    # side-by-side across rewards / estimators.
    is_shortest = 1.0 if (reached and L is not None and L_star is not None and L <= L_star) else 0.0

    return {
        "goal_reached": 1.0 if reached else 0.0,
        "wall_collision": 1.0 if hit_wall else 0.0,
        "valid_format": 1.0,
        # actions is non-None on this branch -> DONE was found by the parser.
        "done_token_generated": 1.0,
        "valid_path_no_goal": valid_path_no_goal,
        "is_shortest": is_shortest,
        "path_length": L,
        "optimal_length": L_star,
        "final_pos": final_pos,
        "reached_goal": reached,
        "hit_wall": hit_wall,
        "L": L,
        "L_star": L_star,
    }


def _action_counts_dict(actions):
    counts = _action_distribution(actions)
    return {
        "action_count_UP": counts["UP"],
        "action_count_DOWN": counts["DOWN"],
        "action_count_LEFT": counts["LEFT"],
        "action_count_RIGHT": counts["RIGHT"],
        "action_count_total": sum(counts.values()),
    }


def _public_outcome(out):
    """Strip private fields (final_pos / etc) from the outcome dict."""
    return {
        "goal_reached": out["goal_reached"],
        "wall_collision": out["wall_collision"],
        "valid_format": out["valid_format"],
        "done_token_generated": out["done_token_generated"],
        "valid_path_no_goal": out["valid_path_no_goal"],
        "is_shortest": out["is_shortest"],
        "path_length": out["path_length"],
        "optimal_length": out["optimal_length"],
    }


def judge_maze(solution_str: str, ground_truth: str) -> dict:
    """Binary reached-goal reward (legacy maze_17 reward).

    Returns a dict with ``score`` plus trajectory components for logging. The
    training pipeline pulls ``score`` via PrimeRewardManager._compute_single_score.
    """
    actions = _parse_actions_from_solution(solution_str)
    env = MazeEnv.from_sequence(ground_truth)
    out = _format_outcome(solution_str, env, actions)

    score = 1.0 if out["reached_goal"] else 0.0
    res = {"score": score, **_public_outcome(out), **_action_counts_dict(actions)}
    return res


def judge_maze_binary_shortest(solution_str, ground_truth):
    """R = 1.0 iff agent reaches goal via shortest path, 0.0 otherwise."""
    actions = _parse_actions_from_solution(solution_str)
    env = MazeEnv.from_sequence(ground_truth)
    out = _format_outcome(solution_str, env, actions)

    L, L_star = out["L"], out["L_star"]
    # is_shortest is now computed uniformly in _format_outcome -> _public_outcome.
    # Score for the binary-shortest reward is exactly that field.
    score = out["is_shortest"]
    length_ratio = (L / L_star) if (out["reached_goal"] and L is not None and L_star) else None

    res = {
        "score": score,
        **_public_outcome(out),
        "length_ratio": length_ratio,
        **_action_counts_dict(actions),
    }
    return res


def judge_maze_continuous(solution_str, ground_truth, UB=40):
    """R = max(0, (UB - L) / (UB - L*)) on goal reach, capped at 1.0; else 0."""
    actions = _parse_actions_from_solution(solution_str)
    env = MazeEnv.from_sequence(ground_truth)
    out = _format_outcome(solution_str, env, actions)

    L, L_star = out["L"], out["L_star"]
    score = 0.0
    length_excess = None
    exceeded_UB = 0.0
    length_ratio = None

    if out["reached_goal"] and L is not None and L_star is not None and L_star > 0:
        length_ratio = L / L_star
        length_excess = L - L_star
        if UB <= L_star:
            score = 1.0 if L <= L_star else 0.0
        else:
            score = max(0.0, (UB - L) / (UB - L_star))
            score = min(score, 1.0)
        if L > UB:
            exceeded_UB = 1.0

    res = {
        "score": score,
        **_public_outcome(out),
        "length_ratio": length_ratio,
        "length_excess": length_excess,
        "exceeded_UB": exceeded_UB,
        **_action_counts_dict(actions),
    }
    return res


def judge_maze_composite(solution_str, ground_truth, UB=40, D=20):
    """Composite proximity + solution-quality reward; wall collision zeros it."""
    actions = _parse_actions_from_solution(solution_str)
    env = MazeEnv.from_sequence(ground_truth)
    out = _format_outcome(solution_str, env, actions)

    R_dist = 0.0
    R_sol = 0.0
    bfs_dist = None
    length_ratio = None

    if env is not None and actions is not None and not out["hit_wall"]:
        if out["reached_goal"]:
            R_dist = 0.5
        else:
            d = _bfs_distance(env.grid, out["final_pos"], env.goal)
            if d is not None and 0 < d <= D:
                R_dist = 0.5 * (1.0 - d / D)
            if d is not None:
                bfs_dist = d

        if out["reached_goal"] and out["L"] is not None and out["L_star"] is not None and out["L_star"] > 0:
            L, L_star = out["L"], out["L_star"]
            length_ratio = L / L_star
            if L <= L_star:
                R_sol = 0.5
            elif L <= UB and UB > L_star:
                R_sol = 0.5 * (UB - L) / (UB - L_star)

    score = R_dist + R_sol
    is_proximity_only = 1.0 if (R_dist > 0.0 and R_sol == 0.0) else 0.0
    is_full_success = 1.0 if (R_dist == 0.5 and R_sol == 0.5) else 0.0
    is_partial_success = 1.0 if (0.0 < score < 1.0) else 0.0

    res = {
        "score": score,
        **_public_outcome(out),
        "R_dist": R_dist,
        "R_sol": R_sol,
        "length_ratio": length_ratio,
        "bfs_distance_to_goal": bfs_dist,
        "is_proximity_only": is_proximity_only,
        "is_full_success": is_full_success,
        "is_partial_success": is_partial_success,
        **_action_counts_dict(actions),
    }
    return res


# ---------------------------------------------------------------------------
# Aggregate metrics — called per-batch from the trainer.
# ---------------------------------------------------------------------------

_REWARD_FIELD_KEYS = {"score", "reward"}
_SENTINEL_FIELDS = {
    "path_length",
    "optimal_length",
    "length_ratio",
    "length_excess",
    "bfs_distance_to_goal",
    "progress_fraction",
    "d_start",
}


def _safe_arr(values):
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    return arr


def compute_maze_aggregate_metrics(reward_extra_infos: dict, data_source: str = "maze") -> dict:
    """Compute aggregate maze metrics from per-rollout reward_extra_info dicts.

    Sentinel-valued fields (-1) are filtered before averaging. Returns a dict
    of W&B-friendly metric name -> scalar.
    """
    metrics: Dict[str, float] = {}
    if not reward_extra_infos:
        return metrics

    n = max(
        (len(v) for v in reward_extra_infos.values() if isinstance(v, (list, tuple))),
        default=0,
    )
    if n == 0:
        return metrics

    def _vals(key):
        return reward_extra_infos.get(key, []) or []

    # Trajectory outcomes (universal).
    for key in ["goal_reached", "wall_collision", "valid_format", "done_token_generated", "valid_path_no_goal"]:
        vals = _vals(key)
        if vals:
            metrics[f"trajectory/{key}"] = float(sum(vals) / max(1, len(vals)))

    # Path quality.
    lengths = [v for v in _vals("path_length") if v is not None and v > 0]
    opt_lengths = [v for v in _vals("optimal_length") if v is not None and v > 0]
    ratios = [v for v in _vals("length_ratio") if v is not None and v > 0]

    if lengths:
        arr = np.array(lengths, dtype=np.float64)
        metrics["path/length/mean"] = float(arr.mean())
        metrics["path/length/std"] = float(arr.std())
        metrics["path/length/min"] = float(arr.min())
        metrics["path/length/max"] = float(arr.max())
    if opt_lengths:
        metrics["path/optimal_length/mean"] = float(np.mean(opt_lengths))
    if ratios:
        rarr = np.array(ratios, dtype=np.float64)
        metrics["path/length_ratio/mean"] = float(rarr.mean())
        metrics["path/length_ratio/std"] = float(rarr.std())

    is_shortest_vals = _vals("is_shortest")
    goal_vals = _vals("goal_reached")
    if is_shortest_vals and goal_vals:
        goal_reaching_shortest = [
            s for s, g in zip(is_shortest_vals, goal_vals) if g > 0
        ]
        if goal_reaching_shortest:
            metrics["path/is_shortest/fraction"] = float(
                sum(goal_reaching_shortest) / len(goal_reaching_shortest)
            )

    bfs_dists_unreached = [
        v for v in _vals("bfs_distance_to_goal") if v is not None and v >= 0
    ]
    if bfs_dists_unreached:
        darr = np.array(bfs_dists_unreached, dtype=np.float64)
        metrics["path/bfs_distance_to_goal/mean"] = float(darr.mean())
        metrics["path/bfs_distance_to_goal/median"] = float(np.median(darr))

    # Reward distribution.
    score_key = "score" if _vals("score") else ("reward" if _vals("reward") else None)
    if score_key is not None:
        s = np.array(_vals(score_key), dtype=np.float64)
        if len(s) > 0:
            metrics["reward/mean"] = float(s.mean())
            metrics["reward/std"] = float(s.std())
            metrics["reward/median"] = float(np.median(s))
            metrics["reward/min"] = float(s.min())
            metrics["reward/max"] = float(s.max())
            metrics["reward/fraction_zero"] = float((s == 0).mean())
            metrics["reward/fraction_one"] = float((s == 1.0).mean())
            metrics["reward/fraction_positive"] = float((s > 0).mean())
            nonzero = s[s > 0]
            if len(nonzero) > 0:
                metrics["reward/nonzero_mean"] = float(nonzero.mean())
                metrics["reward/nonzero_std"] = float(nonzero.std())

    # Continuous-specific.
    if data_source == "maze_17_continuous":
        excess = [v for v in _vals("length_excess") if v is not None and v >= 0]
        if excess:
            earr = np.array(excess, dtype=np.float64)
            metrics["reward_continuous/L_minus_Lstar/mean"] = float(earr.mean())
            metrics["reward_continuous/L_minus_Lstar/max"] = float(earr.max())
        exceeded_vals = _vals("exceeded_UB")
        if exceeded_vals and goal_vals:
            goal_reaching_exceeded = [
                e for e, g in zip(exceeded_vals, goal_vals) if g > 0
            ]
            if goal_reaching_exceeded:
                metrics["reward_continuous/exceeded_UB"] = float(
                    sum(goal_reaching_exceeded) / len(goal_reaching_exceeded)
                )

    # Composite-specific. Maze v2 custom composite rewards reuse the
    # maze_17_continuous parquet, so detect them by returned component fields.
    if data_source == "maze_17_composite" or _vals("R_dist") or _vals("R_sol"):
        rd_vals = _vals("R_dist")
        rs_vals = _vals("R_sol")
        if rd_vals:
            rd = np.array(rd_vals, dtype=np.float64)
            metrics["reward_composite/R_dist/mean"] = float(rd.mean())
            metrics["reward_composite/R_dist/std"] = float(rd.std())
            metrics["reward_composite/R_dist/fraction_zero"] = float((rd == 0).mean())
            metrics["reward_composite/R_dist/fraction_half"] = float((rd == 0.5).mean())
        if rs_vals:
            rs = np.array(rs_vals, dtype=np.float64)
            metrics["reward_composite/R_sol/mean"] = float(rs.mean())
            metrics["reward_composite/R_sol/std"] = float(rs.std())
            metrics["reward_composite/R_sol/fraction_zero"] = float((rs == 0).mean())
            metrics["reward_composite/R_sol/fraction_half"] = float((rs == 0.5).mean())
        prox_vals = _vals("is_proximity_only")
        if prox_vals:
            metrics["reward_composite/proximity_only"] = float(
                sum(prox_vals) / len(prox_vals)
            )
        full_vals = _vals("is_full_success")
        if full_vals:
            metrics["reward_composite/full_success"] = float(
                sum(full_vals) / len(full_vals)
            )
        partial_vals = _vals("is_partial_success")
        if partial_vals:
            metrics["reward_composite/partial_success"] = float(
                sum(partial_vals) / len(partial_vals)
            )
        prox_dists = [
            d for d, p in zip(
                _vals("bfs_distance_to_goal"),
                prox_vals,
            )
            if p > 0 and d is not None and d >= 0
        ]
        if prox_dists:
            metrics["reward_composite/bfs_dist_when_proximity_only/mean"] = float(
                np.mean(prox_dists)
            )

    # Action distribution.
    total = sum(int(v) for v in _vals("action_count_total") if v is not None)
    if total > 0:
        for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            cnt = sum(int(v) for v in _vals(f"action_count_{action}") if v is not None)
            metrics[f"policy/action_distribution/{action}"] = float(cnt) / float(total)

    return metrics


def compute_advantage_diagnostics(advantages, response_mask, index=None) -> dict:
    """Compute advantage estimator diagnostics from a single training batch.

    Args:
        advantages: tensor of shape (bs, response_length).
        response_mask: mask tensor of shape (bs, response_length).
        index: optional np.ndarray of group ids per sample.

    Returns dict of metric_name -> float.
    """
    import torch

    metrics: Dict[str, float] = {}
    valid = advantages[response_mask.bool()]
    if valid.numel() == 0:
        return metrics

    valid_f = valid.float()
    metrics["advantage/mean"] = float(valid_f.mean().item())
    metrics["advantage/std"] = float(valid_f.std().item())
    metrics["advantage/max"] = float(valid_f.max().item())
    metrics["advantage/min"] = float(valid_f.min().item())
    metrics["advantage/abs_mean"] = float(valid_f.abs().mean().item())
    metrics["advantage/fraction_zero"] = float((valid_f == 0).float().mean().item())
    metrics["advantage/fraction_positive"] = float((valid_f > 0).float().mean().item())

    pos = valid_f[valid_f > 0]
    neg = valid_f[valid_f < 0]
    if pos.numel() > 0 and neg.numel() > 0:
        ratio = pos.mean().item() / neg.mean().abs().item()
        metrics["advantage/pos_neg_ratio"] = float(ratio)

    if index is not None:
        unique_ids = np.unique(index)
        per_sample_adv = advantages[:, 0].float()  # outcome-level reward broadcasts
        group_stds = []
        group_ranges = []
        zero_groups = 0
        for uid in unique_ids:
            mask = (index == uid)
            group_adv = per_sample_adv[mask]
            if group_adv.numel() == 0:
                continue
            if torch.allclose(group_adv, torch.zeros_like(group_adv)):
                zero_groups += 1
            group_stds.append(float(group_adv.std().item()) if group_adv.numel() > 1 else 0.0)
            group_ranges.append(
                float((group_adv.max() - group_adv.min()).item())
                if group_adv.numel() > 1
                else 0.0
            )
        if group_stds:
            metrics["advantage_per_group/within_group_std/mean"] = float(np.mean(group_stds))
        if group_ranges:
            metrics["advantage_per_group/within_group_range/mean"] = float(np.mean(group_ranges))
        metrics["advantage_per_group/num_zero_groups"] = float(zero_groups)
        if len(unique_ids) > 0:
            metrics["advantage_per_group/num_zero_groups_fraction"] = float(zero_groups) / float(len(unique_ids))

    return metrics
