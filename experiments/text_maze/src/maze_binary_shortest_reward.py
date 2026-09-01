from verl.utils.reward_score.maze import judge_maze_binary_shortest


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "maze_17_continuous":
        raise NotImplementedError(f"Unsupported data_source for binary-shortest maze reward: {data_source}")
    return judge_maze_binary_shortest(solution_str=solution_str, ground_truth=ground_truth)
