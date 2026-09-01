from verl.utils.reward_score.maze import (
    MazeEnv,
    _action_counts_dict,
    _bfs_distance,
    _format_outcome,
    _parse_actions_from_solution,
    _public_outcome,
)


def _composite_v2_components(env, out):
    """L*-normalized proximity plus L*/L solution quality."""
    R_dist = 0.0
    R_sol = 0.0
    bfs_distance_to_goal = None
    length_ratio = None
    progress_fraction = None
    d_start = None

    L_star = out["L_star"]
    if env is None or out["hit_wall"] or L_star is None or L_star <= 0:
        return R_dist, R_sol, length_ratio, bfs_distance_to_goal, progress_fraction, d_start

    d_start = L_star

    final_pos = out.get("final_pos")
    if final_pos is not None:
        bfs_distance_to_goal = _bfs_distance(env.grid, final_pos, env.goal)
        if bfs_distance_to_goal is not None:
            progress_fraction = max(
                0.0,
                (float(L_star) - float(bfs_distance_to_goal)) / float(L_star),
            )
            R_dist = 0.5 * min(1.0, progress_fraction)

    if out["reached_goal"] and out["L"] is not None and out["L"] > 0:
        L = out["L"]
        length_ratio = L / L_star
        R_sol = 0.5 * min(1.0, float(L_star) / float(L))

    return R_dist, R_sol, length_ratio, bfs_distance_to_goal, progress_fraction, d_start


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "maze_17_continuous":
        raise NotImplementedError(f"Unsupported data_source for composite_v2 maze reward: {data_source}")

    actions = _parse_actions_from_solution(solution_str)
    env = MazeEnv.from_sequence(ground_truth)
    out = _format_outcome(solution_str, env, actions)

    score = 0.0
    R_sol = 0.0
    R_dist = 0.0
    length_ratio = None
    bfs_distance_to_goal = None
    progress_fraction = None
    d_start = None

    if actions is not None:
        (
            R_dist,
            R_sol,
            length_ratio,
            bfs_distance_to_goal,
            progress_fraction,
            d_start,
        ) = _composite_v2_components(env, out)
        score = R_dist + R_sol

    is_full_success = 1.0 if (R_dist == 0.5 and R_sol == 0.5) else 0.0
    is_partial_success = 1.0 if (0.0 < score < 1.0) else 0.0

    return {
        "score": score,
        **_public_outcome(out),
        "R_dist": R_dist,
        "R_sol": R_sol,
        "R_path": R_sol,
        "length_ratio": length_ratio,
        "bfs_distance_to_goal": bfs_distance_to_goal,
        "progress_fraction": progress_fraction,
        "d_start": d_start,
        "is_full_success": is_full_success,
        "is_partial_success": is_partial_success,
        **_action_counts_dict(actions),
    }
