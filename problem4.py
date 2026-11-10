import heapq
import pandas as pd
from dataclasses import dataclass

grid = [
    [1, 1, 1, "X", 2, 1, 1],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, 1, 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, 1, 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    [1, 1, 1, "X", 1, 1, 1]
]

def astar(start, goal, map):
    rows,cols = len(map), len(map[0])
    def h(pos):
        return (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))#与goal相邻的格子的代价为0

    open_set = [(h(start), start)]
    g_score = {start: 0}
    parent = {}

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_set:
        f, current = heapq.heappop(open_set)

        # 跳过过期的队列项
        if f != g_score[current] + h(current):
            continue

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            return g_score[goal], path[::-1]

        r, c = current

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if map[nr][nc] == "X":
                continue

            neighbor = (nr, nc)
            new_g = g_score[current] + map[nr][nc]

            if new_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_g
                parent[neighbor] = current
                new_f = new_g + h(neighbor)
                heapq.heappush(open_set, (new_f, neighbor))

    return float('inf'), []

points = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0)
}

names = list(points.keys())

distance_matrix = []

for start_name in names:
    row = []
    for goal_name in names:
        start_pos = points[start_name]
        goal_pos = points[goal_name]
        cost, _ = astar(start_pos, goal_pos, grid)
        row.append(cost if cost is not None else float('inf'))

    distance_matrix.append(row)

df = pd.DataFrame(
    distance_matrix,
    index=names,
    columns=names
)

print(df)

treasure_values = {
    "T1": 9,
    "T2": 6,
    "T3": 7,
    "T4": 7
}

# 定义博弈状态
@dataclass(frozen=True)
class GameState:
    pos_A: str                  # A 当前所在位置
    pos_B: str                  # B 当前所在位置
    remaining: frozenset        # 尚未获取的宝藏
    score_A: int                # A 的累计收益
    score_B: int                # B 的累计收益
    current_player: str         # 当前行动玩家
    depth: int                  # 已经行动的回合数


# 根据当前宝藏价值表创建初始状态。价值扫描时必须重新创建状态，
# 否则 remaining 可能与实验中的 treasure_values 不一致。
def make_initial_state():
    return GameState(
        pos_A="A",
        pos_B="B",
        remaining=frozenset(treasure_values.keys()),
        score_A=0,
        score_B=0,
        current_player="A",
        depth=0
    )


initial_state = make_initial_state()

print(initial_state)

def get_next_states(state, move_order=None):
    """
    根据当前状态生成后继状态
    move_order：宝藏访问顺序
    """
    next_states = []

    if not state.remaining:
        return next_states

    if move_order is None:
        targets = sorted(state.remaining)
    else:
        targets = [t for t in move_order if t in state.remaining]
        targets += sorted(state.remaining - set(targets))

    for target in targets:

        if state.current_player == "A":
            gain = treasure_values[target] - df.loc[state.pos_A, target]

            new_state = GameState(
                pos_A=target,
                pos_B=state.pos_B,
                remaining=state.remaining - {target},
                score_A=state.score_A + gain,
                score_B=state.score_B,
                current_player="B",
                depth=state.depth + 1
            )

        else:
            gain = treasure_values[target] - df.loc[state.pos_B, target]

            new_state = GameState(
                pos_A=state.pos_A,
                pos_B=target,
                remaining=state.remaining - {target},
                score_A=state.score_A,
                score_B=state.score_B + gain,
                current_player="A",
                depth=state.depth + 1
            )

        next_states.append(new_state)

    return next_states

def game_pruning_count(
    state, H, alpha, beta, counter, move_order=None
):
    counter["nodes"] += 1

    if state.depth >= H or not state.remaining:
        return state.score_A - state.score_B

    next_states = get_next_states(state, move_order)

    if state.current_player == "A":
        value = float('-inf')

        for i, next_state in enumerate(next_states):
            child_value = game_pruning_count(
                next_state, H, alpha, beta,
                counter, move_order
            )

            value = max(value, child_value)
            alpha = max(alpha, value)

            if alpha >= beta:
                counter["cutoff_events"] += 1
                counter["pruned_branches"] += len(next_states) - i - 1
                break

        return value

    else:
        value = float('inf')

        for i, next_state in enumerate(next_states):
            child_value = game_pruning_count(
                next_state, H, alpha, beta,
                counter, move_order
            )

            value = min(value, child_value)
            beta = min(beta, value)

            if alpha >= beta:
                counter["cutoff_events"] += 1
                counter["pruned_branches"] += len(next_states) - i - 1
                break

        return value


def get_best_move_pruning_count(state, H, move_order=None):
    counter = {
        "nodes": 1,
        "cutoff_events": 0,
        "pruned_branches": 0
    }

    if state.depth >= H or not state.remaining:
        return None, state.score_A - state.score_B, counter

    next_states = get_next_states(state, move_order)

    alpha = float('-inf')
    beta = float('inf')

    if state.current_player == "A":
        best_value = float('-inf')
        best_target = None

        for next_state in next_states:
            value = game_pruning_count(
                next_state, H, alpha, beta,
                counter, move_order
            )

            if value > best_value:
                best_value = value
                best_target = next_state.pos_A

            alpha = max(alpha, best_value)

    else:
        best_value = float('inf')
        best_target = None

        for next_state in next_states:
            value = game_pruning_count(
                next_state, H, alpha, beta,
                counter, move_order
            )

            if value < best_value:
                best_value = value
                best_target = next_state.pos_B

            beta = min(beta, best_value)

    return best_target, best_value, counter

def game_minimax_count(state, H, counter, move_order=None):
    counter["nodes"] += 1

    if state.depth >= H or not state.remaining:
        return state.score_A - state.score_B

    next_states = get_next_states(state, move_order)

    if state.current_player == "A":
        value = float('-inf')

        for next_state in next_states:
            value = max(
                value,
                game_minimax_count(
                    next_state, H, counter, move_order
                )
            )

        return value

    else:
        value = float('inf')

        for next_state in next_states:
            value = min(
                value,
                game_minimax_count(
                    next_state, H, counter, move_order
                )
            )

        return value


def get_best_move_minimax_count(state, H, move_order=None):
    counter = {"nodes": 1}

    if state.depth >= H or not state.remaining:
        return None, state.score_A - state.score_B, counter

    next_states = get_next_states(state, move_order)

    best_target = None

    if state.current_player == "A":
        best_value = float('-inf')

        for next_state in next_states:
            value = game_minimax_count(
                next_state, H, counter, move_order
            )

            if value > best_value:
                best_value = value
                best_target = next_state.pos_A

    else:
        best_value = float('inf')

        for next_state in next_states:
            value = game_minimax_count(
                next_state, H, counter, move_order
            )

            if value < best_value:
                best_value = value
                best_target = next_state.pos_B

    return best_target, best_value, counter
def game_pruning(state, H, alpha=float('-inf'), beta=float('inf')):
    """
    Alpha-Beta 剪枝
    state：当前博弈状态
    H：最大搜索深度
    alpha：A 当前已知的最大下界
    beta：B 当前已知的最小上界
    """

    # 到达搜索深度，或者所有宝藏都已被获取
    if state.depth >= H or not state.remaining:
        return state.score_A - state.score_B

    # 生成所有可能的后继状态
    next_states = get_next_states(state)

    # A 是最大化玩家
    if state.current_player == "A":
        value = float('-inf')

        for next_state in next_states:
            child_value = game_pruning(
                next_state, H, alpha, beta
            )

            value = max(value, child_value)
            alpha = max(alpha, value)

            # Alpha-Beta 剪枝条件
            if alpha >= beta:
                break

        return value

    # B 是最小化玩家
    else:
        value = float('inf')

        for next_state in next_states:
            child_value = game_pruning(
                next_state, H, alpha, beta
            )

            value = min(value, child_value)
            beta = min(beta, value)

            # Alpha-Beta 剪枝条件
            if alpha >= beta:
                break

        return value


def get_best_move_pruning(state, H):
    """
    使用 Alpha-Beta 剪枝获取当前玩家的最优选择
    返回：最优目标、最优收益差
    """

    if state.depth >= H or not state.remaining:
        return None, state.score_A - state.score_B

    next_states = get_next_states(state)

    alpha = float('-inf')
    beta = float('inf')

    if state.current_player == "A":
        best_value = float('-inf')
        best_target = None

        for next_state in next_states:
            value = game_pruning(
                next_state, H, alpha, beta
            )

            if value > best_value:
                best_value = value
                best_target = next_state.pos_A

            alpha = max(alpha, best_value)

    else:
        best_value = float('inf')
        best_target = None

        for next_state in next_states:
            value = game_pruning(
                next_state, H, alpha, beta
            )

            if value < best_value:
                best_value = value
                best_target = next_state.pos_B

            beta = min(beta, best_value)

    return best_target, best_value

def get_move_orders():
    """根据当前价值表生成访问顺序，避免价值变化后仍使用旧顺序。"""
    value_descending = sorted(
        treasure_values,
        key=lambda target: (-treasure_values[target], target)
    )
    return {
        "正序": ["T1", "T2", "T3", "T4"],
        "逆序": ["T4", "T3", "T2", "T1"],
        "按宝藏价值降序": value_descending
    }


def evaluate_root_actions(state, H, move_order=None):
    """返回根节点每个候选目标的 Minimax 值，用于识别并列最优。"""
    action_values = {}

    for next_state in get_next_states(state, move_order):
        target = (
            next_state.pos_A
            if state.current_player == "A"
            else next_state.pos_B
        )
        counter = {"nodes": 0}
        action_values[target] = int(
            game_minimax_count(next_state, H, counter, move_order)
        )

    return action_values


def run_baseline_experiment():
    """复现实验题目四给定数据，并比较不同节点访问顺序。"""
    state = make_initial_state()

    for H in [2, 4]:
        print(f"\n========== 基础实验 H={H} ==========")

        for order_name, order in get_move_orders().items():
            target1, value1, count1 = get_best_move_minimax_count(
                state, H, order
            )
            target2, value2, count2 = get_best_move_pruning_count(
                state, H, order
            )

            print(f"\n访问顺序：{order_name} {order}")
            print(
                f"Minimax：首选={target1}，收益差={value1}，"
                f"访问节点={count1['nodes']}"
            )
            print(
                f"Alpha-Beta：首选={target2}，收益差={value2}，"
                f"访问节点={count2['nodes']}，"
                f"剪枝事件={count2['cutoff_events']}，"
                f"跳过分支={count2['pruned_branches']}"
            )


def run_treasure_value_experiment(
    changed_target="T1",
    test_values=range(9, 17),
    depths=(2, 4)
):
    """
    单变量对照实验：只改变一个宝藏的价值，观察 A 的首选是否改变。

    搜索始终采用固定正序，防止“价值变化”和“访问顺序变化”两个变量
    同时改变。optimal_targets 会保留所有并列最优目标，first_target
    则是当前固定访问顺序下实际返回的第一个目标。
    """
    global treasure_values

    if changed_target not in treasure_values:
        raise ValueError(f"未知宝藏：{changed_target}")

    original_values = treasure_values.copy()
    fixed_order = ["T1", "T2", "T3", "T4"]
    rows = []

    try:
        for changed_value in test_values:
            treasure_values = original_values.copy()
            treasure_values[changed_target] = changed_value
            state = make_initial_state()

            for H in depths:
                action_values = evaluate_root_actions(
                    state, H, fixed_order
                )
                best_value = max(action_values.values())
                optimal_targets = [
                    target
                    for target, value in action_values.items()
                    if value == best_value
                ]

                minimax_target, minimax_value, minimax_count = (
                    get_best_move_minimax_count(state, H, fixed_order)
                )
                ab_target, ab_value, ab_count = (
                    get_best_move_pruning_count(state, H, fixed_order)
                )

                rows.append({
                    "changed_target": changed_target,
                    "changed_value": changed_value,
                    "H": H,
                    "optimal_targets": "/".join(optimal_targets),
                    "first_target": minimax_target,
                    "utility": int(minimax_value),
                    "root_action_values": str(action_values),
                    "minimax_nodes": minimax_count["nodes"],
                    "alpha_beta_target": ab_target,
                    "alpha_beta_utility": int(ab_value),
                    "alpha_beta_nodes": ab_count["nodes"],
                    "cutoff_events": ab_count["cutoff_events"],
                    "pruned_branches": ab_count["pruned_branches"]
                })
    finally:
        treasure_values = original_values

    result = pd.DataFrame(rows)
    print("\n========== 自设对照实验：宝藏价值与 A 的首选 ==========")
    print(f"控制变量：地图、位置、其他宝藏价值、固定访问顺序 {fixed_order}")
    print(f"自变量：{changed_target} 的价值")
    print(result.to_string(index=False))

    return result


if __name__ == "__main__":
    run_baseline_experiment()

    # T1 原始价值最高但离 A 较远，适合检验“价值提高后是否必然优先”。
    value_experiment_result = run_treasure_value_experiment(
        changed_target="T1",
        test_values=range(9, 17),
        depths=(2, 4)
    )
