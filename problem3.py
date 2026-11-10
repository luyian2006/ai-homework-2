
import math

class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


# Minimax 算法
def minimax(node, maximizing_player, counter):
    counter["nodes"] += 1

    # 叶子节点
    if not node.children:
        return node.value

    # MAX 玩家
    if maximizing_player:
        value = float('-inf')

        for child in node.children:
            value = max(
                value,
                minimax(child, False, counter)
            )

        return value

    # MIN 玩家
    else:
        value = float('inf')

        for child in node.children:
            value = min(
                value,
                minimax(child, True, counter)
            )

        return value


# Alpha-Beta 剪枝算法
def pruning(node, alpha, beta, maximizing_player, counter):
    counter["nodes"] += 1

    # 叶子节点
    if not node.children:
        return node.value

    # MAX 玩家
    if maximizing_player:
        value = float('-inf')

        for i, child in enumerate(node.children):
            child_value = pruning(
                child, alpha, beta, False, counter
            )

            value = max(value, child_value)
            alpha = max(alpha, value)

            if alpha >= beta:
                skipped = len(node.children) - i - 1

                if skipped > 0:
                    counter["cutoffs"] += 1
                    counter["pruned_branches"] += skipped

                break

        return value

    # MIN 玩家
    else:
        value = float('inf')

        for i, child in enumerate(node.children):
            child_value = pruning(
                child, alpha, beta, True, counter
            )

            value = min(value, child_value)
            beta = min(beta, value)

            if alpha >= beta:
                skipped = len(node.children) - i - 1

                if skipped > 0:
                    counter["cutoffs"] += 1
                    counter["pruned_branches"] += skipped

                break

        return value


# 构建博弈树
def build_tree(order="original"):

    # 创建 MAX 节点
    max1 = TreeNode(children=[
        TreeNode(value=3),
        TreeNode(value=5)
    ])

    max2_left = TreeNode(children=[
        TreeNode(value=6),
        TreeNode(value=9)
    ])

    max2_right = TreeNode(children=[
        TreeNode(value=6),
        TreeNode(value=9)
    ])

    max3 = TreeNode(children=[
        TreeNode(value=1),
        TreeNode(value=2)
    ])

    # 左侧 MIN 节点
    if order == "swap_left":
        min1 = TreeNode(children=[max2_left, max1])
    else:
        min1 = TreeNode(children=[max1, max2_left])

    # 右侧 MIN 节点
    if order == "swap_right":
        min2 = TreeNode(children=[max3, max2_right])
    else:
        min2 = TreeNode(children=[max2_right, max3])

    # 根节点 MAX
    root = TreeNode(children=[min1, min2])

    return root


# 比较不同访问顺序
orders = {
    "原始左到右": "original",
    "交换左侧MIN节点顺序": "swap_left",
    "交换右侧MIN节点顺序": "swap_right"
}

for name, order in orders.items():

    root = build_tree(order)

    # Minimax
    m_counter = {"nodes": 0}
    m_value = minimax(root, True, m_counter)

    # Alpha-Beta
    p_counter = {
        "nodes": 0,
        "cutoffs": 0,
        "pruned_branches": 0
    }

    p_value = pruning(
        root,
        -math.inf,
        math.inf,
        True,
        p_counter
    )

    print(f"\n访问顺序：{name}")
    print(f"Minimax：结果={m_value}，节点数={m_counter['nodes']}")
    print(
        f"Alpha-Beta：结果={p_value}，"
        f"节点数={p_counter['nodes']}，"
        f"剪枝事件={p_counter['cutoffs']}，"
        f"跳过分支={p_counter['pruned_branches']}"
    )
  
  