import heapq
import math
grid=[
  [0, 1, 1, 8, 1],
  [1, 8, 3, 1, 1],
  [1, 8, 1, 8, 1],
  [1, 1, 1, 8, 1],
  [8, 8, 1, 1, 0]
]
def astar(start, goal, grid, use_heuristic=True):
    rows, cols = len(grid), len(grid[0])

    def h(pos):
        if not use_heuristic:
            return 0
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_set = [(h(start), start)]
    g_score = {start: 0}
    parent = {}
    visited_nodes = 0

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_set:
        f, current = heapq.heappop(open_set)

        if f != g_score[current] + h(current):
            continue

        visited_nodes += 1

        if current == goal:
            path = []
            temp = current

            while temp in parent:
                path.append(temp)
                temp = parent[temp]

            path.append(start)
            return path[::-1], visited_nodes

        r, c = current

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if grid[nr][nc] == 'X':
                continue

            neighbor = (nr, nc)
            new_g = g_score[current] + 1

            if new_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_g
                parent[neighbor] = current

                new_f = new_g + h(neighbor)
                heapq.heappush(open_set, (new_f, neighbor))

    return [], visited_nodes

# 不使用启发函数
path1, nodes1 = astar((0, 0), (3, 4), grid, False)

# 使用曼哈顿距离
path2, nodes2 = astar((0, 0), (3, 4), grid, True)

print("h(n)=0")
print("最短路径：", path1)
print("路径长度：", len(path1) - 1)
print("搜索节点数：", nodes1)

print("\n曼哈顿距离")
print("最短路径：", path2)
print("路径长度：", len(path2) - 1)
print("搜索节点数：", nodes2)