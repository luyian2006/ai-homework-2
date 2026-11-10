import heapq

grid = [
    [0, 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, 0]
]

def astar(start, goal, grid, return_path=False):
    rows, cols = len(grid), len(grid[0])

    def h(pos):
        distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        return max(0, distance - 1)

    open_set = [(h(start), start)]
    g_score = {start: 0}
    parent = {}

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_set:
        f, current = heapq.heappop(open_set)

        if f != g_score[current] + h(current):
            continue

        if current == goal:
            path = []
            temp = current

            while temp in parent:
                path.append(temp)
                temp = parent[temp]

            path.append(start)
            path.reverse()

            if return_path:
                return g_score[current], path
            return g_score[current]

        r, c = current

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            neighbor = (nr, nc)
            new_g = g_score[current] + grid[nr][nc]

            if new_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_g
                parent[neighbor] = current

                new_f = new_g + h(neighbor)
                heapq.heappush(open_set, (new_f, neighbor))

    if return_path:
        return -1, []
    return -1


cost, path = astar((0, 0), (4, 4), grid, return_path=True)

print("最小总代价：", cost)
print("最短代价路径：", path)

# 展示路径代价的累计过程
total = 0
for r, c in path[1:]:
    total += grid[r][c]
    print((r, c), "进入代价：", grid[r][c], "累计代价：", total)