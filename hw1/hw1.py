import random

# =========================
# 爬山演算法
# =========================
def hillClimbing(s, maxGens, maxFails):
    print("start:", s.str())

    fails = 0

    for gens in range(maxGens):

        snew = s.neighbor()

        sheight = s.height()
        nheight = snew.height()

        if nheight >= sheight:
            print(gens, ":", snew.str())
            s = snew
            fails = 0
        else:
            fails += 1

        if fails >= maxFails:
            break

    print("solution:", s.str())
    return s


# =========================
# TSP 解
# =========================
class Solution:

    def __init__(self, path, dist):
        self.path = path
        self.dist_matrix = dist

    # 計算總距離
    def total_distance(self):
        total = 0
        n = len(self.path)

        for i in range(n):
            a = self.path[i]
            b = self.path[(i + 1) % n]
            total += self.dist_matrix[a][b]

        return total

    # 高度 = -距離
    def height(self):
        return -self.total_distance()

    # 產生鄰居 (2-opt)
    def neighbor(self):

        n = len(self.path)

        new_path = self.path.copy()

        # 任選兩個位置
        i, j = sorted(random.sample(range(n), 2))

        # 避免相鄰點
        while j == i + 1:
            i, j = sorted(random.sample(range(n), 2))

        # 2-opt：反轉中間路段
        new_path[i+1:j+1] = reversed(new_path[i+1:j+1])

        return Solution(new_path, self.dist_matrix)

    # 顯示路徑
    def str(self):

        route = " -> ".join(str(x+1) for x in self.path)
        route += " -> " + str(self.path[0]+1)

        return route + "  distance=" + str(self.total_distance())


# =========================
# 主程式
# =========================

# 距離矩陣
dist = [
    [0,10,15,20,25],
    [10,0,35,25,17],
    [15,35,0,30,28],
    [20,25,30,0,16],
    [25,17,28,16,0]
]

n = len(dist)

# 初始解 1→2→3→4→5→1
initial_path = list(range(n))

s = Solution(initial_path, dist)

hillClimbing(s, maxGens=1000, maxFails=100)
