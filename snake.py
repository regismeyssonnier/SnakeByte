# -*- coding: latin-1 -*-
import pygame
import random
import sys
import heapq
import optuna
import time

INF = 10**9

# ===== CONFIG =====
CELL_SIZE = 20
#W, H = 45, 30

EMPTY  = 0,
GWALL   = 1<<0
GSNAKE  = 1<<1
GENERGY = 1<<2
dist_maps = []

DX = [0, 0, -1, 1]  # mouvements horizontaux
DY = [-1, 1, 0, 0]  # mouvements verticaux

UP=0
DOWN=1
LEFT=2
RIGHT=3
BORDER=0
BORDERH=5

# directions
DIRS = [(0,-1),(0,1),(-1,0),(1,0)]  # UP DOWN LEFT RIGHT

from collections import deque

EMPTY = 0
WALL = 1<<0
SNAKE = 1<<1
ENERGY = 1<<2

MIN_H = 10
MAX_H = 24
ASPECT = 1.8

MAX_BODY = 256

import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

class PolicyNet(nn.Module):
    def __init__(self, input_size=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 directions
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class MapGen:
    def __init__(self, leagueLevel=2, seed=None):
        self.league = leagueLevel
        if seed is not None:
            random.seed(seed)

    def make(self):
        # --- skew (comme Java)
        if self.league == 1:
            skew = 2
        elif self.league == 2:
            skew = 1
        elif self.league == 3:
            skew = 0.8
        else:
            skew = 0.3

        r = random.random()

        H = MIN_H + int(round((r ** skew) * (MAX_H - MIN_H)))
        W = int(round(H * ASPECT))
        if W % 2: W += 1

        self.W, self.H = W, H
        self.grid = [[EMPTY for _ in range(W)] for _ in range(H)]

        # --- param murs
        b = 5 + random.random() * 10

        # --- sol
        for x in range(W):
            self.grid[H-1][x] = WALL

        # --- murs aléatoires
        for y in range(H-2, -1, -1):
            yNorm = (H-1-y)/(H-1)
            p = 1/(yNorm+0.1)/b

            for x in range(W//2):
                if random.random() < p:
                    self.grid[y][x] = WALL
                    self.grid[y][W-1-x] = WALL

        # --- remove petites zones
        self.remove_small_islands(10)

        # --- casser murs enfermants
        changed = True
        while changed:
            changed = False
            for y in range(H):
                for x in range(W):
                    if self.grid[y][x] == WALL:
                        continue

                    neigh_walls = []
                    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nx, ny = x+dx, y+dy
                        if 0<=nx<W and 0<=ny<H and self.grid[ny][nx]==WALL:
                            neigh_walls.append((nx,ny))

                    if len(neigh_walls) >= 3:
                        cand = [c for c in neigh_walls if c[1] <= y]
                        if cand:
                            nx,ny = random.choice(cand)
                            self.grid[ny][nx] = EMPTY
                            self.grid[ny][W-1-nx] = EMPTY
                            changed = True

        # --- energy (apples)
        energy = []
        for y in range(H):
            for x in range(W//2):
                if self.grid[y][x] == EMPTY and random.random() < 0.04:
                    ox = W-1-x
                    self.grid[y][x] = ENERGY
                    self.grid[y][ox] = ENERGY
                    energy.append((x,y))
                    energy.append((ox,y))

        # --- fallback si pas assez
        if len(energy) < 8:
            energy.clear()
            free = [(x,y) for y in range(H) for x in range(W) if self.grid[y][x]==EMPTY]
            random.shuffle(free)

            target = max(4, int(0.04*len(free)))
            while len(energy) < target*2 and free:
                x,y = free.pop()
                ox = W-1-x
                if self.grid[y][x]==EMPTY:
                    self.grid[y][x] = ENERGY
                    self.grid[y][ox] = ENERGY
                    energy.append((x,y))
                    energy.append((ox,y))

        return self.grid, energy, W, H

    # ------------------------
    # Flood fill
    # ------------------------
    def flood(self, sx, sy, vis):
        q = deque([(sx,sy)])
        comp = []
        vis.add((sx,sy))

        while q:
            x,y = q.popleft()
            comp.append((x,y))

            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx,ny = x+dx,y+dy
                if 0<=nx<self.W and 0<=ny<self.H:
                    if (nx,ny) not in vis and self.grid[ny][nx]!=WALL:
                        vis.add((nx,ny))
                        q.append((nx,ny))
        return comp

    def remove_small_islands(self, min_size):
        vis = set()
        for y in range(self.H):
            for x in range(self.W):
                if self.grid[y][x]!=WALL and (x,y) not in vis:
                    comp = self.flood(x,y,vis)
                    if len(comp) < min_size:
                        for cx,cy in comp:
                            self.grid[cy][cx] = WALL

class MapGenerator:
    def __init__(self, W, H, seed=None):
        self.W = W
        self.H = H
        self.NBENERGY = (self.W*self.H) // 10
        self.energy = []
        self.grid = [[EMPTY for _ in range(W)] for _ in range(H)]
        if seed is not None:
            random.seed(seed)

    # ------------------------
    # Génération principale
    # ------------------------
    def generate(self):
        # --- sol (ligne du bas)
        for x in range(self.W):
            self.grid[self.H - 1][x] = WALL

        # --- murs aléatoires
        for y in range(self.H - 2, -1, -1):
            y_norm = (self.H - 1 - y) / (self.H - 1)
            prob = 1 / (y_norm + 0.1) / 6.0  # ajustable

            for x in range(self.W // 2):
                if random.random() < prob:
                    self.grid[y][x] = WALL
                    self.grid[y][self.W - 1 - x] = WALL

        # --- enlever petites zones
        self.remove_small_islands(min_size=10)

        # --- ajouter energy
        self.add_energy(n=self.NBENERGY)

        return self.grid, self.energy

    # ------------------------
    # Flood fill
    # ------------------------
    def flood_fill(self, sx, sy, visited):
        q = deque([(sx, sy)])
        comp = []
        visited.add((sx, sy))

        while q:
            x, y = q.popleft()
            comp.append((x, y))

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.W and 0 <= ny < self.H and
                    (nx, ny) not in visited and
                    self.grid[ny][nx] != WALL):

                    visited.add((nx, ny))
                    q.append((nx, ny))

        return comp

    def remove_small_islands(self, min_size=30):
        visited = set()

        for y in range(self.H):
            for x in range(self.W):
                if self.grid[y][x] != WALL and (x, y) not in visited:
                    comp = self.flood_fill(x, y, visited)
                    if len(comp) < min_size:
                        for (cx, cy) in comp:
                            self.grid[cy][cx] = WALL

    # ------------------------
    # Energy symétrique
    # ------------------------
    def add_energy(self, n=6):
        placed = 0
        tries = 0

        while placed < n and tries < 500:
            tries += 1
            x = random.randint(0, self.W // 2 - 1)
            y = random.randint(0, self.H - 1)

            if self.grid[y][x] == EMPTY:
                ox = self.W - 1 - x

                self.grid[y][x] = ENERGY
                self.grid[y][ox] = ENERGY
                self.energy.append((x, y))
                self.energy.append((ox,y))

                placed += 2


def is_vertical(snake):
    x0 = snake.body[0][0]
    return all(x == x0 for (x, y) in snake.body)

def move_cost(dir_idx):
    #if dir_idx == 0:  # UP
    #    return 2
    return 1          # others

def compute_all_energy_distances(game):

    distc_maps = []

    for (ex, ey) in game.energy:
               

        dist = [[INF for _ in range(game.W)] for _ in range(game.H)]
        pq = []

        dist[ey][ex] = 0
        heapq.heappush(pq, (0, ex, ey))

        while pq:
            d, x, y = heapq.heappop(pq)

            if d != dist[y][x]:
                continue

            for i, (dx, dy) in enumerate(DIRS):
                nx, ny = x + dx, y + dy

                if not (0 <= nx < game.W and 0 <= ny < game.H):
                    continue

                if game.grid[ny][nx] == WALL:
                    continue

                nd = d + 1#(2 if i == 0 else 1)

                if nd < dist[ny][nx]:
                    dist[ny][nx] = nd
                    heapq.heappush(pq, (nd, nx, ny))

        distc_maps.append(dist)

    return distc_maps


def flood_fill_limited(start_x, start_y, game, max_iter=30):
    visited = set()
    q = deque()
    
    q.append((start_x, start_y))
    visited.add((start_x, start_y))
    
    count = 0

    while q and count < max_iter:
        x, y = q.popleft()
        count += 1

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy

            # limites map
            if nx < 0 or ny < 0 or nx >= game.W or ny >= game.H:
                continue

            # obstacles
            if game.grid[ny][nx] == WALL:
                continue

            # snake
            occupied = any((nx, ny) in s.body for s in game.snakes if s.alive)
            if occupied:
                continue

            if (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))

    return count

def get_moves(game, i, H, W):

    moves = []
    x, y = game.snakes[i].head()

    for d,(dx,dy) in enumerate(DIRS):
        nx, ny = x+dx, y+dy

        if nx < 0 or ny < 0 or nx >= W or ny >= H:
            continue

        if game.grid[ny][nx] == WALL:
            continue

        if any((nx, ny) in s.body for s in game.snakes if s.alive):
            continue

        moves.append(d)

    return moves if moves else [0]

def get_movesIA(game, i, H, W):

    move = [True] * 4
    x, y = game.snakes[i].head()

    for d,(dx,dy) in enumerate(DIRS):
        nx, ny = x+dx, y+dy

        if nx < 0 or ny < 0 or nx >= W or ny >= H:
            move[d] = False
            continue

        if game.grid[ny][nx] == WALL:
            move[d] = False
            continue

        if any((nx, ny) in s.body for s in game.snakes if s.alive):
            move[d] = False
            continue

        move[d] = True

    return move

def encode_state_full(snake, energies, enemies, W, H, MAX_LEN,
                      max_energies=5, max_enemies=3, prev_dir=None, flood=0):
    """
    snake: objet Snake
    energies: liste de (x,y)
    enemies: liste de Snake
    prev_dir: tuple (dx,dy)
    
    max_energies: nombre max d'�nergies � encoder
    max_enemies: nombre max d'ennemis � encoder
    """

    hx, hy = snake.head()

    state_vec = []

    # --- position de la t�te
    state_vec.extend([hx/W, hy/H])

    # --- encode distances aux �nergies
    energies_sorted = sorted(energies, key=lambda e: abs(e[0]-hx) + abs(e[1]-hy))
    for i in range(max_energies):
        if i < len(energies_sorted):
            ex, ey = energies_sorted[i]
            state_vec.extend([(ex-hx)/W, (ey-hy)/H])
        else:
            state_vec.extend([0.0, 0.0])  # padding

    # --- encode distances aux ennemis
    enemies_sorted = sorted(enemies, key=lambda e: abs(e.head()[0]-hx) + abs(e.head()[1]-hy))
    for i in range(max_enemies):
        if i < len(enemies_sorted):
            ex, ey = enemies_sorted[i].head()
            state_vec.extend([(ex-hx)/W, (ey-hy)/H, len(enemies_sorted[i].body)/MAX_LEN])
        else:
            state_vec.extend([0.0, 0.0, 0.0])

    # --- taille du serpent
    state_vec.append(len(snake.body)/MAX_LEN)

    #flood fill
    #state_vec.append(flood/30.0)

    # --- orientation pr�c�dente
    if prev_dir is None:
        state_vec.extend([0.0, 0.0])
    else:
        dx_prev, dy_prev = prev_dir
        state_vec.extend([dx_prev, dy_prev])

    return np.array(state_vec, dtype=np.float32)

class Node:
    __slots__ = ("parent", "first_child", "child_count", "score", "visits", "move", "mult", "prior", "game")

    def __init__(self, parent=-1):
        self.parent = parent
        self.first_child = -1
        self.child_count = 0
        self.score = 0.0
        self.visits = 0
        self.move = -1
        self.mult = 1.0
        self.prior = 0.0
        self.game = None


MAX_NODE = 10000
MAX_CHILD = 40000

MAX_SNAKES = 8
MAX_POWER = 400

# --- Pos
class Pos:
    __slots__ = ("x", "y")

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def copy(self):
        return Pos(self.x, self.y)

# --- Grid
class Grid:
    __slots__ = ("cell",)

    def __init__(self, H, W):
        # tableau 2D
        self.cell = [[0 for _ in range(W)] for _ in range(H)]

    def copy(self):
        new_grid = Grid(len(self.cell), len(self.cell[0]))
        # deep copy des lignes
        new_grid.cell = [row[:] for row in self.cell]
        return new_grid


# --- GameState
class GameState:
    __slots__ = (
        "w", "h",
        "gridu",
        "grid",
        "snakeCount",
        "snakes",
        "energyCount",
        "energy"
    )

    def __init__(self, w, h):
        self.w = w
        self.h = h

        self.grid = Grid(h, w)

        self.snakeCount = 0

        #  Snake doit exister (à adapter à ta classe)
        self.snakes = [None] * MAX_SNAKES

        self.energyCount = 0
        self.energy = [Pos() for _ in range(MAX_POWER)]

class SM:
    def __init__(self):
        self.nodes = [Node() for _ in range(MAX_NODE)]
        self.children = [0] * MAX_CHILD

        self.nodeCount = 0
        self.childCount = 0

        self.ITER = 0

        self.params = {}

    def createNode(self, parent=-1):
        id = self.nodeCount
        self.nodeCount += 1

        n = self.nodes[id]
        n.parent = parent
        n.first_child = -1
        n.child_count = 0
        n.score = 0.0
        n.visits = 0
        n.move = UP
        n.mult = 1.0

        return id

    def addChild(self, parent, child):
        if self.nodes[parent].first_child == -1:
            self.nodes[parent].first_child = self.childCount

        self.children[self.childCount] = child
        self.childCount += 1
        self.nodes[parent].child_count += 1

    def selection(self, node):
        n = self.nodes[node]

        start = n.first_child
        count = n.child_count

        if count == 0:
            return node  # sécurité

        best_ucb = -1e18
        best_child = self.children[start]

        log_parent = math.log(n.visits + 1)

        for i in range(count):
            c = self.children[start + i]
            child = self.nodes[c]

            if child.visits == 0:
                ucb = 1e18
            else:
                c_explore = self.params['Cexplore']

                exploit = child.score / child.visits
                explore = c_explore * math.sqrt(log_parent / child.visits)

                ucb = exploit + explore
                ucb *= child.mult

            if ucb > best_ucb:
                best_ucb = ucb
                best_child = c

        return best_child

    def selection_puct2(self, node_idx):
        """
        Sélectionne le meilleur enfant selon PUCT pour ton système
        nodes[node_idx] = parent node
        children = tableau continu des indices des enfants
        """
        n = self.nodes[node_idx]

        start = n.first_child
        count = n.child_count

        if count == 0:
            return node_idx  # pas d'enfant, sécurité

        best_score = -1e18
        best_child_idx = self.children[start]

        log_parent = math.log(n.visits + 1)

        for i in range(count):
            c_idx = self.children[start + i]
            child = self.nodes[c_idx]

            Q = 0.0
            # valeur Q (exploitation)
            if child.visits > 0:
                Q = child.score / child.visits
                       

            # prior / exploration U
            P = child.prior if hasattr(child, 'prior') else 1.0 / count  # si pas de policy, uniform
            U = self.params['Cexplore'] * P * math.sqrt(log_parent) / (1 + child.visits)

            ucb = Q + U
            ucb *= child.mult  # multiplicateur spécifique à ton système

            if ucb > best_score:
                best_score = ucb
                best_child_idx = c_idx

        return best_child_idx

    def selection_puct(self, node_idx):
        """
        Sélectionne le meilleur enfant selon PUCT pour ton système
        nodes[node_idx] = parent node
        children = tableau continu des indices des enfants
        """
        n = self.nodes[node_idx]

        start = n.first_child
        count = n.child_count

        if count == 0:
            return node_idx  # pas d'enfant, sécurité

        best_score = -1e18
        best_child_idx = self.children[start]

        log_parent = math.log(n.visits + 1)

        for i in range(count):
            c_idx = self.children[start + i]
            child = self.nodes[c_idx]

            Q = 0.0
            # valeur Q (exploitation)
            if child.visits == 0:
                ucb = 1e18
            else:
                Q = child.score / child.visits
                       
                # prior / exploration U
                P = child.prior if hasattr(child, 'prior') else 1.0 / count  # si pas de policy, uniform
                U = self.params['Cexplore'] * P * math.sqrt(log_parent) / (1 + child.visits)

                ucb = Q + U
                ucb *= child.mult  # multiplicateur spécifique à ton système

            if ucb > best_score:
                best_score = ucb
                best_child_idx = c_idx

        return best_child_idx

    def is_fully_vertical(self, snake):
        if snake.len < 3:
            return False

        head_x = snake.body[snake.head].x

        for i in range(snake.len):
            idx = (snake.tail + i) % SnakeS.MAX_BODY
            if snake.body[idx].x != head_x:
                return False

        return True

    def expand(self, node, game, ind_snake, opp_snake_len=0):

        grcurp = game.grid
        snake = game.snakes[ind_snake]

        h = snake.body[snake.head]  # headPos

        # --- moves normaux
        for i in range(4):

            nx = h.x + DX[i]
            ny = h.y + DY[i]

            if nx < 0 or ny < 0 or nx >= game.w or ny >= game.h:
                continue

            if grcurp.cell[ny][nx] & WALL:
                continue

            #sf = self.get_safe_moves(nx, ny, game.grid, game.w, game.h)
            #if sf == 0:
            #    continue

            if i == 0 and grcurp.cell[ny][nx] == EMPTY and (self.is_fully_vertical(snake)):
                continue

            # --- collision corps
            #coll = False
            #for k in range(snake.len):
            #    curid = (snake.tail + k) % SnakeS.MAX_BODY
            #    cur = snake.body[curid]

            #    if nx == cur.x and ny == cur.y:
            #        coll = True
            #        break

            #if coll:
            #    continue

            curid = (game.snakes[ind_snake].head - 1 + MAX_BODY) % MAX_BODY;
            cur = game.snakes[ind_snake].body[curid];
            if nx == cur.x and ny == cur.y:
                continue

            child = self.createNode(node)
            self.nodes[child].move = i
            if grcurp.cell[ny][nx] & ENERGY:
                self.nodes[child].mult = self.params['eat']

            self.addChild(node, child)

        """
            # --- fallback (aucun move)
            if self.nodes[node].child_count == 0:

                bestDir = 0
                minDist = 1e9

                tail = snake.body[snake.tail]

                for i in range(4):

                    nx = h.x + DX[i]
                    ny = h.y + DY[i]

                    if nx < 0 or ny < 0 or nx >= game.w or ny >= game.h:
                        continue

                    if grcurp.cell[ny][nx] & WALL:
                        continue

                    if(self.is_fully_vertical(snake) and i == 0 and grcurp.cell[ny][nx] == EMPTY):
                        continue

                    coll = False
                    for k in range(1, snake.len):  # skip head
                        curid = (snake.tail + k) % SnakeS.MAX_BODY
                        cur = snake.body[curid]

                        if nx == cur.x and ny == cur.y:
                            coll = True
                            break

                    if coll:
                        continue

                    dist = abs(nx - tail.x) + abs(ny - tail.y)

                    if dist < minDist:
                        minDist = dist
                        bestDir = i

                child = self.createNode(node)
                self.nodes[child].move = bestDir
                self.addChild(node, child)
        """

    def expandPUCT(self, node_idx, game, ind_snake, opp_snake_len=0, policy_priors=None):
        """
        Expansion d'un noeud avec création d'enfants et assignation des priors pour PUCT.
        policy_priors : tableau de 4 floats correspondant à la probabilité de chaque move
        """
        grcurp = game.grid
        snake = game.snakes[ind_snake]
        h = snake.body[snake.head]  # headPos

        # --- moves normaux ---
        for i in range(4):
            nx = h.x + DX[i]
            ny = h.y + DY[i]

            if nx < 0 or ny < 0 or nx >= game.w or ny >= game.h:
                continue

            if grcurp.cell[ny][nx] & WALL:
                continue

            #sf = self.get_safe_moves(nx, ny, game.grid, game.w, game.h)
            #if sf == 0:
            #    continue

            if i == 0 and grcurp.cell[ny][nx] == EMPTY and self.is_fully_vertical(snake):
                continue

            # --- collision corps ---
            #coll = False
            #for k in range(snake.len):
            #    curid = (snake.tail + k) % SnakeS.MAX_BODY
            #    cur = snake.body[curid]
            #    if nx == cur.x and ny == cur.y:
            #        coll = True
            #        break
            #if coll:
            #    continue

            curid = (game.snakes[ind_snake].head - 1 + MAX_BODY) % MAX_BODY;
            cur = game.snakes[ind_snake].body[curid];
            if nx == cur.x and ny == cur.y:
                continue
            

            child = self.createNode(node_idx)
            self.nodes[child].move = i

            # assigner multiplicateur si c'est de l'energy
            if grcurp.cell[ny][nx] & ENERGY:
                self.nodes[child].mult = self.params['eat']

            # assigner le prior (policy network ou uniform si non fourni)
            if policy_priors is not None:
                self.nodes[child].prior = policy_priors[i]
            else:
                self.nodes[child].prior = 1.0 / 4  # uniform

            self.addChild(node_idx, child)

        """
        # --- fallback (aucun move valide) ---
            if self.nodes[node_idx].child_count == 0:
                bestDir = 0
                minDist = 1e9
                tail = snake.body[snake.tail]

                for i in range(4):
                    nx = h.x + DX[i]
                    ny = h.y + DY[i]
                    if nx < 0 or ny < 0 or nx >= game.w or ny >= game.h:
                        continue
                    if grcurp.cell[ny][nx] & WALL:
                        continue
                    if self.is_fully_vertical(snake) and i == 0 and grcurp.cell[ny][nx] == EMPTY:
                        continue

                    coll = False
                    for k in range(1, snake.len):  # skip head
                        curid = (snake.tail + k) % SnakeS.MAX_BODY
                        cur = snake.body[curid]
                        if nx == cur.x and ny == cur.y:
                            coll = True
                            break
                    if coll:
                        continue

                    dist = abs(nx - tail.x) + abs(ny - tail.y)
                    if dist < minDist:
                        minDist = dist
                        bestDir = i

                child = self.createNode(node_idx)
                self.nodes[child].move = bestDir
                self.nodes[child].prior = policy_priors[bestDir] if policy_priors is not None else 1.0 / 4
                self.addChild(node_idx, child)
        """

    def backprop(self, node, result):

        while node != -1:
            n = self.nodes[node]
            n.visits += 1
            n.score += result
            node = n.parent

    import torch.nn.functional as F

    def get_policy_priors(self, model, snake_idx, game):
        """
        Retourne un tableau [4] des probabilités de moves pour la tête du snake_idx.
        """
        snake = game.snakes[snake_idx]

        sn = []
        for k in range(snake.len):
            id_ = (snake.head - k + MAX_BODY) % MAX_BODY
            x, y = snake.body[id_].x, snake.body[id_].y
            sn.append((x, y))
        snakegame = Snake(sn)

        snake = game.snakes[1-snake_idx]

        sn = []
        for k in range(snake.len):
            id_ = (snake.head - k + MAX_BODY) % MAX_BODY
            x, y = snake.body[id_].x, snake.body[id_].y
            sn.append((x, y))
        snakegame2 = Snake(sn)

        energy = []
        for p in game.energy:
            energy.append((p.x, p.y))

    
        # Encode l'état du jeu pour le snake
        state = encode_state_full(snakegame, energy, [snakegame2], game.w, game.h, 256)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # batch=1

        
        with torch.no_grad():
            logits = model(state_tensor)  # sortie brute du policy network
            logits = logits.squeeze(0)    # shape = [4] si tu as 4 actions

            """
                    # Masquer les moves impossibles
                    mask = torch.zeros(4, dtype=torch.bool)
                    head = snake.body[snake.head]
                    for i, (dx, dy) in enumerate(DIRS):
                        nx, ny = head.x + dx, head.y + dy
                        if nx < 0 or nx >= game.w or ny < 0 or ny >= game.h:
                            mask[i] = True
                        elif game.grid.cell[ny][nx] & WALL:
                            mask[i] = True
                        # tu peux aussi ajouter collisions avec corps ou safe_moves
                        elif self.get_safe_moves(nx, ny, game.grid, game.w, game.h) == 0:
                            mask[i] = True
                        else:
                            coll = False
                            for k in range(1, snake.len):  # skip head
                                curid = (snake.tail + k) % SnakeS.MAX_BODY
                                cur = snake.body[curid]
                                if nx == cur.x and ny == cur.y:
                                    coll = True
                                    break
                            if coll:
                                mask[i] = True

                    logits[mask] = -1e9  # invalide moves
            """
            # softmax pour obtenir les probabilités
            policy_priors = F.softmax(logits, dim=0).cpu().numpy()

        return policy_priors  # array de 4 floats sommant à 1


    def doFalls2(self, g, total_snakes):

        fallDist = [0] * total_snakes
        alive = [True] * total_snakes

        airborne = []
        grounded = []

        # init airborne
        for i in range(total_snakes):
            if g.snakes[i].alive:
                airborne.append(i)

        while True:

            somethingFell = False
            # --- propagation grounded

            changed = True
            while changed:
                changed = False

                idx = 0
                while idx < len(airborne):
                    i = airborne[idx]
                    s = g.snakes[i]

                    isGrounded = False

                    for k in range(s.len):
                        id_ = (s.tail + k) % MAX_BODY
                        c = s.body[id_]

                        # sol
                        if c.y + 1 >= g.h:
                            isGrounded = True
                            break

                        # mur / énergie
                        if 0<=c.x<g.w and 0<=c.y+1<g.h and (g.grid.cell[c.y+1][c.x] & (WALL | ENERGY)):
                            isGrounded = True
                            break

                        # touche grounded
                        for j in range(total_snakes):
                            if j == i:
                                continue

                            os = g.snakes[j]
                            if not os.alive:
                                continue

                            for kk in range(os.len):
                                id2 = (os.tail + kk) % MAX_BODY
                                p2 = os.body[id2]

                                if p2.x == c.x and p2.y == c.y + 1:
                                    isGrounded = True
                                    break
                            if isGrounded:
                                break

                        if isGrounded:
                            break

                    if isGrounded:
                        grounded.append(i)
                        airborne[idx] = airborne[-1]
                        airborne.pop()
                        changed = True
                    else:
                        idx += 1

            # --- chute
            """
                for i in airborne:
                    s = g.snakes[i]

                    somethingFell = True

                    for k in range(s.len):
                        id_ = (s.tail + k) % MAX_BODY
                        s.body[id_].y += 1

                    fallDist[i] += 1
            """
            
                
            for i in airborne:
                s = g.snakes[i]

                canFall = True

                for k in range(s.len):
                    id_ = (s.tail + k) % MAX_BODY
                    c = s.body[id_]

                    if c.y + 1 >= g.h:
                        canFall = False
                        break

                    if 0 <= c.x < g.w and 0 <= c.y+1 < g.h:
                        if g.grid.cell[c.y+1][c.x] & (WALL | ENERGY):
                            canFall = False
                            break

                if not canFall:
                    continue
                
                somethingFell = True


                for k in range(s.len):
                    id_ = (s.tail + k) % MAX_BODY
                    s.body[id_].y += 1

                fallDist[i] += 1


                # death
                dead = True
                for k in range(s.len):
                    id_ = (s.tail + k) % MAX_BODY
                    if s.body[id_].y < g.h + 1:
                        dead = False
                        break

                if dead:
                    s.alive = False

            if not somethingFell:
                break

    def touchingVertical2(self, a, b):

        for i in range(a.len):
            id1 = (a.tail + i) % MAX_BODY
            p1 = a.body[id1]

            for j in range(b.len):
                id2 = (b.tail + j) % MAX_BODY
                p2 = b.body[id2]

                if abs(p1.x - p2.x) == 0 and abs(p1.y - p2.y) == 1:
                    return True

        return False


    def getIntercoiled(self, g, total):

        groups = []
        vis = [False] * total

        for i in range(total):
            if vis[i] or not g.snakes[i].alive:
                continue

            group = []
            q = deque([i])

            while q:
                u = q.popleft()

                if vis[u]:
                    continue

                vis[u] = True
                group.append(u)

                for v in range(total):
                    if u == v or vis[v]:
                        continue
                    if not g.snakes[v].alive:
                        continue

                    if self.touchingVertical2(g.snakes[u], g.snakes[v]):
                        q.append(v)

            if len(group) > 1:
                groups.append(group)

        return groups

    def doIntercoiledFalls(self, g, total):

        while True:

            fell = False
            groups = self.getIntercoiled(g, total)

            for grp in groups:

                canFall = True

                for id_ in grp:
                    s = g.snakes[id_]

                    for k in range(s.len):
                        idx = (s.tail + k) % MAX_BODY
                        c = s.body[idx]

                        if c.y + 1 >= g.h:
                            canFall = False
                            break

                        if 0 <= c.x < g.w and (g.grid.cell[c.y+1][c.x] & (WALL | ENERGY)):
                            canFall = False
                            break

                    if not canFall:
                        break

                if canFall:
                    fell = True

                    for id_ in grp:
                        s = g.snakes[id_]

                        for k in range(s.len):
                            idx = (s.tail + k) % MAX_BODY
                            s.body[idx].y += 1

            if not fell:
                break

    def something_solid_under3(self, g, x, y, ignore, total):
        ny = y + 1

        # hors map = sol
        if ny >= g.h:
            return True

        # mur
        if g.grid.cell[ny][x] & WALL:
            return True

        # énergie
        if g.grid.cell[ny][x] & ENERGY:
            return True

        # snake
        for i in range(total):
            s = g.snakes[i]
            if s is None or not s.alive:
                continue

            for k in range(s.len):
                idx = (s.tail + k) % MAX_BODY
                p = s.body[idx]

                if (p.x, p.y) in ignore:
                    continue

                if p.x == x and p.y == ny:
                    return True

        return False

    def something_solid_under(self, g, x, y, meta, total):
        ny = y + 1

        # --- hors map = solide
        if x < 0 or x >= g.w or ny < 0:
            return True

        # hors map = sol
        if ny >= g.h:
            return True

        # ⚠️ IGNORE groupe entier SI EN DESSOUS
        if (x, ny) in meta:
            return False

        # mur
        if g.grid.cell[ny][x] & WALL:
            return True

        # énergie
        if g.grid.cell[ny][x] & ENERGY:
            return True

        # autres snakes
        for i in range(total):
            s = g.snakes[i]
            if s is None or not s.alive:
                continue

            for k in range(s.len):
                idx = (s.tail + k) % MAX_BODY
                p = s.body[idx]

                if p.x == x and p.y == ny:
                    return True

        return False

    def touching_vertical(self, a, b):
        for i in range(a.len):
            id1 = (a.tail + i) % MAX_BODY
            p1 = a.body[id1]

            for j in range(b.len):
                id2 = (b.tail + j) % MAX_BODY
                p2 = b.body[id2]

                if p1.x == p2.x and abs(p1.y - p2.y) == 1:
                    return True

        return False

    def get_groups(self, g, total):
        visited = [False] * total
        groups = []

        for i in range(total):
            if visited[i] or not g.snakes[i].alive:
                continue

            stack = [i]
            group = []

            while stack:
                u = stack.pop()
                if visited[u]:
                    continue

                visited[u] = True
                group.append(u)

                for v in range(total):
                    if v == u or visited[v]:
                        continue
                    if not g.snakes[v].alive:
                        continue

                    if self.touching_vertical(g.snakes[u], g.snakes[v]):
                        stack.append(v)

            groups.append(group)

        return groups

    def doFalls(self, g, total):

        while True:
            something_fell = False

            groups = self.get_groups(g, total)

            for grp in groups:

                # --- build meta body
                meta = set()
                for i in grp:
                    s = g.snakes[i]
                    for k in range(s.len):
                        idx = (s.tail + k) % MAX_BODY
                        p = s.body[idx]
                        meta.add((p.x, p.y))

                # --- check si peut tomber
                can_fall = True

                for (x, y) in meta:
                    if self.something_solid_under(g, x, y, meta, total):
                        can_fall = False
                        break

                # --- chute
                if can_fall:
                    something_fell = True

                    for i in grp:
                        s = g.snakes[i]

                        for k in range(s.len):
                            idx = (s.tail + k) % MAX_BODY
                            s.body[idx].y += 1

                        # mort hors map
                        dead = True
                        for k in range(s.len):
                            idx = (s.tail + k) % MAX_BODY
                            if s.body[idx].y < g.h + 1:
                                dead = False
                                break

                        if dead:
                            s.alive = False

            if not something_fell:
                break


    def playMoveTurn(self, g, score, opp_len):

        total = g.snakeCount + opp_len
        ven = []  # positions energy à supprimer

        # --- MOVE + ENERGY
        for id in range(total):

            s = g.snakes[id]
            if not s.alive:
                continue

            h = s.body[s.head]
            move = s.dir

            nx = h.x + DX[move]
            ny = h.y + DY[move]

            # --- move normal
            s.tail = (s.tail + 1) % MAX_BODY
            s.head = (s.head + 1) % MAX_BODY
            s.body[s.head] = Pos(nx, ny)

            #print(nx, ny, g.w, g.h)
            # --- energy
            if 0<=nx<g.w and 0<=ny<g.h and g.grid.cell[ny][nx] & ENERGY:
                s.tail = (s.tail - 1 + MAX_BODY) % MAX_BODY
                ven.append((nx, ny))
                s.len += 1
                score[id] += self.params['eat']

        # --- remove energy
        for (nx, ny) in ven:
            g.grid.cell[ny][nx] &= ~ENERGY

        # --- COLLISIONS
        for id in range(total):

            s = g.snakes[id]
            if not s.alive:
                continue

            h = s.body[s.head]
            nx, ny = h.x, h.y

            collision = False
            ind_other = -1
            head = False

            for j in range(total):
                end = 0
                if id == j:
                    end = 1

                other = g.snakes[j]
                if not other.alive:
                    continue

                cur = other.tail
                for k in range(other.len-end):
                    b = other.body[(cur + k) % MAX_BODY]

                    if b.x == nx and b.y == ny:
                        ind_other = j
                        if k == other.len - 1:
                            head = True
                        collision = True
                        break

                    #cur = (cur + 1) % MAX_BODY

                if collision:
                    break

            if collision:
                # --- rollback
                prev = (s.head - 1 + MAX_BODY) % MAX_BODY
                s.head = prev
                s.len -= 1

                if s.len < 3:
                    s.alive = False
                else:
                    score[id] += self.params['lose_part']

                # --- head-to-head
                if head:
                    so = g.snakes[ind_other]

                    prev2 = (so.head - 1 + MAX_BODY) % MAX_BODY
                    so.head = prev2
                    so.len -= 1

                    if so.len < 3:
                        so.alive = False

                        if s.alive:
                            if id < g.snakeCount and ind_other >= g.snakeCount:
                                score[id] += self.params['kill']
                            elif id < g.snakeCount and ind_other < g.snakeCount:
                                score[id] += self.params['kill_dude']
                            elif id >= g.snakeCount and ind_other >= g.snakeCount:
                                score[id] += self.params['kill_dude']
                            elif id >= g.snakeCount and ind_other < g.snakeCount:
                                score[id] += self.params['kill']

                    elif s.len > so.len:
                        score[id] += self.params['lose_part'] * -2.0
                

        # --- gravité
        self.doFalls2(g, total)
        #self.doIntercoiledFalls(g, total)

        needIntercoil = False

        for i in range(total):
            si = g.snakes[i]
            if si is None or not si.alive:
                continue

            for j in range(i + 1, total):
                sj = g.snakes[j]
                if sj is None or not sj.alive:
                    continue

                if self.touchingVertical2(si, sj):
                    needIntercoil = True
                    break

            if needIntercoil:
                break

        if needIntercoil:
            self.doIntercoiledFalls(g, total)

    def getDirection(self, s):

        head = s.body[s.head]
        next_idx = (s.head - 1 + MAX_BODY) % MAX_BODY
        neck = s.body[next_idx]

        dx = head.x - neck.x
        dy = head.y - neck.y

        if dx == 0 and dy == -1:
            return 0  # UP
        if dx == 0 and dy == 1:
            return 1  # DOWN
        if dx == -1 and dy == 0:
            return 2  # LEFT
        if dx == 1 and dy == 0:
            return 3  # RIGHT

        return -1

    def best_child_index(self, root_node):
        start = self.nodes[root_node].first_child
        best_child = -1
        best_score = -1e18

        for j in range(self.nodes[root_node].child_count):
            c = self.children[start + j]  # indice réel de l'enfant
            avg = self.nodes[c].score / self.nodes[c].visits if self.nodes[c].visits > 0 else 0.0
            avg *= self.nodes[c].mult
            if avg > best_score:
                best_score = avg
                best_child = c

            # debug
            #print(f"move={self.nodes[c].move} visits={self.nodes[c].visits} "
            #      f"score={self.nodes[c].score:.6f} avg={avg:.6f}")

        return best_child, best_score  # retourne l’indice réel dans nodes[]

    def best_move(self, root_node):
        c, sc = self.best_child_index(root_node)
        if c != -1:
            return self.nodes[c].move, sc
        return -1, -1e18  # pas de coup valide
   

    def flood_fill(self, grid, sx, sy, W, H, step=30):
        """
        grid: Grid avec grid.cell[y][x]
        sx, sy: position de départ
        W, H: dimensions
        step: limite du flood
        """
        if sx < 0 or sx >= W or sy < 0 or sy >= H:
            return 0


        vis = [[False for _ in range(W)] for _ in range(H)]
        q = deque()
        q.append((sx, sy))
        vis[sy][sx] = True

        count = 0

        while q:
            x, y = q.popleft()
            count += 1
            if count > step:
                break

            for d in range(4):
                nx = x + DX[d]
                ny = y + DY[d]

                if nx < 0 or ny < 0 or nx >= W or ny >= H:
                    continue
                if vis[ny][nx]:
                    continue
                if grid.cell[ny][nx] & (WALL | SNAKE):
                    continue

                vis[ny][nx] = True
                q.append((nx, ny))

        return count

    def play(self, width, height, my_snake, my_snake_len,
             opp_snake, opp_snake_len,
             energy, power_source_count, grid, distg, parentg, time_limit_ms):

        start = time.perf_counter()

        def getTime():
            return (time.perf_counter() - start) * 1000 <= time_limit_ms

        totalSnake = my_snake_len + opp_snake_len

        self.nodeCount = 0
        self.childCount = 0

        root = [self.createNode(-1) for _ in range(totalSnake)]

        DEPTH = 9 - my_snake_len
        turn = 0


        while getTime():
        #for ILOOP in range(2000):
            node = root[:]  # copie

            # --- init game
            game = GameState(width, height)
            game.snakeCount = my_snake_len

            # --- my snakes
            for i in range(my_snake_len):
                game.snakes[i] = my_snake[i].copy()  # ⚠️ important
                game.snakes[i].dir = UP
                game.snakes[i].alive = True

            # --- opp snakes
            for i in range(opp_snake_len):
                game.snakes[my_snake_len + i] = opp_snake[i].copy()
                game.snakes[my_snake_len + i].dir = UP
                game.snakes[my_snake_len + i].alive = True

            # --- energy
            for i in range(power_source_count):
                game.energy[i] = energy[i].copy()

            game.grid = grid.copy()
            game.energyCount = power_source_count

            score = [0.0] * 8

            # =====================
            #  SIMULATION
            # =====================
            for depth in range(DEPTH):

                for i in range(totalSnake):

                    s = game.snakes[i]
                    if not s.alive:
                        continue

                    if self.nodes[node[i]].first_child == -1:
                        self.expand(node[i], game, i, opp_snake_len)

                        if self.nodes[node[i]].child_count == 0:
                            s.dir = self.getDirection(s)
                            continue

                    node[i] = self.selection(node[i])
                    s.dir = self.nodes[node[i]].move

                self.playMoveTurn(game, score, opp_snake_len)

                # --- stop si plus d'énergie
                count = 0
                for e in game.energy:
                    if game.grid.cell[e.y][e.x] & ENERGY:
                        count += 1

                if count == 0:
                    break

            # =====================
            #  EVALUATION
            # =====================
            tot_team = 0
            tot_opp = 0

            for i in range(totalSnake):
                s = game.snakes[i]
                if s.alive:
                    if i < game.snakeCount:
                        tot_team += 1
                    else:
                        tot_opp += 1

            # --- copy grid
            grcurp = game.grid.copy()

            # --- mark snakes
            for i in range(totalSnake):
                s = game.snakes[i]
                for k in range(s.len):
                    idx = (s.tail + k) % MAX_BODY
                    p = s.body[idx]

                    if 0 <= p.y < game.h and 0 <= p.x < game.w:
                        grcurp.cell[p.y][p.x] |= SNAKE

            # --- score final
            for i in range(totalSnake):

                s = game.snakes[i]
                hx = s.body[s.head].x
                hy = s.body[s.head].y

                sc = score[i]

                # --- mort
                if not s.alive:
                    sc = self.params['death']
                    self.backprop(node[i], sc / 200.0)
                    continue

                # --- taille
                sc += s.len * self.params['size']

                # --- distance énergie
                bestDist = 1e9
                count = 0

                if 0 <= hx < game.w and 0 <= hy < game.h:
                    bestDist = 1e9
                    sumScore = 0

                    for ind in range(power_source_count):
                        e = game.energy[ind]

                        if game.grid.cell[e.y][e.x] & ENERGY:
                            gap, di = self.compute_max_gap(hx, hy, parentg[e.y * width + e.x], grcurp, game.w, game.h)

                            if gap > s.len - 2:
                                gap += 1000

                            d = di + gap

                            if d < bestDist:
                                bestDist = d

                            count += 1

                            sumScore += self.params['dist'] / (distg[e.y * width + e.x][hy][hx] + 1)

                    # combo
                    if bestDist < 1e9:
                        sc += self.params['dist']*30.0 / (bestDist + 1)  # focus
                        sc += 0.3 * sumScore  # global attraction

                # --- plus d'énergie
                if count == 0:
                    if i < game.snakeCount:
                        sc += self.params['win'] if tot_team > tot_opp else self.params['lose']
                    else:
                        sc += self.params['win'] if tot_opp > tot_team else self.params['lose']

                # --- flood fill
                fl = self.flood_fill(grcurp,hx, hy, game.w, game.h, min(4, s.len))
                if fl < min(4, s.len):
                    sc += self.params['flood']

                self.backprop(node[i], sc / 200.0)

            turn += 1

        #print("TURN", turn)

        indc = []
        sc = []
        for i in range(my_snake_len):
            a, s = self.best_move(root[i])
            indc.append(a)
            sc.append(s)

        return indc, sc[0]

    def get_safe_moves(self, hx, hy, grid, w, h):
        count = 0
        for dx, dy in DIRS:
            nx, ny = hx + dx, hy + dy

            # vérifier qu'on reste dans la grille
            if not (0 <= nx < w and 0 <= ny < h):
                continue

            # vérifier collision mur ou snake
            if grid.cell[ny][nx] & (WALL | SNAKE):
                continue

            count += 1

        return count


    def playVN(self, width, height, my_snake, my_snake_len,
             opp_snake, opp_snake_len,
             energy, power_source_count, grid, distg, time_limit_ms, model):

        start = time.perf_counter()

        def getTime():
            return (time.perf_counter() - start) * 1000 <= time_limit_ms

        totalSnake = my_snake_len + opp_snake_len

        self.nodeCount = 0
        self.childCount = 0

        root = [self.createNode(-1) for _ in range(totalSnake)]

        DEPTH = 9 - my_snake_len
        turn = 0


        while getTime():
        #for ILOOP in range(2000):
            node = root[:]  # copie

            # --- init game
            game = GameState(width, height)
            game.snakeCount = my_snake_len
            #print(type(my_snake))
            # --- my snakes
            for i in range(my_snake_len):
                game.snakes[i] = my_snake[i].copy()  # ⚠️ important
                game.snakes[i].dir = UP
                game.snakes[i].alive = True

            # --- opp snakes
            for i in range(opp_snake_len):
                game.snakes[my_snake_len + i] = opp_snake[i].copy()
                game.snakes[my_snake_len + i].dir = UP
                game.snakes[my_snake_len + i].alive = True

            # --- energy
            for i in range(power_source_count):
                game.energy[i] = energy[i].copy()

            game.grid = grid.copy()
            game.energyCount = power_source_count

            score = [0.0] * 8

            # =====================
            #  SIMULATION
            # =====================
            for depth in range(DEPTH):

                for i in range(totalSnake):

                    s = game.snakes[i]
                    if not s.alive:
                        continue

                    if self.nodes[node[i]].first_child == -1:
                        self.expand(node[i], game, i, opp_snake_len)

                        if self.nodes[node[i]].child_count == 0:
                            s.dir = getDirection(s)
                            continue

                    node[i] = self.selection(node[i])
                    s.dir = self.nodes[node[i]].move

                self.playMoveTurn(game, score, opp_snake_len)

                # --- stop si plus d'énergie
                count = 0
                for e in game.energy:
                    if game.grid.cell[e.y][e.x] & ENERGY:
                        count += 1

                if count == 0:
                    break

            # =====================
            #  EVALUATION
            # =====================
            tot_team = 0
            tot_opp = 0

            for i in range(totalSnake):
                s = game.snakes[i]
                if s.alive:
                    if i < game.snakeCount:
                        tot_team += 1
                    else:
                        tot_opp += 1

            # --- copy grid
            grcurp = game.grid.copy()

            # --- mark snakes
            for i in range(totalSnake):
                s = game.snakes[i]
                for k in range(s.len):
                    idx = (s.tail + k) % MAX_BODY
                    p = s.body[idx]

                    if 0 <= p.y < game.h and 0 <= p.x < game.w:
                        grcurp.cell[p.y][p.x] |= SNAKE

            # --- score final
            for i in range(totalSnake):

                s = game.snakes[i]
                hx = s.body[s.head].x
                hy = s.body[s.head].y

                sc = score[i]

              
                # --- mort
                if not s.alive:
                    sc = self.params['death']
                    self.backprop(node[i], (sc / 200.0))
                    continue

                # --- taille
                sc += s.len * self.params['size']

                # --- distance énergie
                bestDist = 1e9
                count = 0

                if 0 <= hx < game.w and 0 <= hy < game.h:
                    for ind in range(power_source_count):
                        e = game.energy[ind]

                        if game.grid.cell[e.y][e.x] & ENERGY:
                            d = distg[e.y * width + e.x][hy][hx] #abs(e.x - hx) + abs(e.y - hy) #d
                            count += 1
                            #if d < bestDist:
                            bestDist = d

                            sc += self.params['dist'] / (bestDist + 1)

                # --- plus d'énergie
                if count == 0:
                    if i < game.snakeCount:
                        sc += self.params['win'] if tot_team > tot_opp else self.params['lose']
                    else:
                        sc += self.params['win'] if tot_opp > tot_team else self.params['lose']

                # --- flood fill
                fl = self.flood_fill(grcurp,hx, hy, game.w, game.h, min(4, s.len))
                if fl < min(4, s.len) or self.get_safe_moves(hx, hy, grcurp, game.w, game.h) == 0 :
                    sc += self.params['flood']

                body = []
                body2 = []

                for k in range(s.len):
                    idx = (s.tail + k) % MAX_BODY
                    p = s.body[idx]
                    body.append((p.x, p.y))

                for k in range(game.snakes[1-i].len):
                    idx = (game.snakes[1-i].tail + k) % MAX_BODY
                    p = game.snakes[1-i].body[idx]
                    body2.append((p.x, p.y))

                my_snake_g = Snake(list(body[::-1]))
                opp_snake_g = Snake(list(body2[::-1]))
                energys = []
                for e in game.energy:
                    energys.append((e.x, e.y))

                state = encode_state_full(my_snake_g, energys, [opp_snake_g], game.w, game.h, 256)

                with torch.no_grad():
                    x = torch.tensor(state).unsqueeze(0)
                    value = model(x).item()
                    #print( sc/200.0, value)
                    
                    self.backprop(node[i], (sc/200.0)*0.6 + value*0.4 )

            turn += 1

        #print("TURN", turn)

        indc = []
        for i in range(my_snake_len):
            a, sc = self.best_move(root[i])
            indc.append(a)

        return indc, sc

    def playVNTraining(self, width, height, my_snake, my_snake_len,
             opp_snake, opp_snake_len,
             energy, power_source_count, grid, distg, time_limit_ms, gg):

        start = time.perf_counter()

        def getTime():
            return (time.perf_counter() - start) * 1000 <= time_limit_ms

        totalSnake = my_snake_len + opp_snake_len

        self.nodeCount = 0
        self.childCount = 0

        root = [self.createNode(-1) for _ in range(totalSnake)]

        DEPTH = 9 - my_snake_len
        turn = 0


        while getTime():
        #for ILOOP in range(2000):
            node = root[:]  # copie

            # --- init game
            game = GameState(width, height)
            game.snakeCount = my_snake_len
            #print(type(my_snake))
            # --- my snakes
            for i in range(my_snake_len):
                game.snakes[i] = my_snake[i].copy()  # ⚠️ important
                game.snakes[i].dir = UP
                game.snakes[i].alive = True

            # --- opp snakes
            for i in range(opp_snake_len):
                game.snakes[my_snake_len + i] = opp_snake[i].copy()
                game.snakes[my_snake_len + i].dir = UP
                game.snakes[my_snake_len + i].alive = True

            # --- energy
            for i in range(power_source_count):
                game.energy[i] = energy[i].copy()

            game.grid = grid.copy()
            game.energyCount = power_source_count

            score = [0.0] * 8

            # =====================
            #  SIMULATION
            # =====================
            for depth in range(DEPTH):

                for i in range(totalSnake):

                    s = game.snakes[i]
                    if not s.alive:
                        continue

                    if self.nodes[node[i]].first_child == -1:
                        self.expand(node[i], game, i, opp_snake_len)

                        if self.nodes[node[i]].child_count == 0:
                            s.dir = getDirection(s)
                            continue

                    node[i] = self.selection(node[i])
                    s.dir = self.nodes[node[i]].move

                self.playMoveTurn(game, score, opp_snake_len)

                # --- stop si plus d'énergie
                count = 0
                for e in game.energy:
                    if game.grid.cell[e.y][e.x] & ENERGY:
                        count += 1

                if count == 0:
                    break

            # =====================
            #  EVALUATION
            # =====================
            tot_team = 0
            tot_opp = 0

            for i in range(totalSnake):
                s = game.snakes[i]
                if s.alive:
                    if i < game.snakeCount:
                        tot_team += 1
                    else:
                        tot_opp += 1

            # --- copy grid
            grcurp = game.grid.copy()

            # --- mark snakes
            for i in range(totalSnake):
                s = game.snakes[i]
                for k in range(s.len):
                    idx = (s.tail + k) % MAX_BODY
                    p = s.body[idx]

                    if 0 <= p.y < game.h and 0 <= p.x < game.w:
                        grcurp.cell[p.y][p.x] |= SNAKE

            # --- score final
            for i in range(totalSnake):

                s = game.snakes[i]
                hx = s.body[s.head].x
                hy = s.body[s.head].y

                sc = score[i]

                # --- mort
                if not s.alive:
                    sc = self.params['death']
                    self.backprop(node[i], sc / 200.0)
                    continue

                # --- taille
                sc += s.len * self.params['size']

                # --- distance énergie
                bestDist = 1e9
                count = 0

                if 0 <= hx < game.w and 0 <= hy < game.h:
                    for ind in range(power_source_count):
                        e = game.energy[ind]

                        if game.grid.cell[e.y][e.x] & ENERGY:
                            d = distg[e.y * width + e.x][hy][hx] #abs(e.x - hx) + abs(e.y - hy) #d
                            count += 1
                            #if d < bestDist:
                            bestDist = d

                            sc += self.params['dist'] / (bestDist + 1)

                # --- plus d'énergie
                if count == 0:
                    if i < game.snakeCount:
                        sc += self.params['win'] if tot_team > tot_opp else self.params['lose']
                    else:
                        sc += self.params['win'] if tot_opp > tot_team else self.params['lose']

                # --- flood fill
                fl = self.flood_fill(grcurp,hx, hy, game.w, game.h, min(4, s.len))
                if fl < min(4, s.len):
                    sc += self.params['flood']

                body = []
                body2 = []

                for k in range(s.len):
                    idx = (s.tail + k) % MAX_BODY
                    p = s.body[idx]
                    body.append((p.x, p.y))

                for k in range(game.snakes[1-i].len):
                    idx = (game.snakes[1-i].tail + k) % MAX_BODY
                    p = game.snakes[1-i].body[idx]
                    body2.append((p.x, p.y))

                my_snake_g = Snake(list(body[::-1]))
                opp_snake_g = Snake(list(body2[::-1]))
                energys = []
                for e in game.energy:
                    energys.append((e.x, e.y))

                state = encode_state_full(my_snake_g, energys, [opp_snake_g], game.w, game.h, 256)

                gg.states.append(state)
                gg.targets.append(sc / 200.0)
                    
                self.backprop(node[i], sc / 200.0)

            turn += 1

        #print("TURN", turn)

        indc = []
        sc = []
        for i in range(my_snake_len):
            a, s = self.best_move(root[i])
            indc.append(a)
            sc.append(s)

        return indc, sc


    def playTrainingPUCT(self, width, height, my_snake, my_snake_len,
             opp_snake, opp_snake_len,
             energy, power_source_count, grid, distg, parentg, time_limit_ms, gg):

        start = time.perf_counter()

        def getTime():
            return (time.perf_counter() - start) * 1000 <= time_limit_ms

        totalSnake = my_snake_len + opp_snake_len

        self.nodeCount = 0
        self.childCount = 0

        root = [self.createNode(-1) for _ in range(totalSnake)]

        DEPTH = 5 #9 - my_snake_len
        turn = 0


        while getTime():
        #for ILOOP in range(2000):
            node = root[:]  # copie

            # --- init game
            game = GameState(width, height)
            game.snakeCount = my_snake_len
            #print(type(my_snake))
            # --- my snakes
            for i in range(my_snake_len):
                game.snakes[i] = my_snake[i].copy()  # ⚠️ important
                game.snakes[i].dir = UP
                game.snakes[i].alive = True

            # --- opp snakes
            for i in range(opp_snake_len):
                game.snakes[my_snake_len + i] = opp_snake[i].copy()
                game.snakes[my_snake_len + i].dir = UP
                game.snakes[my_snake_len + i].alive = True

            # --- energy
            for i in range(power_source_count):
                game.energy[i] = energy[i].copy()

            game.grid = grid.copy()
            game.energyCount = power_source_count

            score = [0.0] * 8

            # =====================
            #  SIMULATION
            # =====================
            for depth in range(DEPTH):

                for i in range(totalSnake):

                    s = game.snakes[i]
                    if not s.alive:
                        continue

                    if self.nodes[node[i]].first_child == -1:
                        self.expand(node[i], game, i, opp_snake_len)

                        if self.nodes[node[i]].child_count == 0:
                            s.dir = getDirection(s)
                            continue

                    node[i] = self.selection(node[i])
                    s.dir = self.nodes[node[i]].move

                self.playMoveTurn(game, score, opp_snake_len)

                # --- stop si plus d'énergie
                count = 0
                for e in game.energy:
                    if game.grid.cell[e.y][e.x] & ENERGY:
                        count += 1

                if count == 0:
                    break

            # =====================
            #  EVALUATION
            # =====================
            tot_team = 0
            tot_opp = 0

            for i in range(totalSnake):
                s = game.snakes[i]
                if s.alive:
                    if i < game.snakeCount:
                        tot_team += 1
                    else:
                        tot_opp += 1

            # --- copy grid
            grcurp = game.grid.copy()

            # --- mark snakes
            for i in range(totalSnake):
                s = game.snakes[i]
                for k in range(s.len):
                    idx = (s.tail + k) % MAX_BODY
                    p = s.body[idx]

                    if 0 <= p.y < game.h and 0 <= p.x < game.w:
                        grcurp.cell[p.y][p.x] |= SNAKE

            # --- score final
            for i in range(totalSnake):

                s = game.snakes[i]
                hx = s.body[s.head].x
                hy = s.body[s.head].y

                sc = score[i]

                # --- mort
                if not s.alive:
                    sc = self.params['death']
                    self.backprop(node[i], sc / 200.0)
                    continue

                # --- taille
                sc += s.len * self.params['size']

                # --- distance énergie
                bestDist = 1e9
                count = 0

                if 0 <= hx < game.w and 0 <= hy < game.h:
                    bestDist = 1e9
                    sumScore = 0

                    for ind in range(power_source_count):
                        e = game.energy[ind]

                        if game.grid.cell[e.y][e.x] & ENERGY:
                            gap, di = self.compute_max_gap(hx, hy, parentg[e.y * width + e.x], grcurp, game.w, game.h)

                            if gap > s.len - 2:
                                gap += 1000

                            d = di + gap

                            if d < bestDist:
                                bestDist = d

                            sumScore += self.params['dist'] / (d + 1)

                    # combo
                    if bestDist < 1e9:
                        sc += self.params['dist'] / (bestDist + 1)  # focus
                        sc += 0.3 * sumScore  # global attraction

                # --- plus d'énergie
                if count == 0:
                    if i < game.snakeCount:
                        sc += self.params['win'] if tot_team > tot_opp else self.params['lose']
                    else:
                        sc += self.params['win'] if tot_opp > tot_team else self.params['lose']

                # --- flood fill
                fl = self.flood_fill(grcurp,hx, hy, game.w, game.h, min(4, s.len))
                if fl < min(4, s.len) or self.get_safe_moves(hx, hy, grcurp, game.w, game.h) == 0 :
                    sc += self.params['flood']
                                                       
                self.backprop(node[i], sc / 200.0)

            turn += 1

        #print("TURN", turn)

        indc = []
        sc = []
        for i in range(my_snake_len):
            a, s = self.best_move(root[i])
            indc.append(a)
            sc.append(s)

     
        start = self.nodes[root[0]].first_child
        visits = [0.0] * 4

        for j in range(self.nodes[root[0]].child_count):
            c = self.children[start + j]
            move = self.nodes[c].move
            visits[move] += self.nodes[c].visits

        # --- softmax avec temperature T ---
        T = 2.0 # ajustable
        log_visits = [math.log(v + 1) for v in visits]
        max_v = max(log_visits)
        exp_visits = [math.exp((v - max_v) / T) for v in log_visits]
        total = sum(exp_visits)
        policy = [v / total for v in exp_visits]
        #print(policy, " - ", max(policy))
           
        return indc, sc, policy

    
    def playPUCT(self, width, height, my_snake, my_snake_len,
             opp_snake, opp_snake_len,
             energy, power_source_count, grid, distg, parentg, time_limit_ms, model):

        start = time.perf_counter()

        def getTime():
            return (time.perf_counter() - start) * 1000 <= time_limit_ms

        totalSnake = my_snake_len + opp_snake_len

        self.nodeCount = 0
        self.childCount = 0

        root = [self.createNode(-1) for _ in range(totalSnake)]

        DEPTH = 5#9 - my_snake_len
        turn = 0


        while getTime():
        #for ILOOP in range(2000):
            node = root[:]  # copie

            # --- init game
            game = GameState(width, height)
            game.snakeCount = my_snake_len
            #print(type(my_snake))
            # --- my snakes
            for i in range(my_snake_len):
                game.snakes[i] = my_snake[i].copy()  # ⚠️ important
                game.snakes[i].dir = UP
                game.snakes[i].alive = True

            # --- opp snakes
            for i in range(opp_snake_len):
                game.snakes[my_snake_len + i] = opp_snake[i].copy()
                game.snakes[my_snake_len + i].dir = UP
                game.snakes[my_snake_len + i].alive = True

            # --- energy
            for i in range(power_source_count):
                game.energy[i] = energy[i].copy()

            game.grid = grid.copy()
            game.energyCount = power_source_count

            score = [0.0] * 8

            # =====================
            #  SIMULATION
            # =====================
            for depth in range(DEPTH):

                for i in range(totalSnake):

                    s = game.snakes[i]
                    if not s.alive:
                        continue

                    if self.nodes[node[i]].first_child == -1:
                        self.expandPUCT(node[i], game, i, opp_snake_len, self.get_policy_priors(model, i, game))

                        if self.nodes[node[i]].child_count == 0:
                            s.dir = self.getDirection(s)
                            continue

                    node[i] = self.selection_puct(node[i])
                    s.dir = self.nodes[node[i]].move

                self.playMoveTurn(game, score, opp_snake_len)

                # --- stop si plus d'énergie
                count = 0
                for e in game.energy:
                    if game.grid.cell[e.y][e.x] & ENERGY:
                        count += 1

                if count == 0:
                    break

            # =====================
            #  EVALUATION
            # =====================
            tot_team = 0
            tot_opp = 0

            for i in range(totalSnake):
                s = game.snakes[i]
                if s.alive:
                    if i < game.snakeCount:
                        tot_team += 1
                    else:
                        tot_opp += 1

            # --- copy grid
            grcurp = game.grid.copy()

            # --- mark snakes
            for i in range(totalSnake):
                s = game.snakes[i]
                for k in range(s.len):
                    idx = (s.tail + k) % MAX_BODY
                    p = s.body[idx]

                    if 0 <= p.y < game.h and 0 <= p.x < game.w:
                        grcurp.cell[p.y][p.x] |= SNAKE

            # --- score final
            for i in range(totalSnake):

                s = game.snakes[i]
                hx = s.body[s.head].x
                hy = s.body[s.head].y

                sc = score[i]

                # --- mort
                if not s.alive:
                    sc = self.params['death']
                    self.backprop(node[i], sc / 200.0)
                    continue

                # --- taille
                sc += s.len * self.params['size']

                # --- distance énergie
                bestDist = 1e9
                count = 0

                if 0 <= hx < game.w and 0 <= hy < game.h:
                    bestDist = 1e9
                    sumScore = 0

                    for ind in range(power_source_count):
                        e = game.energy[ind]

                        if game.grid.cell[e.y][e.x] & ENERGY:
                            gap, di = self.compute_max_gap(hx, hy, parentg[e.y * width + e.x], grcurp, game.w, game.h)

                            if gap > s.len - 2:
                                gap += 1000

                            d = di + gap

                            if d < bestDist:
                                bestDist = d

                            sumScore += self.params['dist'] / (distg[e.y * width + e.x][hy][hx] + 1)

                    # combo
                    if bestDist < 1e9:
                        sc += self.params['dist']*30.0 / (bestDist + 1)  # focus
                        sc += 0.3 * sumScore  # global attraction

                # --- plus d'énergie
                if count == 0:
                    if i < game.snakeCount:
                        sc += self.params['win'] if tot_team > tot_opp else self.params['lose']
                    else:
                        sc += self.params['win'] if tot_opp > tot_team else self.params['lose']

                # --- flood fill
                fl = self.flood_fill(grcurp,hx, hy, game.w, game.h, min(4, s.len))
                if fl < min(4, s.len) or self.get_safe_moves(hx, hy, grcurp, game.w, game.h) == 0 :
                    sc += self.params['flood']
                                                       
                self.backprop(node[i], sc / 200.0)

            turn += 1

        #print("TURN", turn)

        indc = []
        for i in range(my_snake_len):
            a, sc = self.best_move(root[i])
            indc.append(a)

                 

        return indc, sc

    def reconstruct_path(self, tx, ty, parent):
        path = []

        x, y = tx, ty

        while x != -1 and y != -1:
            path.append((x, y))
            px, py = parent[y][x]
            x, y = px, py

        path.reverse()
        return path

    def compute_max_gap(self, tx, ty, parent, grid, W, H):
        max_gap = 0
        current_gap = 0
        d = 0

        x, y = tx, ty

        while x != -1 and y != -1:
            d += 1
            ny = y + 1

            # --- check support dessous ---
            if ny >= H:
                no_support = True
            else:
                if not (grid.cell[ny][x] & (WALL | ENERGY | SNAKE)):
                    no_support = True
                else:
                    no_support = False

            # --- si on touche un snake ---
            if grid.cell[y][x] & SNAKE:
                d += 5
                

            # --- gestion gap ---
            if no_support:
                current_gap += 1
                if current_gap > max_gap:
                    max_gap = current_gap
            else:
                current_gap = 0  # reset si support

            # --- remonter le path ---
            px, py = parent[y][x]
            x, y = px, py

        return max_gap, d

    def expandBeam(self, grcurp, snake, wt, ht):
              
        h = snake.body[snake.head]  # headPos

        move = []
        # --- moves normaux
        for i in range(4):

            nx = h.x + DX[i]
            ny = h.y + DY[i]

            if nx < 0 or ny < 0 or nx >= wt or ny >= ht:
                continue

            if grcurp.cell[ny][nx] & WALL:
                continue

            if i == 0 and grcurp.cell[ny][nx] == EMPTY and (self.is_fully_vertical(snake)):
                continue
               
            curid = (snake.head - 1 + MAX_BODY) % MAX_BODY;
            cur = snake.body[curid];
            if nx == cur.x and ny == cur.y:
                continue

            if grcurp.cell[ny][nx] & ENERGY:
                move = [i]
                return move

            move.append(i)

        return move

    def expandBeamP(self, grcurp, snake, wt, ht):
              
        h = snake.body[snake.head]  # headPos

        move = []
        # --- moves normaux
        for i in range(4):

            nx = h.x + DX[i]
            ny = h.y + DY[i]

            if nx < 0 or ny < 0 or nx >= wt or ny >= ht:
                continue

            if grcurp.cell[ny][nx] & WALL:
                continue

            if i == 0 and grcurp.cell[ny][nx] == EMPTY and (self.is_fully_vertical(snake)):
                continue
               
            curid = (snake.head - 1 + MAX_BODY) % MAX_BODY;
            cur = snake.body[curid];
            if nx == cur.x and ny == cur.y:
                continue
                    
            move.append(i)

        return move

    def evaluation(self, ind_snake, game, score, totalSnake, power_source_count, distg, parentg, width, height, energy):

        # =====================
        #  EVALUATION
        # =====================
        tot_team = 0
        tot_opp = 0

        for i in range(totalSnake):
            s = game.snakes[i]
            if s.alive:
                if i < game.snakeCount:
                    tot_team += 1
                else:
                    tot_opp += 1

        # --- copy grid
        grcurp = game.grid.copy()

        # --- mark snakes
        for i in range(totalSnake):
            s = game.snakes[i]
            for k in range(s.len):
                idx = (s.tail + k) % MAX_BODY
                p = s.body[idx]

                if 0 <= p.y < game.h and 0 <= p.x < game.w:
                    grcurp.cell[p.y][p.x] |= SNAKE

        best_score = [0] * 2

        # --- score final
        i = ind_snake

        s = game.snakes[i]
        hx = s.body[s.head].x
        hy = s.body[s.head].y

        sc = score[i]

        # --- mort
        if not s.alive:
            sc = self.params['death']
                    
            return sc

        # --- taille
        sc += s.len * self.params['size']

        # --- distance énergie
        bestDist = 1e9
        count = 0

        if 0 <= hx < game.w and 0 <= hy < game.h:
            bestDist = 1e9
            sumScore = 0

            for ind in range(power_source_count):
                e = energy[ind]

                if game.grid.cell[e.y][e.x] & ENERGY:
                    gap, di = self.compute_max_gap(hx, hy, parentg[e.y * width + e.x], grcurp, game.w, game.h)

                    if gap > s.len - 2:
                        gap += 1000

                    d = di + gap

                    if d < bestDist:
                        bestDist = d

                    count += 1

                    sumScore += self.params['dist']*10 / (d + 1)

            # combo
            if bestDist < 1e9:
                sc += self.params['dist']*30.0 / (bestDist + 1)  # focus
                sc += 0.3 * sumScore  # global attraction

        # --- plus d'énergie
        if count == 0:
            if i < game.snakeCount:
                sc += self.params['win'] if tot_team > tot_opp else self.params['lose']
            else:
                sc += self.params['win'] if tot_opp > tot_team else self.params['lose']

        # --- flood fill
        fl = self.flood_fill(grcurp,hx, hy, game.w, game.h, min(4, s.len))
        if fl < min(4, s.len):
            sc += self.params['flood']

        return sc


    def BS(self, width, height, my_snake, my_snake_len,
             opp_snake, opp_snake_len,
             energy, power_source_count, grid, distg, parentg, time_limit_ms, model=None):

        start = time.perf_counter()

        self.nodeCount = 0
        self.childCount = 0

        def getTime():
            return (time.perf_counter() - start) * 1000 <= time_limit_ms

        totalSnake = my_snake_len + opp_snake_len
     
        best_move   = [5] * (my_snake_len+opp_snake_len)
        best_score_f  = [-float('inf')] * 2

        WIDTHB = 60  # ou ta valeur

        beam = [[0 for _ in range(WIDTHB)] for _ in range(8)]
        maxBeam = 0
        maxTS = [1] * 8

        TS = [[0 for _ in range(WIDTHB)] for _ in range(8)]

        for player in range(my_snake_len):
            beam[player][0] = self.createNode(-1)
            child = beam[player][0]
            self.nodes[child].game = GameState(width, height)
            game = self.nodes[child].game
            game.snakeCount = my_snake_len
            #print(type(my_snake))
            # --- my snakes
            for i in range(my_snake_len):
                game.snakes[i] = my_snake[i].copy()  # ⚠️ important
                game.snakes[i].dir = UP
                game.snakes[i].alive = True

            # --- opp snakes
            for i in range(opp_snake_len):
                game.snakes[my_snake_len + i] = opp_snake[i].copy()
                game.snakes[my_snake_len + i].dir = UP
                game.snakes[my_snake_len + i].alive = True

            # --- energy
            for i in range(power_source_count):
                game.energy[i] = energy[i].copy()

            game.grid = grid.copy()
            game.energyCount = power_source_count

       
        DEPTH = 10
        turn = 0
        depth = -1
        #for depth in range(10):
        while getTime():
            depth +=1
            turn+=1
          
            for player in range(my_snake_len):
        
                ind_beam = 0
                count = 0
                while ind_beam < maxTS[player]:

                    node = self.nodes[beam[player][ind_beam]]
                    game = node.game

                    move = self.expandBeamP(game.grid, game.snakes[player], game.w, game.h)
                    
                    if not game.snakes[player].alive or len(move) == 0:
                        ind_beam += 1
                        continue


                    for iplayer in range(totalSnake):
                        if iplayer == player:
                            continue

                        if not game.snakes[iplayer].alive:
                            continue

                        moveo = self.expandBeam(game.grid, game.snakes[iplayer], game.w, game.h)

                        dir = -1
                        if len(moveo) > 0:
                            dir = moveo[random.randint(0, len(moveo)-1)]
                        else:
                            dir = self.getDirection(game.snakes[iplayer])

            
                        game.snakes[iplayer].dir = dir

                   
                    
                    for m in move:
                        h = game.snakes[player].body[game.snakes[player].head]
                        child = self.createNode(beam[player][ind_beam])
                        self.addChild(beam[player][ind_beam], child)

                        hx = h.x + DX[m]
                        hy = h.y + DY[m]

                                                
                        if depth == 0:
                            self.nodes[child].move = m
                        else:
                            self.nodes[child].move = node.move

                        if game.grid.cell[hy][hx] & ENERGY:
                            self.nodes[child].mult = self.params['eat']

                        self.nodes[child].game = GameState(width, height)
                        gamech = self.nodes[child].game
                        
                        gamech.snakeCount = my_snake_len
                        #print(type(my_snake))
                        # --- my snakes
                        for i in range(my_snake_len):
                            gamech.snakes[i] = game.snakes[i].copy()  # ⚠️ important
                
                        gamech.snakes[player].dir = m
                        # --- opp snakes
                        for i in range(opp_snake_len):
                            gamech.snakes[my_snake_len + i] = game.snakes[my_snake_len + i].copy()
    
                        # --- energy
                        #for i in range(power_source_count):
                        #    gamech.energy[i] = energy[i].copy()

                        gamech.grid = game.grid.copy()
                        gamech.energyCount = power_source_count

                        score = [0.0] * 8
                        self.playMoveTurn(gamech,score, opp_snake_len)

                        sc = self.evaluation(player, gamech, score, totalSnake, power_source_count, distg, parentg, width, height, energy)

                        self.nodes[child].score = node.score + sc * self.nodes[child].mult

                        TS[player][count] = child

                        count += 1
                        if count == WIDTHB:
                            break

                    if count == WIDTHB:
                        break

                    ind_beam += 1

                maxTS[player] = count

                TS[player][:count] = sorted(
                    TS[player][:count],
                    key=lambda a: self.nodes[a].score,
                    reverse=True
                )

                beam[player][:maxTS[player]] = TS[player][:maxTS[player]]

          

        print('turn=', turn)
        #print(["UP","DOWN","LEFT","RIGHT"][best_move[0]])
        
        node = self.nodes[beam[player][0]]
        print("nodes=", self.nodeCount, "score=", node.score)
        return node.move, node.score


#---------------EN SIMULATION

MAX_BODY = 256
# ===== SNAKE =====
class Snake:
    def __init__(self, body, owner=0):
        self.body = body[:]  # list of (x,y)
        self.alive = True
        self.owner = owner
        self.last = body[0]

    def head(self):
        return self.body[0]

    def copy(self):
        new_snake = Snake(self.body[:], self.owner)
        new_snake.alive = self.alive
        new_snake.last = self.last
        return new_snake

class SnakeS:
    MAX_BODY = 256  # ou la valeur que tu utilises en C++

    def __init__(self, id=0):
        self.id = id
        self.alive = True
        self.dir = 0  # direction par défaut

        self.head = 0  # index de la tête
        self.tail = 0  # index de la queue
        self.len = 0   # longueur du serpent

        # liste de positions (x, y) avec une taille max
        self.body = [None] * self.MAX_BODY

    def alive_body_iter(self):
        """Itère sur les positions vivantes du serpent dans l'ordre tail->head."""
        for k in range(self.len):
            idx = (self.tail + k) % self.MAX_BODY
            pos = self.body[idx]
            if pos is not None:
                yield pos

    def copy(self):
        new_snake = SnakeS(self.id)
        new_snake.alive = self.alive
        new_snake.dir = self.dir
        new_snake.head = self.head
        new_snake.tail = self.tail
        new_snake.len = self.len

        new_snake.body = [None] * self.MAX_BODY

        for k in range(self.len):
            idx = (self.tail + k) % self.MAX_BODY
            p = self.body[idx]
            new_snake.body[idx] = Pos(p.x, p.y)

        return new_snake

# ===== GAME =====
class Game:
    def __init__(self):
        self.W = random.randint(20, 45)
        self.H = int(self.W * (30/45));
        self.NBENERGY = (self.W*self.H) // 20#random.randint(10, 45)
        self.dist = None  # distance map
        self.grid = [[EMPTY for _ in range(self.W)] for _ in range(self.H)]
        self.snakes = []
        self.energy = []
        self.idx_energy = []
        #gen = MapGenerator(self.W, self.H, seed=None)
        #self.grid, self.energy = gen.generate()
        #gen = MapGen(1)
        #self.grid, self.energy, self.W, self.H = gen.make()
        
        self.generate_map()
        for i in range(len(self.energy)):
            self.idx_energy.append(i)
        self.spawn_snakes()
        #self.compute_distance_map()
        #for d in self.dist:
        #    print(d)
       
    def init_Smitsimax(self):
        self.simulation = SM()

    def init_Training_ValueNet(self):
        self.states = []
        self.targets = []

    def init_Training_PUCT(self):
        self.states = []
        self.policy = []
        
    def clone(self):
        import copy
        return copy.deepcopy(self)

    def compute_distance_map(self):
        # init distances
        dist = [[INF for _ in range(self.W)] for _ in range(self.H)]
        pq = []

        # multi-source: apples
        for (x, y) in self.energy:
            dist[y][x] = 0
            heapq.heappush(pq, (0, x, y))

        while pq:
            d, x, y = heapq.heappop(pq)

            if d != dist[y][x]:
                continue

            for i, (dx, dy) in enumerate(DIRS):
                nx, ny = x + dx, y + dy

                if nx < 0 or ny < 0 or nx >= self.W or ny >= self.H:
                    continue

                if self.grid[ny][nx] == WALL:
                    continue

                nd = d + move_cost(i)

                if nd < dist[ny][nx]:
                    dist[ny][nx] = nd
                    heapq.heappush(pq, (nd, nx, ny))

        self.dist = dist
        #for d in self.dist:
        #    print(d)

    def compute_distance_mapg(self):
        # init distances
        
        dm = [None] * (self.W * self.H)
        # multi-source: apples
        for (x, y) in self.energy:
            dist = [[INF for _ in range(self.W)] for _ in range(self.H)]
            pq = []
            dist[y][x] = 0
            heapq.heappush(pq, (0, x, y))

            while pq:
                d, x, y = heapq.heappop(pq)

                if d != dist[y][x]:
                    continue

                for i, (dx, dy) in enumerate(DIRS):
                    nx, ny = x + dx, y + dy

                    if nx < 0 or ny < 0 or nx >= self.W or ny >= self.H:
                        continue

                    if self.grid[ny][nx] == WALL:
                        continue

                    nd = d + move_cost(i)

                    if nd < dist[ny][nx]:
                        dist[ny][nx] = nd
                        heapq.heappush(pq, (nd, nx, ny))

        
            dm[y*self.W+x] = dist

        return dm

    def compute_distance_mapgstest(self):
        # init distances
        
        dm = [None] * (self.W * (self.H+BORDERH))
        # multi-source: apples
        for (x, y) in self.energy:
            y += BORDERH
            dist = [[INF for _ in range(self.W)] for _ in range(self.H+BORDERH)]
            pq = []
            dist[y][x] = 0
            heapq.heappush(pq, (0, x, y))

            while pq:
                d, x, y = heapq.heappop(pq)

                if d != dist[y][x]:
                    continue

                for i, (dx, dy) in enumerate(DIRS):
                    nx, ny = x + dx, y + dy

                    if nx < 0 or ny < 0 or nx >= self.W or ny >= self.H:
                        continue

                    if self.grid[ny][nx] == WALL:
                        continue

                    nd = d + move_cost(i)

                    if nd < dist[ny][nx]:
                        dist[ny][nx] = nd
                        heapq.heappush(pq, (nd, nx, ny))

        
            dm[y*self.W+x] = dist

        return dm

    def compute_distance_mapgs(self):
        dm = [None] * (self.W * (self.H + BORDERH))
    
        for (x, y) in self.energy:
            # ne fais pas y += BORDERH ici
            dist = [[INF for _ in range(self.W)] for _ in range(self.H + BORDERH)]
            pq = []
            dist[y + BORDERH][x] = 0  # décale dans dist seulement
            heapq.heappush(pq, (0, x, y + BORDERH))

            while pq:
                d, x0, y0 = heapq.heappop(pq)

                if d != dist[y0][x0]:
                    continue

                for i, (dx, dy) in enumerate(DIRS):
                    nx, ny = x0 + dx, y0 + dy

                    if nx < 0 or ny < 0 or nx >= self.W or ny >= self.H + BORDERH:
                        continue

                    #if ny < BORDERH:  # ligne de bord haute, consider comme WALL
                    #    continue

                    if self.grid[ny - BORDERH][nx] & (WALL):  # map vers grid réel
                        continue

                    nd = d + move_cost(i)

                    if nd < dist[ny][nx]:
                        dist[ny][nx] = nd
                        heapq.heappush(pq, (nd, nx, ny))

            dm[(y + BORDERH) * self.W + x] = dist

        return dm

    def compute_distance_mapgsp(self):
        dm = [None] * (self.W * (self.H + BORDERH))
        parentg = [None] * (self.W * (self.H + BORDERH))
    
        for (x, y) in self.energy:
            # ne fais pas y += BORDERH ici
            dist = [[INF for _ in range(self.W)] for _ in range(self.H + BORDERH)]
            parent = [[(-1, -1) for _ in range(self.W)] for _ in range(self.H + BORDERH)]
            pq = []
            dist[y + BORDERH][x] = 0  # décale dans dist seulement
            heapq.heappush(pq, (0, x, y + BORDERH))

            while pq:
                d, x0, y0 = heapq.heappop(pq)

                if d != dist[y0][x0]:
                    continue

                for i, (dx, dy) in enumerate(DIRS):
                    nx, ny = x0 + dx, y0 + dy

                    if nx < 0 or ny < 0 or nx >= self.W or ny >= self.H + BORDERH:
                        continue

                    #if ny < BORDERH:  # ligne de bord haute, consider comme WALL
                    #    continue

                    if self.grid[ny - BORDERH][nx] & (WALL | ENERGY):  # map vers grid réel
                        continue

                    nd = d + move_cost(i)

                    if nd < dist[ny][nx]:
                        dist[ny][nx] = nd
                        parent[ny][nx] = (x0, y0)
                        heapq.heappush(pq, (nd, nx, ny))

            dm[(y + BORDERH) * self.W + x] = dist
            parentg[(y + BORDERH) * self.W + x] = parent


        return dm, parentg

    def compute_distance_map2(self):
            dist = [[INF for _ in range(self.W)] for _ in range(self.H)]
            dq = deque()

            # multi-source : apples
            for (x, y) in self.energy:
                if 0 <= x < self.W and 0 <= y < self.H:
                    dist[y][x] = 0
                    dq.append((x, y))

            while dq:
                x, y = dq.popleft()
                d = dist[y][x]

                for i, (dx, dy) in enumerate(DIRS):
                    nx, ny = x + dx, y + dy

                    if nx < 0 or ny < 0 or nx >= self.W or ny >= self.H:
                        continue

                    if self.grid[ny][nx] == WALL:
                        continue

                    # coût
                    if i == 0:  # UP → coût 2
                        nd = d + 2
                    else:
                        nd = d + 1

                    if nd < dist[ny][nx]:
                        dist[ny][nx] = nd

                        # astuce BFS
                        if i == 0:
                            dq.append((nx, ny))      # plus lent
                        else:
                            dq.appendleft((nx, ny))  # plus rapide

            self.dist = dist
            self.dist_computed = True

    # ---------- SYMMETRIC MAP ----------
    def generate_map(self):

        for x in range(self.W):
            self.grid[self.H-1][x] = WALL

        # walls sym�triques
        for y in range(self.H):
            for x in range(self.W//2):
                if random.random() < 0.1:
                    self.grid[y][x] = WALL
                    self.grid[y][self.W-1-x] = WALL

        IDX = 0
        # energy sym�trique
        for _ in range(self.NBENERGY):
            x = random.randint(0, self.W//2 - 1)
            y = random.randint(0, self.H-1)

            if self.grid[y][x] == EMPTY and self.grid[y][self.W-1-x] == EMPTY and (x, y) not in self.energy and (self.W-1-x, y) not in self.energy:
                self.grid[y][x] = ENERGY
                self.grid[y][self.W-1-x] = ENERGY
                self.energy.append((x, y))
                self.energy.append((self.W-1-x, y))
                self.idx_energy.append(IDX)
                IDX+=1
                self.idx_energy.append(IDX)
                IDX+=1

    # ---------- SYMMETRIC SNAKES ----------
    def spawn_snakes2(self):

        # snake joueur
        x = random.randint(2, self.W//2 - 3)
        y = random.randint(2, self.H-3)

        my_body = [(x,y),(x,y+1),(x,y+2)]
 
        # miroir
        opp_body = [(self.W-1-x,y),(self.W-1-x,y+1),(self.W-1-x,y+2)]

        self.snakes = [
            Snake(my_body),
            Snake(opp_body)
        ]

    def spawn_snakes(self):
   
        def is_free(x, y):
            # vérifier que la case est vide
            return self.grid[y][x] == EMPTY and (x, y) not in self.energy

        # snake joueur
        while True:
            x = random.randint(2, self.W//2 - 3)
            y = random.randint(2, self.H - 3)
            my_body = [(x, y), (x, y+1), (x, y+2)]
            if all(is_free(px, py) for px, py in my_body):
                break  # on a trouvé un emplacement libre

        # retirer l'énergie si elle est sur le spawn
        self.energy = [(ex, ey) for ex, ey in self.energy if (ex, ey) not in my_body]

        # serpent miroir pour l’adversaire
        opp_body = [(self.W - 1 - px, py) for px, py in my_body]

        # ajouter les serpents
        self.snakes = [
            Snake(my_body),
            Snake(opp_body)
        ]

    # ---------- MOVE ----------
    def move_snake(self, snake, d):
        if not snake.alive:
            return

        dx, dy = DIRS[d]
        hx, hy = snake.head()
        nx, ny = hx + dx, hy + dy

        # collision avec le mur
        if nx >= 0 and ny >= 0 and nx < self.W and ny < self.H and self.grid[ny][nx] == WALL:
            # si snake a moins de 3 segments mort
            if len(snake.body) < 3:
                snake.alive = False
                return
            # sinon la deuxi�me partie devient la nouvelle t�te
            snake.body = snake.body[1:]
                        
            return

        # collision avec un autre snake
        occupied = any((nx, ny) in s.body for s in self.snakes if s.alive)
        if occupied:
            if len(snake.body) < 3:
                snake.alive = False
                return

            snake.body = snake.body[1:]

            return

        # d�placer la t�te normalement
        snake.body.insert(0, (nx, ny))

        # manger �nergie
        if nx >= 0 and ny >= 0 and nx < self.W and ny < self.H and self.grid[ny][nx] == ENERGY:
            self.grid[ny][nx] = EMPTY       # retirer de la grille
            # retirer de la liste energies
            if (nx, ny) in self.energy:
                idx = self.energy.index((nx, ny))
                self.energy.pop(idx)
                self.idx_energy.pop(idx)
            # ne pas retirer la queue le snake grandit
        else:
            snake.body.pop()

    # ---------- GRAVITY ----------
    def apply_gravity(self):
        for snake in self.snakes:
            if not snake.alive:
                continue

            # calculer de combien le snake peut tomber
            min_fall = self.H  # commencer avec une chute max possible

            for (x, y) in snake.body:
                fall = 0
                while True:
                    by = y + fall + 1
                    if by >= self.H:
                        break  # atteint le sol
                    if 0 <= x < self.W and 0 <= by < self.H and self.grid[by][x] == WALL:  # obstacle solide
                        break
                    if any((x, by) in s.body for s in self.snakes if s != snake and s.alive):
                        break
                    if 0 <= x < self.W and 0 <= by < self.H and self.grid[by][x] == ENERGY:  # obstacle solide
                        break
                    fall += 1
                min_fall = min(min_fall, fall)  # choisir la plus petite chute

            # appliquer la chute minimale � tout le snake
            if min_fall > 0:
                snake.body = [(x, y + min_fall) for (x, y) in snake.body]

    # ---------- STEP ----------
    def step(self, moves):

        # moves = [dir_me, dir_opp]

        for i, snake in enumerate(self.snakes):
            self.move_snake(snake, moves[i])

        self.apply_gravity()

    def step_IA(self, agent):

        my_snake = self.snakes[0]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy

        
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self.clone(), 0, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            hx, hy = self.snakes[0].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[0].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            action = q_values.argmax().item()

        my_snake.last = my_snake.head()
        self.move_snake(my_snake, action)
        
        #print("action=", action)
        #if (self.snakes[0].head() in self.energy):
        #    self.energy.remove(self.snakes[0].head())
                           

        score, opp_move = minimax(self.clone(), 3, 1, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[1], opp_move)
        #if (self.snakes[1].head() in self.energy):
        #    self.energy.remove(self.snakes[1].head())



        self.apply_gravity()


    
    def step_IAOnlyFan(self, agent, agent2):

        my_snake = self.snakes[0]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy

        
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self, 0, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            
            hx, hy = self.snakes[0].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[0].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False
            
            if is_vertical(self.snakes[0]) and 0<=hx<self.W and 0<=hy-1<self.H and self.grid[hy-1][hx] == EMPTY:
                valid_actions[0] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            action = q_values.argmax().item()

        my_snake.last = my_snake.head()
        self.move_snake(my_snake, action)
        
        #print("action=", action)
        #if (self.snakes[0].head() in self.energy):
        #    self.energy.remove(self.snakes[0].head())
                           
        
        #-------
        my_snake = self.snakes[1]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy
                
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent2.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self, 1, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            hx, hy = self.snakes[1].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[1].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False

            if is_vertical(self.snakes[1]) and 0<=hx<self.W and 0<=hy-1<self.H and self.grid[hy-1][hx] == EMPTY:
                valid_actions[0] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            opp_action = q_values.argmax().item()


        #score, opp_move = minimax(self.clone(), 3, 1, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[1], opp_action)
        #if (self.snakes[1].head() in self.energy):
        #    self.energy.remove(self.snakes[1].head())



        self.apply_gravity()

    def step_IAOnlyFanMiniMax(self, agent):
                           
        score, action = minimax(self.clone(), 3, 0, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[0], action)

        #-------
        my_snake = self.snakes[1]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy

        
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self, 1, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            hx, hy = self.snakes[1].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[1].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            opp_action = q_values.argmax().item()


        #score, opp_move = minimax(self.clone(), 3, 1, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[1], opp_action)
        #if (self.snakes[1].head() in self.energy):
        #    self.energy.remove(self.snakes[1].head())



        self.apply_gravity()

    def step_IAOnlyFanTraining(self, agent, action):

        self.move_snake(self.snakes[0], action)
        
        #print("action=", action)
        #if (self.snakes[0].head() in self.energy):
        #    self.energy.remove(self.snakes[0].head())
                           
        
        #-------
        my_snake = self.snakes[1]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy

        
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self, 1, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            hx, hy = self.snakes[1].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[1].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            opp_action = q_values.argmax().item()


        #score, opp_move = minimax(self.clone(), 3, 1, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[1], opp_action)
        #if (self.snakes[1].head() in self.energy):
        #    self.energy.remove(self.snakes[1].head())



        self.apply_gravity()

    def step_IAOnlyFanSmitsimax(self, agent):
        
        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        my_snake = [snakeme]

        snakeme = SnakeS(1)  # crée un nouveau serpent avec id=0
        body = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[1].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0
        opp_snake = [snakeme]

        energy = []
        for ex, ey in self.energy:
            energy.append(Pos(ex, ey+BORDERH))

        distg, parentg = self.compute_distance_mapgsp()
        #print(self.compute_distance_mapg())
        action, score = self.simulation.play(self.W, self.H+BORDERH, my_snake, 1, opp_snake, 1, energy, len(energy), ngrid, distg, parentg, 100)
        self.move_snake(self.snakes[0], action[0])
        #print(["UP", "DOWN", "LEFT", "RIGHT"][action[0]])

        #-------
        my_snake = self.snakes[1]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy

        
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self, 1, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            hx, hy = self.snakes[1].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[1].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            opp_action = q_values.argmax().item()


        #score, opp_move = minimax(self.clone(), 3, 1, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[1], opp_action)
        #if (self.snakes[1].head() in self.energy):
        #    self.energy.remove(self.snakes[1].head())

        #gravity--------
        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        snakeme2 = SnakeS(1)  # crée un nouveau serpent avec id=0
        body = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme2.body = list(sbody)
        snakeme2.len = len(self.snakes[1].body)
        snakeme2.head = snakeme2.len - 1
        snakeme2.tail = 0

        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        game = GameState(self.W, self.H+BORDERH)
        game.snakeCount = 1

        # --- my snakes
        for i in range(1):
            game.snakes[i] = snakeme # ⚠️ important
            game.snakes[i].dir = UP
            game.snakes[i].alive = True

        # --- opp snakes
        for i in range(1):
            game.snakes[1 + i] = snakeme2
            game.snakes[1 + i].dir = UP
            game.snakes[1 + i].alive = True

        # --- energy
        #for i in range(power_source_count):
        #    game.energy[i] = energy[i].copy()

        game.grid = ngrid
        #game.energyCount = power_source_count

        self.simulation.doFalls(game, 2)
        #self.simulation.doIntercoiledFalls(game, 2)

        self.snakes[0].body = []
        for k in range(game.snakes[0].len):
            id_ = (game.snakes[0].head - k + MAX_BODY) % MAX_BODY
            x, y = game.snakes[0].body[id_].x, game.snakes[0].body[id_].y
            self.snakes[0].body.append((x, y-BORDERH))

        self.snakes[1].body = []
        for k in range(game.snakes[1].len):
            id_ = (game.snakes[1].head - k + MAX_BODY) % MAX_BODY
            x, y = game.snakes[1].body[id_].x, game.snakes[1].body[id_].y
            self.snakes[1].body.append((x, y-BORDERH))


        #self.apply_gravity()

        return score


    def step_IAOnlyFanSmitsimaxValueNet(self, agent, model):
        
        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        #sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            snakeme.body[i] = Pos(bx, by+BORDERH)
        #snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        my_snake = [snakeme]

        snakeme2 = SnakeS(1)  # crée un nouveau serpent avec id=0
        body = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        #sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            snakeme2.body[i] = Pos(bx, by+BORDERH)
        #snakeme2.body = list(sbody)
        snakeme2.len = len(self.snakes[1].body)
        snakeme2.head = snakeme2.len - 1
        snakeme2.tail = 0
        opp_snake = [snakeme2]

        energy = []
        for ex, ey in self.energy:
            energy.append(Pos(ex, ey+BORDERH))

        
        #print(self.compute_distance_mapg())
        action, score = self.simulation.playVN(self.W, self.H+BORDERH, my_snake, 1, opp_snake, 1, energy, len(energy), ngrid, self.compute_distance_mapgs(), 50, model)
        self.move_snake(self.snakes[0], action[0])
        #print(["UP", "DOWN", "LEFT", "RIGHT"][action[0]])

        #-------
        my_snake = self.snakes[1]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy

        
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self, 1, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            hx, hy = self.snakes[1].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[1].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            opp_action = q_values.argmax().item()


        #score, opp_move = minimax(self.clone(), 3, 1, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[1], opp_action)
        #if (self.snakes[1].head() in self.energy):
        #    self.energy.remove(self.snakes[1].head())

        #gravity--------
        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        snakeme2 = SnakeS(1)  # crée un nouveau serpent avec id=0
        body = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme2.body = list(sbody)
        snakeme2.len = len(self.snakes[1].body)
        snakeme2.head = snakeme2.len - 1
        snakeme2.tail = 0

        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        game = GameState(self.W, self.H+BORDERH)
        game.snakeCount = 1

        # --- my snakes
        for i in range(1):
            game.snakes[i] = snakeme.copy() # ⚠️ important
            game.snakes[i].dir = UP
            game.snakes[i].alive = True

        # --- opp snakes
        for i in range(1):
            game.snakes[1 + i] = snakeme2.copy()
            game.snakes[1 + i].dir = UP
            game.snakes[1 + i].alive = True

        # --- energy
        #for i in range(power_source_count):
        #    game.energy[i] = energy[i].copy()

        game.grid = ngrid
        #game.energyCount = power_source_count

        self.simulation.doFalls(game, 2)
        #self.simulation.doIntercoiledFalls(game, 2)

        self.snakes[0].body = []
        for k in range(game.snakes[0].len):
            id_ = (game.snakes[0].head - k + MAX_BODY) % MAX_BODY
            x, y = game.snakes[0].body[id_].x, game.snakes[0].body[id_].y
            self.snakes[0].body.append((x, y-BORDERH))

        self.snakes[1].body = []
        for k in range(game.snakes[1].len):
            id_ = (game.snakes[1].head - k + MAX_BODY) % MAX_BODY
            x, y = game.snakes[1].body[id_].x, game.snakes[1].body[id_].y
            self.snakes[1].body.append((x, y-BORDERH))


        #self.apply_gravity()

        return score



    def step_IAOnlyFanSmitsimaxTraining(self, agent):
        
        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * 256
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        my_snake = [snakeme]

        snakeme = SnakeS(1)  # crée un nouveau serpent avec id=0
        body2 = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody2 = [None] * 256
        for i, (bx, by) in enumerate(body2):
            sbody2[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody2)
        snakeme.len = len(self.snakes[1].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0
        opp_snake = [snakeme]

        energy = []
        for ex, ey in self.energy:
            energy.append(Pos(ex, ey+BORDERH))

        
        #print(self.compute_distance_mapg())
        action, score = self.simulation.playVNTraining(self.W, self.H+BORDERH, my_snake, 1, opp_snake, 1, energy, len(energy), ngrid, self.compute_distance_mapgs(), 50, self)
        self.move_snake(self.snakes[0], action[0])
        #print(["UP", "DOWN", "LEFT", "RIGHT"][action[0]])

      
        #-------
        my_snake = self.snakes[1]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy

        
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self, 1, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            hx, hy = self.snakes[1].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[1].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            opp_action = q_values.argmax().item()


        #score, opp_move = minimax(self.clone(), 3, 1, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[1], opp_action)
        #if (self.snakes[1].head() in self.energy):
        #    self.energy.remove(self.snakes[1].head())

        #gravity--------
        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * 256
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        snakeme2 = SnakeS(1)  # crée un nouveau serpent avec id=0
        body = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * 256
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme2.body = list(sbody)
        snakeme2.len = len(self.snakes[1].body)
        snakeme2.head = snakeme2.len - 1
        snakeme2.tail = 0

        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        game = GameState(self.W, self.H+BORDERH)
        game.snakeCount = 1

        # --- my snakes
        for i in range(1):
            game.snakes[i] = snakeme # ⚠️ important
            game.snakes[i].dir = UP
            game.snakes[i].alive = True

        # --- opp snakes
        for i in range(1):
            game.snakes[1 + i] = snakeme2
            game.snakes[1 + i].dir = UP
            game.snakes[1 + i].alive = True

        # --- energy
        #for i in range(power_source_count):
        #    game.energy[i] = energy[i].copy()

        game.grid = ngrid
        #game.energyCount = power_source_count

        self.simulation.doFalls(game, 2)
        #self.simulation.doIntercoiledFalls(game, 2)

        self.snakes[0].body = []
        for k in range(game.snakes[0].len):
            id_ = (game.snakes[0].head - k + 256) % 256
            x, y = game.snakes[0].body[id_].x, game.snakes[0].body[id_].y
            self.snakes[0].body.append((x, y-BORDERH))

        self.snakes[1].body = []
        for k in range(game.snakes[1].len):
            id_ = (game.snakes[1].head - k + 256) % 256
            x, y = game.snakes[1].body[id_].x, game.snakes[1].body[id_].y
            self.snakes[1].body.append((x, y-BORDERH))


        #self.apply_gravity()

        return score

    
    def step_IAOnlyFanSmitsimaxTrainingPUCT(self, agent):
        
        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * 256
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        my_snake = [snakeme]

        snakeme = SnakeS(1)  # crée un nouveau serpent avec id=0
        body2 = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody2 = [None] * 256
        for i, (bx, by) in enumerate(body2):
            sbody2[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody2)
        snakeme.len = len(self.snakes[1].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0
        opp_snake = [snakeme]

        energy = []
        for ex, ey in self.energy:
            energy.append(Pos(ex, ey+BORDERH))

    
        my_snake_g = self.snakes[0].copy()
        opp_snake_g = self.snakes[1].copy()

        for i in range(len(my_snake_g.body)):
            x, y = my_snake_g.body[i]
            my_snake_g.body[i] = (x, y + BORDERH)

        for i in range(len(opp_snake_g.body)):
            x, y = opp_snake_g.body[i]
            opp_snake_g.body[i] = (x, y + BORDERH)

        energystate = []
        for ex, ey in self.energy:
            energystate.append((ex, ey+BORDERH))

        
        state = encode_state_full(my_snake_g, energystate, [opp_snake_g], self.W, self.H, 256)
         
        distg, parentg = self.compute_distance_mapgsp()
        
        #print(self.compute_distance_mapg())
        action, score, policy_val= self.simulation.playTrainingPUCT(self.W, self.H+BORDERH, my_snake, 1, opp_snake, 1, energy, len(energy), ngrid, distg, parentg, 50, self)
        self.move_snake(self.snakes[0], action[0])
        #print(["UP", "DOWN", "LEFT", "RIGHT"][action[0]])

        self.states.append(state)
        self.policy.append(policy_val)

      
        #-------
        my_snake = self.snakes[1]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy

        
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self, 1, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            hx, hy = self.snakes[1].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[1].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            opp_action = q_values.argmax().item()


        #score, opp_move = minimax(self.clone(), 3, 1, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[1], opp_action)
        #if (self.snakes[1].head() in self.energy):
        #    self.energy.remove(self.snakes[1].head())

        #gravity--------
        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * 256
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        snakeme2 = SnakeS(1)  # crée un nouveau serpent avec id=0
        body = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * 256
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme2.body = list(sbody)
        snakeme2.len = len(self.snakes[1].body)
        snakeme2.head = snakeme2.len - 1
        snakeme2.tail = 0

        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        game = GameState(self.W, self.H+BORDERH)
        game.snakeCount = 1

        # --- my snakes
        for i in range(1):
            game.snakes[i] = snakeme # ⚠️ important
            game.snakes[i].dir = UP
            game.snakes[i].alive = True

        # --- opp snakes
        for i in range(1):
            game.snakes[1 + i] = snakeme2
            game.snakes[1 + i].dir = UP
            game.snakes[1 + i].alive = True

        # --- energy
        #for i in range(power_source_count):
        #    game.energy[i] = energy[i].copy()

        game.grid = ngrid
        #game.energyCount = power_source_count

        self.simulation.doFalls(game, 2)
        #self.simulation.doIntercoiledFalls(game, 2)

        self.snakes[0].body = []
        for k in range(game.snakes[0].len):
            id_ = (game.snakes[0].head - k + 256) % 256
            x, y = game.snakes[0].body[id_].x, game.snakes[0].body[id_].y
            self.snakes[0].body.append((x, y-BORDERH))

        self.snakes[1].body = []
        for k in range(game.snakes[1].len):
            id_ = (game.snakes[1].head - k + 256) % 256
            x, y = game.snakes[1].body[id_].x, game.snakes[1].body[id_].y
            self.snakes[1].body.append((x, y-BORDERH))


        #self.apply_gravity()

        return score[0]

    
    def step_IAOnlyFanSmitsimaxPUCT(self, agent, model):
        
        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        #sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            snakeme.body[i] = Pos(bx, by+BORDERH)
        #snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        my_snake = [snakeme]

        snakeme2 = SnakeS(1)  # crée un nouveau serpent avec id=0
        body = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        #sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            snakeme2.body[i] = Pos(bx, by+BORDERH)
        #snakeme2.body = list(sbody)
        snakeme2.len = len(self.snakes[1].body)
        snakeme2.head = snakeme2.len - 1
        snakeme2.tail = 0
        opp_snake = [snakeme2]

        energy = []
        for ex, ey in self.energy:
            energy.append(Pos(ex, ey+BORDERH))

        distg, parentg = self.compute_distance_mapgsp()
        #print(self.compute_distance_mapg())
        action, score = self.simulation.playPUCT(self.W, self.H+BORDERH, my_snake, 1, opp_snake, 1, energy, len(energy), ngrid, distg, parentg, 50, model)
        self.move_snake(self.snakes[0], action[0])
        #print(["UP", "DOWN", "LEFT", "RIGHT"][action[0]])

        #-------
        my_snake = self.snakes[1]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy

        
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self, 1, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            hx, hy = self.snakes[1].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[1].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            opp_action = q_values.argmax().item()


        #score, opp_move = minimax(self.clone(), 3, 1, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[1], opp_action)
        #if (self.snakes[1].head() in self.energy):
        #    self.energy.remove(self.snakes[1].head())

        #gravity--------
        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        snakeme2 = SnakeS(1)  # crée un nouveau serpent avec id=0
        body = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme2.body = list(sbody)
        snakeme2.len = len(self.snakes[1].body)
        snakeme2.head = snakeme2.len - 1
        snakeme2.tail = 0

        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        game = GameState(self.W, self.H+BORDERH)
        game.snakeCount = 1

        # --- my snakes
        for i in range(1):
            game.snakes[i] = snakeme.copy() # ⚠️ important
            game.snakes[i].dir = UP
            game.snakes[i].alive = True

        # --- opp snakes
        for i in range(1):
            game.snakes[1 + i] = snakeme2.copy()
            game.snakes[1 + i].dir = UP
            game.snakes[1 + i].alive = True

        # --- energy
        #for i in range(power_source_count):
        #    game.energy[i] = energy[i].copy()

        game.grid = ngrid
        #game.energyCount = power_source_count

        self.simulation.doFalls(game, 2)
        #self.simulation.doIntercoiledFalls(game, 2)

        self.snakes[0].body = []
        for k in range(game.snakes[0].len):
            id_ = (game.snakes[0].head - k + MAX_BODY) % MAX_BODY
            x, y = game.snakes[0].body[id_].x, game.snakes[0].body[id_].y
            self.snakes[0].body.append((x, y-BORDERH))

        self.snakes[1].body = []
        for k in range(game.snakes[1].len):
            id_ = (game.snakes[1].head - k + MAX_BODY) % MAX_BODY
            x, y = game.snakes[1].body[id_].x, game.snakes[1].body[id_].y
            self.snakes[1].body.append((x, y-BORDERH))


        #self.apply_gravity()

        return score

    def step_IAOnlyFanBS(self, agent):
        
        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        my_snake = [snakeme]

        snakeme = SnakeS(1)  # crée un nouveau serpent avec id=0
        body = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[1].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0
        opp_snake = [snakeme]

        energy = []
        for ex, ey in self.energy:
            energy.append(Pos(ex, ey+BORDERH))

        distg, parentg = self.compute_distance_mapgsp()
        #print(self.compute_distance_mapg())
        action, score = self.simulation.BS(self.W, self.H+BORDERH, my_snake, 1, opp_snake, 1, energy, len(energy), ngrid, distg, parentg, 50)
        self.move_snake(self.snakes[0], action)
        #print(["UP", "DOWN", "LEFT", "RIGHT"][action[0]])

        #-------
        my_snake = self.snakes[1]
        enemies = [s for s in self.snakes if s != my_snake and s.alive]
        energies = self.energy

        
        #hx, hy = my_snake.head()
        #space = flood_fill_limited(hx, hy, self, 30)
        state = encode_state_full(my_snake, energies, enemies, self.W, self.H, 64)#, prev_dir=my_snake.last, flood=space)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            q_values = q_values[0]   
            #print("qvalues=", q_values)
            valid_actions = get_movesIA(self, 1, self.H, self.W)  # adapter selon obstacles
            #if my_snake.last == my_snake.head():valid_actions[0] = False
            hx, hy = self.snakes[1].head()
            countf = flood_fill_limited(hx, hy, self, 10)
            if countf > 1:
                for i, v in enumerate(valid_actions):
                    if v:
                        hx, hy = self.snakes[1].head()
                        count = flood_fill_limited(hx + DIRS[i][0], hy + DIRS[i][1], self, 10)
                        if count <= 1:
                            valid_actions[i] = False

            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9
            opp_action = q_values.argmax().item()


        #score, opp_move = minimax(self.clone(), 3, 1, -1e9, 1e9, self.H, self.W)
        self.move_snake(self.snakes[1], opp_action)
        #if (self.snakes[1].head() in self.energy):
        #    self.energy.remove(self.snakes[1].head())

        #gravity--------
        snakeme = SnakeS(0)  # crée un nouveau serpent avec id=0
        body = self.snakes[0].body[:len(self.snakes[0].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme.body = list(sbody)
        snakeme.len = len(self.snakes[0].body)
        snakeme.head = snakeme.len - 1
        snakeme.tail = 0

        snakeme2 = SnakeS(1)  # crée un nouveau serpent avec id=0
        body = self.snakes[1].body[:len(self.snakes[1].body)][::-1]  # copie et inverse seulement la partie "alive"
        sbody = [None] * MAX_BODY
        for i, (bx, by) in enumerate(body):
            sbody[i] = Pos(bx, by+BORDERH)
        snakeme2.body = list(sbody)
        snakeme2.len = len(self.snakes[1].body)
        snakeme2.head = snakeme2.len - 1
        snakeme2.tail = 0

        ngrid = Grid(self.H+BORDERH, self.W)
        ngrid.cell = []
        for i in range(BORDERH):
            r = []
            for x in range(self.W):
                r.append(EMPTY)
            ngrid.cell.append(r)

        for row in self.grid:
            r = []
            for e in row:
                if e == EMPTY:
                    r.append(EMPTY)
                elif e == WALL:
                    r.append(GWALL)
                elif e == ENERGY:
                    r.append(GENERGY)
                else:
                    r.append(EMPTY)
            ngrid.cell.append(r)


        game = GameState(self.W, self.H+BORDERH)
        game.snakeCount = 1

        # --- my snakes
        for i in range(1):
            game.snakes[i] = snakeme # ⚠️ important
            game.snakes[i].dir = UP
            game.snakes[i].alive = True

        # --- opp snakes
        for i in range(1):
            game.snakes[1 + i] = snakeme2
            game.snakes[1 + i].dir = UP
            game.snakes[1 + i].alive = True

        # --- energy
        #for i in range(power_source_count):
        #    game.energy[i] = energy[i].copy()

        game.grid = ngrid
        #game.energyCount = power_source_count

        self.simulation.doFalls(game, 2)
        #self.simulation.doIntercoiledFalls(game, 2)

        self.snakes[0].body = []
        for k in range(game.snakes[0].len):
            id_ = (game.snakes[0].head - k + MAX_BODY) % MAX_BODY
            x, y = game.snakes[0].body[id_].x, game.snakes[0].body[id_].y
            self.snakes[0].body.append((x, y-BORDERH))

        self.snakes[1].body = []
        for k in range(game.snakes[1].len):
            id_ = (game.snakes[1].head - k + MAX_BODY) % MAX_BODY
            x, y = game.snakes[1].body[id_].x, game.snakes[1].body[id_].y
            self.snakes[1].body.append((x, y-BORDERH))


        #self.apply_gravity()

        return score


# ===== RENDER =====
class Renderer:
    def __init__(self, h, w):
        pygame.init()
        self.H = h
        self.W = w
        self.screen = pygame.display.set_mode((self.W*CELL_SIZE, self.H*CELL_SIZE))

    def Reload(self, H, W):
        self.H = H
        self.W = W
        self.screen = pygame.display.set_mode((self.W*CELL_SIZE, self.H*CELL_SIZE))

    def draw(self, game):

        self.screen.fill((0,0,0))

        # grid
        for y in range(self.H):
            for x in range(self.W):
                rect = (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)

                if game.grid[y][x] == WALL:
                    pygame.draw.rect(self.screen, (100,100,100), rect)

                elif game.grid[y][x] == ENERGY:
                    pygame.draw.rect(self.screen, (0,255,0), rect)

        # snakes
        for i, snake in enumerate(game.snakes):
            color = (0,0,255-i*5) if i == 0 else (255-i*5,0,0)

            for (x,y) in snake.body:
                rect = (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)

        pygame.display.flip()


def evaluate(game, me, H, W):
    global dist_maps

    if not game.snakes[me].alive:
        return -1e9

    opp = 1 - me
    if not game.snakes[opp].alive:
        return 1e9

    my_len = len(game.snakes[me].body)
    opp_len = len(game.snakes[opp].body)

    hx, hy = game.snakes[me].head()

    # distance pomme
    
    best_dist = 999
    for i, (x, y) in enumerate(game.energy):
        d = abs(x - hx) + abs(y - hy)
        best_dist = min(best_dist, d)
    
        
    """
        best_dist = INF
        if 0<=hx<game.W and 0<=hy<game.H:
            for i in game.idx_energy:
                d = dist_maps[i][hy][hx]
                if d < best_dist:
                    best_dist = d
    """

    score = 0
    score += (my_len - opp_len) * 50
    #score += -best_dist
    score += 20 / (best_dist + 1)

    return score



def order_moves(game, i, moves, H, W):

    scored = []

    for m in moves:
        g2 = game.clone()
        
        if i == 0:
            g2.step([m, 0])
        else:
            g2.step([0, m])

        score = evaluate(g2, i, H, W)
        scored.append((score, m))

    # tri du meilleur au pire
    scored.sort(reverse=True)

    return [m for _, m in scored]


def minimax(game, depth, me, alpha, beta, H, W):

    opp = 1 - me

    #  terminal
    if depth == 0 or not game.snakes[me].alive:
        return evaluate(game, me, H, W), None

    my_moves = get_moves(game, me, H, W)
    opp_moves = get_moves(game, opp, H, W)

    #  move ordering (ULTRA IMPORTANT)
    my_moves = order_moves(game, me, my_moves, H, W)
    opp_moves = order_moves(game, opp, opp_moves, H, W)

    best_score = -1e9
    best_move = my_moves[0]

    for m in my_moves:

        worst_case = 1e9

        for om in opp_moves:

            g2 = game.clone()
            g2.step([m, om])  # simultan�
            #g2.compute_distance_map()

            score, _ = minimax(g2, depth-1, me, alpha, beta, H, W)

            #  suicide instant = skip
            if not g2.snakes[me].alive:
                score -= 1e6

            worst_case = min(worst_case, score)

            beta = min(beta, worst_case)

            #  PRUNE ennemi
            if beta <= alpha:
                break

        if worst_case > best_score:
            best_score = worst_case
            best_move = m

        alpha = max(alpha, best_score)

        #  PRUNE moi
        if beta <= alpha:
            break

    return best_score, best_move


# ===== DQN discret pour �tats (C,H,W) et 5 actions =====
# - R�seau conv Q(s,a) pour chaque action
# - e-greedy
# - Experience Replay + Target Network
# - Huber loss
# - Soft update (Polyak) optionnel

import math, random, collections, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FC_DQN64(nn.Module):
    def __init__(self, input_dim, num_actions):
        """
        input_dim : dimension du vecteur d'�tat
        num_actions : nombre de directions possibles (UP, DOWN, LEFT, RIGHT)
        """
        super().__init__()
        # R�seau enti�rement connect�
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, num_actions)

        # Initialisation
        nn.init.uniform_(self.fc4.weight, -0.1, 0.1)
        nn.init.constant_(self.fc4.bias, 0.0)

    def forward(self, x):
        # x: (B, input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)   # Q(s,�)
        return x


class FC_DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        """
        input_dim : dimension du vecteur d'�tat
        num_actions : nombre de directions possibles (UP, DOWN, LEFT, RIGHT)
        """
        super().__init__()
        # R�seau enti�rement connect�
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 64)
        self.fc4 = nn.Linear(64, num_actions)

        # Initialisation
        nn.init.uniform_(self.fc4.weight, -0.1, 0.1)
        nn.init.constant_(self.fc4.bias, 0.0)

    def forward(self, x):
        # x: (B, input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)   # Q(s,�)
        return x


# -------------------------------------------------------
# Hyperparam�tres (ajuste si besoin)
# -------------------------------------------------------
NUM_ACTIONS   = 4     # ex: [haut, bas, gauche, droite, rien]
STATE_CHANNELS= 11     # ex: 93
HEIGHT, WIDTH = 10, 20
BUFFER_SIZE   = 100_000
BATCH_SIZE    = 64
GAMMA         = 0.99
LR            = 2.5e-4
TARGET_UPDATE = 750     # tous les N steps on copie vers le target (hard update)
TAU           = 0.005       # si >0, on fait un soft update (Polyak). Laisse 0 si tu utilises TARGET_UPDATE
EPS_START     = 1.0
EPS_END       = 0.1
EPS_DECAY     = 80000 #50000 pour 15min    # plus c'est grand, plus epsilon d�cro�t lentement
GRAD_NORM_CLIP= 1.0
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_INPUT     = 24
MAX_BODYNN      = 64

#-----------------------------------------
# Replay Buffer
# -------------------------------------------------------
Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done", "mask", "next_mask")
)

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        # state / next_state: np.array ou torch.Tensor (C,H,W)
        # action: int
        # reward: float
        # done: bool (ou 0/1)
        self.buf.append(Transition(state, action, reward, next_state, done, mask, next_mask))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)

class PrioritizedReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done, valid=None, next_valid=None):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, valid, next_valid))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done, valid, next_valid)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize

        batch = list(zip(*samples))
        return (*batch, indices, weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------------
# Agent DQN
# -------------------------------------------------------
class DQNAgent:
    def __init__(self, num_input=NUM_INPUT, num_actions=NUM_ACTIONS):
        self.num_actions = num_actions
        self.q_net       = FC_DQN(num_input, num_actions).to(DEVICE)
        self.q_target    = FC_DQN(num_input, num_actions).to(DEVICE)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer   = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        #self.scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #                    self.optimizer,
        #                    mode='min',
        #                    factor=0.5,
        #                    patience=1000
        #                )
        #self.replay      = ReplayBuffer()
        self.replay      = PrioritizedReplayBuffer()
        self.train_steps = 0
        self.eps         = EPS_START
        self.beta_start  = 0.4
        self.beta_end    = 1.0
        self.beta_frames = 200_000
        self.frame = 1

    @torch.no_grad()
    def select_action22(self, state):
        """
        state: torch.Tensor (C,H,W) ou np.ndarray (C,H,W)
        return: int, action choisie pour le joueur
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = state.unsqueeze(0).to(DEVICE)  # (1,C,H,W)

        # e-greedy
        self.eps = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.train_steps / EPS_DECAY
        )

        q_values = self.q_net(state)  # (1, num_actions)
        q_values = q_values[0]        # (num_actions,)

        if random.random() < self.eps:
            action = random.randint(0, q_values.shape[0] - 1)
        else:
            action = q_values.argmax(dim=0).item()

        return action

    @torch.no_grad()
    def select_action(self, state, valid_actions=None):
        # convertir numpy -> torch si n�cessaire
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
    
        # ajouter batch dimension si n�cessaire
        if state.ndim == 1:
            state = state.unsqueeze(0)  # (1, input_dim)

        state = state.to(DEVICE)

        q_values = self.q_net(state)[0]  # (num_actions,)

        if valid_actions is not None:
            mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
            q_values[~mask] = -1e9

        # epsilon-greedy
        self.eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * self.train_steps / EPS_DECAY)
        if random.random() < self.eps:
            if valid_actions is None:
                action = random.randint(0, q_values.shape[0]-1)
            else:
                valid_indices = [i for i, ok in enumerate(valid_actions) if ok]
                if len(valid_indices) == 0:
                    action = random.randint(0, q_values.shape[0]-1)
                else:
                    action = random.choice(valid_indices)
        else:
            action = q_values.argmax().item()

        return action


    def optimize_ancien(self):
        if len(self.replay) < BATCH_SIZE:
            return None

        # --- Sample du replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay.sample(BATCH_SIZE)

        state_batch      = torch.tensor(np.array(state_batch), dtype=torch.float32, device=DEVICE)      # (B,C,H,W)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32, device=DEVICE) # (B,C,H,W)
        action_batch     = torch.tensor(np.array(action_batch), dtype=torch.long, device=DEVICE)        # (B,)
        reward_batch     = torch.tensor(np.array(reward_batch), dtype=torch.float32, device=DEVICE)     # (B,)
        done_batch       = torch.tensor(np.array(done_batch), dtype=torch.float32, device=DEVICE)       # (B,)

        #norm_reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)

        # --- Q(s,a) courant
        q_values = self.q_net(state_batch)              # (B, NUM_ACTIONS)
        q_sa = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            # --- Double DQN
            next_q_values = self.q_net(next_state_batch)             # (B, NUM_ACTIONS)
            next_actions = next_q_values.argmax(dim=1, keepdim=True) # (B,1)

            next_q_target = self.q_target(next_state_batch)          # (B, NUM_ACTIONS)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)  # (B,)

            # --- Cible
            #target = norm_reward_batch + (1.0 - done_batch) * GAMMA * next_q
            target = reward_batch + (1.0 - done_batch) * GAMMA * next_q
            #target = 0.9 * target + 0.1 * q_sa.detach()  # smoothing (optionnel)

        # --- Loss
        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()
        #self.scheduler.step(loss)
        self.train_steps += 1

        # --- Update du target net
        if TAU > 0.0:  # soft update
            with torch.no_grad():
                for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
                    tp.data.mul_(1.0 - TAU).add_(TAU * p.data)
        elif self.train_steps % TARGET_UPDATE == 0:  # hard update
            self.q_target.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def optimizemask(self):
        if len(self.replay) < BATCH_SIZE:
            return None

        # --- Sample
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, mask_batch, next_mask_batch = self.replay.sample(BATCH_SIZE)

        state_batch      = torch.tensor(np.array(state_batch), dtype=torch.float32, device=DEVICE)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32, device=DEVICE)
        action_batch     = torch.tensor(np.array(action_batch), dtype=torch.long, device=DEVICE)
        reward_batch     = torch.tensor(np.array(reward_batch), dtype=torch.float32, device=DEVICE)
        done_batch       = torch.tensor(np.array(done_batch), dtype=torch.float32, device=DEVICE)

        mask_batch       = torch.tensor(np.array(mask_batch), dtype=torch.bool, device=DEVICE)
        next_mask_batch  = torch.tensor(np.array(next_mask_batch), dtype=torch.bool, device=DEVICE)

        # --- Q(s,a)
        q_values = self.q_net(state_batch)
        q_sa = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():

            #  FIX CRITIQUE : éviter "all False"
            invalid_rows = (~next_mask_batch).all(dim=1)
            next_mask_batch[invalid_rows] = True

            # --- Online net (choix action)
            next_q_values = self.q_net(next_state_batch)
            next_q_values = next_q_values.masked_fill(~next_mask_batch, -1e9)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)

            # --- Target net (évaluation)
            next_q_target = self.q_target(next_state_batch)
            next_q_target = next_q_target.masked_fill(~next_mask_batch, -1e9)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)

            #  STABILISATION (très important)
            next_q = torch.clamp(next_q, -10, 10)

            # --- Target
            target = reward_batch + (1.0 - done_batch) * GAMMA * next_q

            #  OPTIONNEL mais recommandé
            target = torch.clamp(target, -10, 10)

        # --- Loss
        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        self.train_steps += 1

        # --- Update target net
        if TAU > 0.0:
            with torch.no_grad():
                for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
                    tp.data.mul_(1.0 - TAU).add_(TAU * p.data)
        elif self.train_steps % TARGET_UPDATE == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def optimize(self, beta=0.4):
        if len(self.replay) < BATCH_SIZE:
            return None

        beta = self.beta_start + (self.beta_end - self.beta_start) * (self.frame / self.beta_frames)
        beta = min(1.0, beta)
        self.frame += 1

        # --- Sample avec PER
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, mask_batch, next_mask_batch, indices, weights = \
            self.replay.sample(BATCH_SIZE, beta=beta)

        state_batch      = torch.tensor(np.array(state_batch), dtype=torch.float32, device=DEVICE)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32, device=DEVICE)
        action_batch     = torch.tensor(np.array(action_batch), dtype=torch.long, device=DEVICE)
        reward_batch     = torch.tensor(np.array(reward_batch), dtype=torch.float32, device=DEVICE)
        done_batch       = torch.tensor(np.array(done_batch), dtype=torch.float32, device=DEVICE)
        mask_batch       = torch.tensor(np.array(mask_batch), dtype=torch.bool, device=DEVICE)
        next_mask_batch  = torch.tensor(np.array(next_mask_batch), dtype=torch.bool, device=DEVICE)
        weights          = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

        # --- Q(s,a)
        q_values = self.q_net(state_batch)
        q_sa = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # --- FIX CRITIQUE : éviter "all False"
            invalid_rows = (~next_mask_batch).all(dim=1)
            next_mask_batch[invalid_rows] = True

            # --- Online net (choix action)
            next_q_values = self.q_net(next_state_batch)
            next_q_values = next_q_values.masked_fill(~next_mask_batch, -1e9)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)

            # --- Target net (évaluation)
            next_q_target = self.q_target(next_state_batch)
            next_q_target = next_q_target.masked_fill(~next_mask_batch, -1e9)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)

            # --- Stabilisation
            next_q = torch.clamp(next_q, -10, 10)
            target = reward_batch + (1.0 - done_batch) * GAMMA * next_q
            target = torch.clamp(target, -10, 10)

        # --- Loss pondérée par IS weights
        td_errors = target - q_sa
        loss = (weights * F.smooth_l1_loss(q_sa, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        # --- Mise à jour des priorités dans le replay
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.replay.update_priorities(indices, new_priorities)

        self.train_steps += 1

        # --- Update target net
        if TAU > 0.0:
            with torch.no_grad():
                for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
                    tp.data.mul_(1.0 - TAU).add_(TAU * p.data)
        elif self.train_steps % TARGET_UPDATE == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return loss.item()


import numpy as np




import time
import torch
from datetime import timedelta
import matplotlib.pyplot as plt

def Train_DQN_FC(agent, params, max_episode_min=40, max_steps=100):
    """
    agent: DQNAgent utilisant FC_DQN
    max_episode_min: dur�e max en minutes
    max_steps: nombre max de steps par partie
    """
    global dist_maps

    agentT = DQNAgent(num_input=24, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128topatt.pth', map_location=DEVICE)
    agentT.q_net.load_state_dict(checkpoint['model_state_dict'])

    # 3. (Optionnel) mettre le r�seau en mode �valuation
    agentT.q_net.eval()

    

    same = 0
    MAXSAME = 2
    episodes = 0
    MAX_EPISODE_T = time.perf_counter() + max_episode_min*60
    all_loss = []
    time_tot = 0
    lossm = 0
    countloss =  0
    total_score = 0

    while time.perf_counter() < MAX_EPISODE_T:
        start_time = time.perf_counter()

        game = Game()  # nouvelle partie
        
        episode_reward = 0
        step_count = 0
        done = False
        last_loss = 0
        while not done and step_count < max_steps:
            my_snake = game.snakes[0]  # ton snake
            opp_snake = game.snakes[1]  # adversaire

            
            # --- encoder �tat
            state = encode_state_full(my_snake, game.energy, [opp_snake], game.W, game.H, MAX_BODYNN)

            # --- valid actions
            valid = np.ones(4, dtype=np.bool_)  # par d�faut toutes valides
            hx, hy = my_snake.head()
            for i, (dx, dy) in enumerate(DIRS):
                nx, ny = hx+dx, hy+dy
                if nx < 0 or ny < 0 or nx >= game.W or ny >= game.H or game.grid[ny][nx] == WALL:
                    valid[i] = False
                if (nx, ny) in opp_snake.body:
                    valid[i] = False
                if (nx, ny) in my_snake.body:
                    valid[i] = False

            if is_vertical(my_snake) and 0<=hx<game.W and 0<=hy-1<game.H and game.grid[hy-1][hx] == EMPTY:
                valid[0] = False
            
            if not valid.any():
                valid[:] = True
                
            # --- choisir action
            action = agent.select_action(state, valid_actions=valid)

            # --- actions adversaire (al�atoire)
            
            """
                import random
                opp_valid = [True]*4
                hx, hy = opp_snake.head()
                for i, (dx, dy) in enumerate(DIRS):
                    nx, ny = hx+dx, hy+dy
                    if nx < 0 or ny < 0 or nx >= game.W or ny >= game.H or game.grid[ny][nx] == WALL:
                        opp_valid[i] = False
                    if (nx, ny) in my_snake.body:
                        opp_valid[i] = False
                    if (nx, ny) in opp_snake.body:
                        opp_valid[i] = False
                ch = [i for i, ok in enumerate(opp_valid) if ok]
                opp_action = random.randint(0, 3)
                if len(ch) > 0:opp_action = random.choice(ch)
            """


            #score, opp_action = minimax(game.clone(), 3, 1, -1e9, 1e9)
            my_snake.last = my_snake.head()

            prev_energy = set(game.energy)
            prev_idx = list(game.idx_energy)
            # --- appliquer moves
            game.step_IAOnlyFanTraining(agentT, action)

            my_snake = game.snakes[0]  # ton snake
            opp_snake = game.snakes[1]  # adversaire

            next_valid = np.ones(4, dtype=np.bool_) # par d�faut toutes valides
            hx, hy = my_snake.head()
            for i, (dx, dy) in enumerate(DIRS):
                nx, ny = hx+dx, hy+dy
                if nx < 0 or ny < 0 or nx >= game.W or ny >= game.H or game.grid[ny][nx] == WALL:
                    next_valid[i] = False
                if (nx, ny) in opp_snake.body:
                    next_valid[i] = False
                if (nx, ny) in my_snake.body:
                    next_valid[i] = False

            if is_vertical(my_snake) and 0<=hx<game.W and 0<=hy-1<game.H and game.grid[hy-1][hx] == EMPTY:
                next_valid[0] = False

            if not next_valid.any():
                next_valid[:] = True

            # --- reward simple
            reward = 0
            done = False

            # --- mort
            
            if not my_snake.alive or len(my_snake.body) < 3:
                reward = params["death"]
                done = True

            # --- kill
            elif len(opp_snake.body) < 3:
                reward = params["kill"]
                done = True

            else:
                # --- eat
                if my_snake.head() in prev_energy:
                    reward += params["eat"]

                # --- anti-loop
                """
                    if my_snake.last == my_snake.head():
                        same += 1
                    else:
                        same = 0

                    if same >= MAXSAME:
                        same = 0
                        reward -= 0.05
                """
                # --- shaping
                hx, hy = my_snake.head()

                if 0 <= hx < game.W and 0 <= hy < game.H and game.energy:
                    bestEn = 1e9
                    for ex, ey in game.energy:
                        d = abs(hx - ex) + abs(hy - ey)
                        if d < bestEn:
                            bestEn = d

                    reward += params["dist"] / (bestEn + 1)
                
                space = flood_fill_limited(hx, hy, game, len(my_snake.body))
                if space < len(my_snake.body):
                    reward -= params["danger"]
                else:
                    reward += params["space"] / 500.0              # ↓ réduit fort

                #size_diff = len(my_snake.body) #- len(opp_snake.body)
                #reward += size_diff * 0.02           # ↑ beaucoup plus utile

            # --- fin jeu
            if len(game.energy) == 0:# or step_count == max_steps-1:
                size_diff = len(my_snake.body) - len(opp_snake.body)
                if size_diff > 0:
                    reward = params["win"]
                else:
                    reward = params["lose"]
                done = True
            

            # --- clip
            #reward = max(-1, min(1, reward))
        
            # --- next_state
            next_state = encode_state_full(my_snake, game.energy, [opp_snake], game.W, game.H, MAX_BODYNN)

            # --- push replay
            agent.replay.push(state, action, reward, next_state, float(done), valid, next_valid)

            

            # --- optimiser le r�seau
            loss = 0
            if len(agent.replay) > 1000 and step_count % 2 == 0:
                l = agent.optimize()
                if l is not None:
                    loss += l
                countloss += 1
                last_loss = loss
                all_loss.append(loss)
                lossm += loss 

            episode_reward += reward
            step_count += 1

        total_score += episode_reward
        episodes += 1

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        time_tot += execution_time

        if episodes % 100 == 0:
            print(f"Episode {episodes}, Reward total: {episode_reward:.2f}, steps={step_count}, eps={agent.eps:.3f},  loss={(last_loss):.4f} lossm={lossm/countloss}")
            print(f"Temps d'execution: {execution_time:.2f}s, total: {time_tot:.2f}s")

    #return total_score / (episodes + 1e-6)

    # --- sauvegarder le mod�le
    torch.save({'model_state_dict': agent.q_net.state_dict()}, 'checkpoint_snake_dqn_fc128.pth')
    # --- courbe perte
    plt.plot(all_loss)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Evolution de la perte moyenne")
    plt.grid(True)
    plt.show()


# ===== MAIN LOOP =====
def main():

    game = Game()
    renderer = Renderer(game.H, game.W)
    clock = pygame.time.Clock()

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = Game()

        # random moves (remplace par ton bot)
        #moves = [random.randint(0,3), random.randint(0,3)]

        score, my_move = minimax(game, 4, 0, -1e9, 1e9, game.H, game.W)
        # ennemi (simple pour test)
        #opp_move = random.randint(0,3)
        score, opp_move = minimax(game, 4, 1, -1e9, 1e9, game.H, game.W)
        moves = [my_move, opp_move]

        game.step(moves)

        renderer.draw(game)

        clock.tick(10)


def mainSm(params):

    global dist_maps
    # 1. Cr�e un agent identique
    agent = DQNAgent(num_input=24, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128topatt.pth', map_location=DEVICE)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])

    # 3. (Optionnel) mettre le r�seau en mode �valuation
    agent.q_net.eval()

    
    game = Game()
    game.init_Smitsimax()
    game.simulation.params = params
    renderer = Renderer(game.H, game.W)
    #dist_maps = compute_all_energy_distances(game)

    clock = pygame.time.Clock()

    WINB = 0
    WINR = 0
    TOTAL = 0

    turn = 0
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = Game()
                    game.init_Smitsimax()
                    game.simulation.params = params
                    renderer.Reload(game.H, game.W)
                    turn = 0
                    #dist_maps = compute_all_energy_distances(game)

        # random moves (remplace par ton bot)
        #moves = [random.randint(0,3), random.randint(0,3)]

        game.step_IAOnlyFanSmitsimax(agent)

        if len(game.snakes[0].body) < 3 or len(game.snakes[1].body) < 3  or turn == 200 or len(game.energy) == 0:
            if len(game.snakes[0].body) >= 3 and len(game.snakes[1].body) < 3:
                WINB+=1
            elif len(game.snakes[0].body) < 3 and len(game.snakes[1].body) >= 3:
                WINR+=1
            elif len(game.snakes[0].body) > len(game.snakes[1].body):
                WINB+=1
            elif len(game.snakes[0].body) < len(game.snakes[1].body):
                WINR+=1

            TOTAL += 1
            print("BLUE ATTEN=", WINB,"/",TOTAL," = ", WINB/TOTAL*100,"%")
            print("RED TOPATT=", WINR,"/",TOTAL," = ", WINR/TOTAL*100, "%")

            game = Game()
            game.init_Smitsimax()
            game.simulation.params = params
            renderer.Reload(game.H, game.W)
            turn = 0

        renderer.draw(game)

        turn += 1

        clock.tick(100)


def mainSmB(params):

    global dist_maps
    # 1. Cr�e un agent identique
    agent = DQNAgent(num_input=24, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128topatt.pth', map_location=DEVICE)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])

    # 3. (Optionnel) mettre le r�seau en mode �valuation
    agent.q_net.eval()

    
    game = Game()
    game.init_Smitsimax()
    game.simulation.params = params
    renderer = Renderer(game.H, game.W)
    #dist_maps = compute_all_energy_distances(game)

    clock = pygame.time.Clock()

    WINB = 0
    WINR = 0
    TOTAL = 0

    turn = 0
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = Game()
                    game.init_Smitsimax()
                    game.simulation.params = params
                    renderer.Reload(game.H, game.W)
                    turn = 0
                    #dist_maps = compute_all_energy_distances(game)

        # random moves (remplace par ton bot)
        #moves = [random.randint(0,3), random.randint(0,3)]

        game.step_IAOnlyFanBS(agent)

        if len(game.snakes[0].body) < 3 or len(game.snakes[1].body) < 3  or turn == 200 or len(game.energy) == 0:
            if len(game.snakes[0].body) >= 3 and len(game.snakes[1].body) < 3:
                WINB+=1
            elif len(game.snakes[0].body) < 3 and len(game.snakes[1].body) >= 3:
                WINR+=1
            elif len(game.snakes[0].body) > len(game.snakes[1].body):
                WINB+=1
            elif len(game.snakes[0].body) < len(game.snakes[1].body):
                WINR+=1

            TOTAL += 1
            print("BLUE ATTEN=", WINB,"/",TOTAL," = ", WINB/TOTAL*100,"%")
            print("RED TOPATT=", WINR,"/",TOTAL," = ", WINR/TOTAL*100, "%")

            game = Game()
            game.init_Smitsimax()
            game.simulation.params = params
            renderer.Reload(game.H, game.W)
            turn = 0

        renderer.draw(game)

        turn += 1

        clock.tick(100)


def mainSmVN(params):

    global dist_maps
    # 1. Cr�e un agent identique
    agent = DQNAgent(num_input=24, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128topatt.pth', map_location=DEVICE)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])

    # 3. (Optionnel) mettre le r�seau en mode �valuation
    agent.q_net.eval()

    model = Net()
    model.load_state_dict(torch.load("snake_model.pth"))
    model.eval()

    
    game = Game()
    game.init_Smitsimax()
    game.simulation.params = params
    renderer = Renderer(game.H, game.W)
    #dist_maps = compute_all_energy_distances(game)

    clock = pygame.time.Clock()

    WINB = 0
    WINR = 0
    TOTAL = 0

    turn = 0
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = Game()
                    game.init_Smitsimax()
                    game.simulation.params = params
                    renderer.Reload(game.H, game.W)
                    turn = 0
                    #dist_maps = compute_all_energy_distances(game)

        # random moves (remplace par ton bot)
        #moves = [random.randint(0,3), random.randint(0,3)]

        game.step_IAOnlyFanSmitsimaxValueNet(agent, model)

        if len(game.snakes[0].body) < 3 or len(game.snakes[1].body) < 3  or turn == 200 or len(game.energy) == 0:
            if len(game.snakes[0].body) >= 3 and len(game.snakes[1].body) < 3:
                WINB+=1
            elif len(game.snakes[0].body) < 3 and len(game.snakes[1].body) >= 3:
                WINR+=1
            elif len(game.snakes[0].body) > len(game.snakes[1].body):
                WINB+=1
            elif len(game.snakes[0].body) < len(game.snakes[1].body):
                WINR+=1

            TOTAL += 1
            print("BLUE ATTEN=", WINB,"/",TOTAL," = ", WINB/TOTAL*100,"%")
            print("RED TOPATT=", WINR,"/",TOTAL," = ", WINR/TOTAL*100, "%")

            game = Game()
            game.init_Smitsimax()
            game.simulation.params = params
            renderer.Reload(game.H, game.W)
            turn = 0

        renderer.draw(game)

        turn += 1

        clock.tick(100)


def mainSmPUCT(params):

    global dist_maps
    # 1. Cr�e un agent identique
    agent = DQNAgent(num_input=24, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128topatt.pth', map_location=DEVICE)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])

    # 3. (Optionnel) mettre le r�seau en mode �valuation
    agent.q_net.eval()

    model = PolicyNet(24)
    model.load_state_dict(torch.load("policy.pth"))
    model.eval()

    
    game = Game()
    game.init_Smitsimax()
    game.simulation.params = params
    renderer = Renderer(game.H, game.W)
    #dist_maps = compute_all_energy_distances(game)

    clock = pygame.time.Clock()

    WINB = 0
    WINR = 0
    TOTAL = 0

    turn = 0
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = Game()
                    game.init_Smitsimax()
                    game.simulation.params = params
                    renderer.Reload(game.H, game.W)
                    turn = 0
                    #dist_maps = compute_all_energy_distances(game)

        # random moves (remplace par ton bot)
        #moves = [random.randint(0,3), random.randint(0,3)]

        game.step_IAOnlyFanSmitsimaxPUCT(agent, model)

        if len(game.snakes[0].body) < 3 or len(game.snakes[1].body) < 3  or turn == 200 or len(game.energy) == 0:
            if len(game.snakes[0].body) >= 3 and len(game.snakes[1].body) < 3:
                WINB+=1
            elif len(game.snakes[0].body) < 3 and len(game.snakes[1].body) >= 3:
                WINR+=1
            elif len(game.snakes[0].body) > len(game.snakes[1].body):
                WINB+=1
            elif len(game.snakes[0].body) < len(game.snakes[1].body):
                WINR+=1

            TOTAL += 1
            print("BLUE ATTEN=", WINB,"/",TOTAL," = ", WINB/TOTAL*100,"%")
            print("RED TOPATT=", WINR,"/",TOTAL," = ", WINR/TOTAL*100, "%")

            game = Game()
            game.init_Smitsimax()
            game.simulation.params = params
            renderer.Reload(game.H, game.W)
            turn = 0

        renderer.draw(game)

        turn += 1

        clock.tick(100)


def mainIA():
    global dist_maps
    # 1. Cr�e un agent identique
    agent = DQNAgent(num_input=24, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128.pth', map_location=DEVICE)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])

    # 3. (Optionnel) mettre le r�seau en mode �valuation
    agent.q_net.eval()

    agent2 = DQNAgent(num_input=24, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128best2.pth', map_location=DEVICE)
    agent2.q_net.load_state_dict(checkpoint['model_state_dict'])

    # 3. (Optionnel) mettre le r�seau en mode �valuation
    agent2.q_net.eval()

    game = Game()
    renderer = Renderer(game.H, game.W)
    #dist_maps = compute_all_energy_distances(game)

    clock = pygame.time.Clock()

    WINB = 0
    WINR = 0
    TOTAL = 0

    turn = 0
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = Game()
                    renderer.Reload(game.H, game.W)
                    turn = 0
                    #dist_maps = compute_all_energy_distances(game)

        # random moves (remplace par ton bot)
        #moves = [random.randint(0,3), random.randint(0,3)]

        game.step_IAOnlyFan(agent, agent2)

        if len(game.snakes[0].body) < 3 or len(game.snakes[1].body) < 3  or turn == 200 or len(game.energy) == 0:
            if len(game.snakes[0].body) >= 3 and len(game.snakes[1].body) < 3:
                WINB+=1
            elif len(game.snakes[0].body) < 3 and len(game.snakes[1].body) >= 3:
                WINR+=1
            elif len(game.snakes[0].body) > len(game.snakes[1].body):
                WINB+=1
            elif len(game.snakes[0].body) < len(game.snakes[1].body):
                WINR+=1

            TOTAL += 1
            print("BLUE ATTEN=", WINB,"/",TOTAL," = ", WINB/TOTAL*100,"%")
            print("RED TOPATT=", WINR,"/",TOTAL," = ", WINR/TOTAL*100, "%")

            game = Game()
            renderer.Reload(game.H, game.W)
            turn = 0

        renderer.draw(game)

        turn += 1

        clock.tick(10)

def objective(trial):

    params = {
        "eat": trial.suggest_float("eat", 0.3, 1.5),
        "dist": trial.suggest_float("dist", 0.01, 0.3),
        "space": trial.suggest_float("space", 0.0005, 0.01),
        "danger": trial.suggest_float("danger", 0.01, 0.2),
        "death": trial.suggest_float("death", -1.5, -0.5),
        "kill": trial.suggest_float("kill", 0.5, 1.5),
        "win": trial.suggest_float("win", 0.5, 2.0),
        "lose": trial.suggest_float("lose", -1.5, -0.5),
    }

    # 1. Cr�e un agent identique
    agent = DQNAgent(num_input=NUM_INPUT, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128.pth', map_location=DEVICE)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])

    score = Train_DQN_FC(agent, params, max_episode_min=2, max_steps=200)

    return score

def objectiveS(trial):

    
    params = {
        
        "death": trial.suggest_float("death", -300, -50),
        "size": trial.suggest_float("size", 0.0, 5.0),
        "dist": trial.suggest_float("dist", 0.0, 20.0),
        "win": trial.suggest_float("win", 50, 200),
        "lose": trial.suggest_float("lose", -200, -50),
        "flood": trial.suggest_float("flood", -50, 0),
        "eat": trial.suggest_float("eat", 1.0, 50.0),
        "lose_part": trial.suggest_float("lose_part", -20, -1),
        "kill": trial.suggest_float("kill", 20, 150),
        "kill_dude": trial.suggest_float("kill_dude", -50, -10),
        'Cexplore': 0.5436079574741095
    }
    
    

    #params = {"Cexplore":  trial.suggest_float("Cexplore", 0.1, 2.0),
    #          'death': -148.42012305961117, 'size': 4.249118757407515, 'dist': 17.495777101115365, 'win': 142.95511897983494, 'lose': -68.94264997675857, 'flood': -21.891401061220208, 'eat': 49.930044240674235, 'lose_part': -6.9065559067686, 'kill': 99.14748219536332, 'kill_dude': -31.657405874965782
    #         }

    #params = {'death': -61.93605385655409, 'size': 4.881137038784745, 'dist': 12.48810009970582, 
    # 'win': 192.94434651075196, 'lose': -190.57311227738293, 'flood': -49.99868548591225, 
    # 'eat': 33.90698313151979, 'lose_part': -19.984215693570313, 'kill': 147.96318546156164, 'kill_dude': -10.693103912807167,
    # "Cexplore":  trial.suggest_float("Cexplore", 0.1, 2.0),
    # }
 
    total_score = 0

    # 1. Cr�e un agent identique
    agent = DQNAgent(num_input=24, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128topatt.pth', map_location=DEVICE)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])

    # 3. (Optionnel) mettre le r�seau en mode �valuation
    agent.q_net.eval()

    
    game = Game()
    game.init_Smitsimax()
    game.simulation.params = params

    N = 0
    turn = 0
    while True:
        N+=1
        total_score += game.step_IAOnlyFanSmitsimax(agent)

        if len(game.snakes[0].body) < 3 or len(game.snakes[1].body) < 3  or turn == 200 or len(game.energy) == 0:
            break;

        turn += 1
    
    return total_score / N


def SaveTrainingValueNet(max_episode_min=5):

    """
        params = {
        
            "death": trial.suggest_float("death", -300, -50),
            "size": trial.suggest_float("size", 0.0, 5.0),
            "dist": trial.suggest_float("dist", 0.0, 20.0),
            "win": trial.suggest_float("win", 50, 200),
            "lose": trial.suggest_float("lose", -200, -50),
            "flood": trial.suggest_float("flood", -50, 0),
            "eat": trial.suggest_float("eat", 1.0, 50.0),
            "lose_part": trial.suggest_float("lose_part", -20, -1),
            "kill": trial.suggest_float("kill", 20, 150),
            "kill_dude": trial.suggest_float("kill_dude", -50, -10),
            'Cexplore': 0.5436079574741095
        }
    """
    

    params = {"Cexplore": 0.5436079574741095,
              'death': -148.42012305961117, 'size': 4.249118757407515, 'dist': 17.495777101115365, 'win': 142.95511897983494, 'lose': -68.94264997675857, 'flood': -21.891401061220208, 'eat': 49.930044240674235, 'lose_part': -6.9065559067686, 'kill': 99.14748219536332, 'kill_dude': -31.657405874965782
             }
        
 
    total_score = 0

    # 1. Cr�e un agent identique
    agent = DQNAgent(num_input=24, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128topatt.pth', map_location=DEVICE)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])

    # 3. (Optionnel) mettre le r�seau en mode �valuation
    agent.q_net.eval()

    all_states = []
    all_targets = []

    MAX_EPISODE_T = time.perf_counter() + max_episode_min*60
    N = 0
    while time.perf_counter() < MAX_EPISODE_T:
    
        game = Game()
        game.init_Smitsimax()
        game.init_Training_ValueNet()
        game.simulation.params = params

        
        turn = 0
        while True:
           
            total_score += game.step_IAOnlyFanSmitsimaxTraining(agent)

            if len(game.snakes[0].body) < 3 or len(game.snakes[1].body) < 3  or turn == 200 or len(game.energy) == 0:
                break;

            turn += 1

        all_states.extend(game.states)
        all_targets.extend(game.targets)

        N += 1

        #if  N % 5 == 0:
        print("N finished= ", N)
    

    all_states = np.array(all_states, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    np.savez("dataset_snake.npz", states=all_states, targets=all_targets)
    print(f"Saved dataset with {len(all_states)} samples.")

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SnakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def Training_Value_Net(max_episode_min = 5):

    data = np.load("dataset_snake.npz")

    X = data["states"]
    y = data["targets"]

    print(X.shape)  # (N, 24)
    print(y.shape)  # (N,)

    dataset = SnakeDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    time_tot = 0
    MAX_EPISODE_T = time.perf_counter() + max_episode_min*60
    episodes = 0
    all_loss = []
    while time.perf_counter() < MAX_EPISODE_T:
        start_time = time.perf_counter()
        total_loss = 0
        count = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count +=1

        all_loss.append(total_loss/count)
        #print(f"Epoch {episodes} loss:", total_loss)

        episodes += 1

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        time_tot += execution_time

        #if episodes % 100 == 0:
        print(f"Episode {episodes}, loss={(total_loss/count):.4f} ")
        print(f"Temps d'execution: {execution_time:.2f}s, total: {time_tot:.2f}s")


    torch.save(model.state_dict(), "snake_model.pth")
    # --- courbe perte
    plt.plot(all_loss)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Evolution de la perte moyenne")
    plt.grid(True)
    plt.show()


def SaveTrainingPUCT(max_episode_min=5):

    """
        params = {
        
            "death": trial.suggest_float("death", -300, -50),
            "size": trial.suggest_float("size", 0.0, 5.0),
            "dist": trial.suggest_float("dist", 0.0, 20.0),
            "win": trial.suggest_float("win", 50, 200),
            "lose": trial.suggest_float("lose", -200, -50),
            "flood": trial.suggest_float("flood", -50, 0),
            "eat": trial.suggest_float("eat", 1.0, 50.0),
            "lose_part": trial.suggest_float("lose_part", -20, -1),
            "kill": trial.suggest_float("kill", 20, 150),
            "kill_dude": trial.suggest_float("kill_dude", -50, -10),
            'Cexplore': 0.5436079574741095
        }
    """
    

    params = {"Cexplore": 0.5436079574741095,
              'death': -148.42012305961117, 'size': 4.249118757407515, 'dist': 17.495777101115365, 'win': 142.95511897983494, 'lose': -68.94264997675857, 'flood': -21.891401061220208, 'eat': 49.930044240674235, 'lose_part': -6.9065559067686, 'kill': 99.14748219536332, 'kill_dude': -31.657405874965782
             }
        
 
    total_score = 0

    # 1. Cr�e un agent identique
    agent = DQNAgent(num_input=24, num_actions=4)  # adapte num_input / num_actions

    # 2. Charger les poids
    checkpoint = torch.load('checkpoint_snake_dqn_fc128topatt.pth', map_location=DEVICE)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])

    # 3. (Optionnel) mettre le r�seau en mode �valuation
    agent.q_net.eval()

    all_states = []
    all_targets = []

    MAX_EPISODE_T = time.perf_counter() + max_episode_min*60
    N = 0
    while time.perf_counter() < MAX_EPISODE_T:
    
        game = Game()
        game.init_Smitsimax()
        game.init_Training_PUCT()
        game.simulation.params = params

        
        turn = 0
        while True:
           
            total_score += game.step_IAOnlyFanSmitsimaxTrainingPUCT(agent)

            if len(game.snakes[0].body) < 3 or len(game.snakes[1].body) < 3  or turn == 200 or len(game.energy) == 0:
                break;

            turn += 1

        all_states.extend(game.states)
        all_targets.extend(game.policy)

        N += 1

        #if  N % 5 == 0:
        print("N finished= ", N)
    

    all_states = np.array(all_states, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    np.savez("dataset_snake_puct.npz", states=all_states, policies=all_targets)
    print(f"Saved dataset with {len(all_states)} samples.")

import torch
from torch.utils.data import Dataset, DataLoader

class PolicyDataset(Dataset):
    def __init__(self, states, policies):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.policies = torch.tensor(policies, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx]

def Training_PUCT(max_episode_min = 5):

    data = np.load("dataset_snake_puct.npz")
    states = data['states']
    policies = data['policies']

    print(states.shape, policies.shape)
      
    dataset = PolicyDataset(states, policies)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = PolicyNet(input_size=states.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    
    time_tot = 0
    MAX_EPISODE_T = time.perf_counter() + max_episode_min*60
    episodes = 0
    all_loss = []
    while time.perf_counter() < MAX_EPISODE_T:
        start_time = time.perf_counter()
        total_loss = 0
        count = 0
        for x, y in loader:
            pred = model(x)
            loss = -(y * F.log_softmax(pred, dim=1)).sum(dim=1).mean()
            #loss = -(y * torch.log(pred + 1e-8)).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            count +=1

        all_loss.append(total_loss/count)
        #print(f"Epoch {episodes} loss:", total_loss)

        episodes += 1

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        time_tot += execution_time

        if episodes % 50 == 0:
            print(f"Episode {episodes}, loss={(total_loss/count):.4f} ")
            print(f"Temps d'execution: {execution_time:.2f}s, total: {time_tot:.2f}s")


    torch.save(model.state_dict(), "policy.pth")
    # --- courbe perte
    plt.plot(all_loss)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Evolution de la perte moyenne")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    #main()

    paramsdqn = {'eat': 1.3593486093991836,
             'dist': 0.2636660780370111,
             'space': 0.0030769617169552027,
             'danger': 0.09326886700397304, 
             'death': -0.5778597016284167, 
             'kill': 1.0721079654935775, 
             'win': 1.8128625974373134, 
             'lose': -1.26584076466265}

    #agent = DQNAgent(24, 4)
    #Train_DQN_FC(agent, paramsdqn, max_episode_min=15, max_steps=200)
        
    #mainIA()


    params1 = {'death': -148.42012305961117, 'size': 4.249118757407515, 'dist': 17.495777101115365, 'win': 142.95511897983494, 
              'lose': -68.94264997675857, 'flood': -21.891401061220208, 'eat': 49.930044240674235, 'lose_part': -6.9065559067686, 
              'kill': 99.14748219536332, 'kill_dude': -31.657405874965782, 
              'Cexplore': 0.5436079574741095}

    params2 = {'death': -61.93605385655409, 'size': 4.881137038784745, 'dist': 12.48810009970582, 
     'win': 192.94434651075196, 'lose': -190.57311227738293, 'flood': -49.99868548591225, 
     'eat': 33.90698313151979, 'lose_part': -19.984215693570313, 'kill': 147.96318546156164, 'kill_dude': -10.693103912807167,
     'Cexplore': 0.6544837330252293
     }

    params = {'death': -267.0341828063319, 'size': 3.2065042230432246, 'dist': 14.68825946000775,
             'win': 185.74016840752748, 'lose': -62.4762792166449, 'flood': -32.36377820117928, 
             'eat': 45.70409988606824, 'lose_part': -17.739604178471264, 'kill': 73.46010334560454, 'kill_dude': -12.231499714189257,
             'Cexplore': 0.5436079574741095}

    #mainSmVN(params)


    #study = optuna.create_study(direction="maximize")
    #study.optimize(objectiveS, timeout=900)  # 15 min
    
    #print("BEST:", study.best_params)

    mainSmB(params)

    #SaveTrainingValueNet()
    #Training_Value_Net()

    #SaveTrainingPUCT(5)
    #Training_PUCT(10)
    
    #mainSmPUCT(params)