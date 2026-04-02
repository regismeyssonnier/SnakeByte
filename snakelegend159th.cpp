#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
#pragma GCC target("movbe,avx,avx2,fma,sse4.2,popcnt,bmi,bmi2,lzcnt")  //Enable AVX
#include <x86intrin.h> //AVX/SSE Extensions
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <math.h>
#include <ctime>
#include <chrono>
#include <map>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <queue>
#include <stack>
#include <cstring>
#include <unordered_set>
#include <string>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <random>
#include <array>
using namespace std::chrono;
using namespace std;




struct Pos{
    int8_t x;
    int8_t y;
};

enum Dir{
    UP,
    DOWN,
    LEFT,
    RIGHT
};

struct Params0 {
    double death      = -148.42012305961117;
    double size       =  4.249118757407515;
    double dist       = 17.495777101115365;
    double win        =142.95511897983494;
    double lose       = -68.94264997675857;
    double flood      = -21.891401061220208;
    double eat        =  49.930044240674235;
    double lose_part  =  -6.9065559067686;
    double kill       =  99.14748219536332;
    double kill_dude  = -31.657405874965782;
    double cexplore   = 0.5436079574741095; 
};

struct Params1 {
    double death     = -61.93605385655409;
    double size      = 4.881137038784745;
    double dist      = 12.48810009970582;
    double win       = 192.94434651075196;
    double lose      = -190.57311227738293;
    double flood     = -49.99868548591225;
    double eat       = 33.90698313151979;
    double lose_part = -19.984215693570313;
    double kill      = 147.96318546156164;
    double kill_dude = -10.693103912807167;
    double cexplore  = 0.5436079574741095; //0.6544837330252293;
};

struct Params {
    double death = -267.0341828063319;
    double size = 3.2065042230432246;
    double dist = 14.68825946000775;

    double win = 185.74016840752748;
    double lose = -62.4762792166449;
    double flood = -32.36377820117928;

    double eat = 45.70409988606824;
    double lose_part = -17.739604178471264;

    double kill = 73.46010334560454;
    double kill_dude = -12.231499714189257;

    double cexplore = 0.5436079574741095;
};



const int OPP[4] = {1,0,3,2};

const string direction[4] = {"UP", "DOWN", "LEFT", "RIGHT"};

const int DX[4] = {0,0,-1,1};
const int DY[4] = {-1,1,0,0};

const int MAX_BODY = 256;

struct Snake{

    int id;
    bool alive;

    int dir;

    int head;
    int tail;
    int len;

    Pos body[MAX_BODY];

};

inline Pos headPos(const Snake &s){
    return s.body[s.head];
}

inline Pos tailPos(const Snake &s){
    return s.body[s.tail];
}


const int MAX_W = 45;
const int MAX_H = 30;
const int BORDER = 0;
const int BORDERH = 5;


uint64_t Z[MAX_BODY][MAX_W+BORDER * 2][MAX_H+BORDERH];  // table Zobrist pour un snake

void initZobrist() {
    std::mt19937_64 rng(42);  // seed fixe pour reproductibilité
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    for(int i=0;i<MAX_BODY;i++)
        for(int x=0;x<MAX_W+BORDER*2;x++)
            for(int y=0;y<MAX_H+BORDERH;y++)
                Z[i][x][y] = dist(rng);
}

uint64_t hashSnake(const Snake &snake) {
    uint64_t h = 0;
    for(int k=0; k<snake.len; k++) {
        int idx = (snake.head - k + MAX_BODY) % MAX_BODY;
        Pos p = snake.body[idx];
        h ^= Z[k][p.x][p.y];
    }
    if(!snake.alive)
        h ^= 0xdeadbeefdeadbeefULL;  // marquer snake mort
    return h;
}


// vector de hashset par snake
std::vector<std::unordered_set<uint64_t>> visitedHashes(4);

// fonction pour ajouter un hash
void addHash(int snakeId, uint64_t h) {
    visitedHashes[snakeId].insert(h);
}

// vérifier si un hash existe
bool hasVisited(int snakeId, uint64_t h) {
    return visitedHashes[snakeId].count(h) > 0;
}

// reset avant un nouveau beam search
void clearHashes() {
    for(int i=0;i<4;i++)
        visitedHashes[i].clear();
}

enum Cell{
    EMPTY  = 0,
    WALL   = 1<<0,
    SNAKE  = 1<<1,
    ENERGY = 1<<2,
    GENERGY = 1<<3
};

struct Grid{

    uint8_t cell[MAX_H+BORDERH][MAX_W+BORDER * 2];

};

const int MAX_SNAKES = 8;
const int MAX_POWER  = 400;

struct GameState{

    int w,h;

    Grid gridu[8];

    Grid grid;

    int snakeCount;
    Snake snakes[8];

    int energyCount;
    Pos energy[MAX_POWER];
   
};

struct Node{
    int parent;

    int first_child;
    int child_count;

    double score = 0.0;
    int visits = 0;
    int move;
    double mult = 1.0;
    double prior = 0.0;
    GameState game;
    int turn = 0;
  
};

struct NodeB : Node{
    GameState game;
    Pos wallenergy={-1, -1};
};



const int MAX_NODE = 20000;
const int MAX_CHILD = 20000;

struct SM{

    Node nodes[MAX_NODE];
    NodeB nodesb[10];
    int children[MAX_CHILD];

    int ldir[8];

    int nodeCount = 0;
    int childCount = 0;

    int ITER = 0;

    Params params;

    int createNode(int parent = -1){

        int id = nodeCount;
        nodeCount = nodeCount + 1;
               
        nodes[id].parent = parent;
        nodes[id].first_child = -1;
        nodes[id].child_count = 0;
        nodes[id].score = 0.0;
        nodes[id].visits = 0;
        nodes[id].move = UP;
        nodes[id].mult = 1.0;
 
        return id;
    }
   

    void addChild(int parent, int child){

        if(nodes[parent].first_child == -1)
            nodes[parent].first_child = childCount;

        children[childCount++] = child;
        nodes[parent].child_count++;
    }
    
    

   
       
    inline int move_cost(int dir) {
        //if(dir <= 1)return 2;
        return 1; // ou autre si tu veux pondérer
    }

    vector<vector<vector<int>>> compute_distance_mapgo(
        int W, int H,
        const Grid &grid,
        const Pos energy[400],
        int max_energy
    ) {
        vector<vector<vector<int>>> dm(W*H); // FIX

        for (int ie = 0; ie < max_energy; ++ie) {

            int sx = energy[ie].x;
            int sy = energy[ie].y;

            vector<vector<int>> dist(H, vector<int>(W, 1e9));

            priority_queue<
                tuple<int,int,int>,
                vector<tuple<int,int,int>>,
                greater<>
            > pq;

            dist[sy][sx] = 0;
            pq.emplace(0, sx, sy);

            while (!pq.empty()) {
                auto [d, x, y] = pq.top();
                pq.pop();

                if (d != dist[y][x]) continue;

                for (int i = 0; i < 4; ++i) {
                    int nx = x + DX[i];
                    int ny = y + DY[i];

                    if (nx < 0 || ny < 0 || nx >= W || ny >= H)
                        continue;

                    if (grid.cell[ny][nx] == WALL)
                        continue;

                    if (grid.cell[ny][nx] == ENERGY)
                        continue;

                    int nd = d + move_cost(i);

                    if (nd < dist[ny][nx]) {
                        dist[ny][nx] = nd;
                        pq.emplace(nd, nx, ny);
                    }
                }
            }

            dm[sy*W+sx] = std::move(dist); // important (perf)
        }

        return dm;
    }

    vector<vector<vector<int>>> compute_distance_mapg(
        int W, int H,
        const Grid &grid,
        const Pos energy[400],
        int max_energy,
        vector<vector<vector<pair<int,int>>>> &parentg // 🔥 ajouté
    ) {
        vector<vector<vector<int>>> dm(W * H);
        parentg.clear();
        parentg.resize(W * H); // 🔥 allocation

        for (int ie = 0; ie < max_energy; ++ie) {

            int sx = energy[ie].x;
            int sy = energy[ie].y;

            vector<vector<int>> dist(H, vector<int>(W, 1e9));
            vector<vector<pair<int,int>>> parent(H, vector<pair<int,int>>(W, {-1,-1}));

            priority_queue<
                tuple<int,int,int>,
                vector<tuple<int,int,int>>,
                greater<>
            > pq;

            dist[sy][sx] = 0;
            pq.emplace(0, sx, sy);

            while (!pq.empty()) {
                auto [d, x, y] = pq.top();
                pq.pop();

                if (d != dist[y][x]) continue;

                for (int i = 0; i < 4; ++i) {
                    int nx = x + DX[i];
                    int ny = y + DY[i];

                    if (nx < 0 || ny < 0 || nx >= W || ny >= H)
                        continue;

                    if (grid.cell[ny][nx] == WALL)
                        continue;

                    if (grid.cell[ny][nx] == ENERGY)
                        continue;

                    int nd = d + move_cost(i);

                    if (nd < dist[ny][nx]) {
                        dist[ny][nx] = nd;

                        // 🔥 on stocke le parent
                        parent[ny][nx] = {x, y};

                        pq.emplace(nd, nx, ny);
                    }
                }
            }

            int id = sy * W + sx;
            dm[id] = std::move(dist);
            parentg[id] = std::move(parent); // 🔥 stockage parent
        }

        return dm;
    }

    vector<Pos> reconstruct_path(
        int tx, int ty,
        const vector<vector<pair<int,int>>> &parent
    ) {
        vector<Pos> path;

        int x = tx, y = ty;

        while (x != -1 && y != -1) {
            path.push_back({x, y});
            auto [px, py] = parent[y][x];
            x = px;
            y = py;
        }

        reverse(path.begin(), path.end());
        return path;
    }

    
    void playMoveTurn(GameState &g, double score[8], int opp_len){

        int total = g.snakeCount + opp_len;

        vector<pair<int, int>> ven;

        for(int id = 0; id < total; ++id){


            Snake &s = g.snakes[id];

            if(!s.alive)continue;

            Pos h = s.body[s.head];
            int move = s.dir;
            int nx = h.x + DX[move];
            int ny = h.y + DY[move];
                
            // --- MOVE NORMAL
            s.tail = (s.tail + 1) % MAX_BODY;
            s.head = (s.head + 1) % MAX_BODY;
            s.body[s.head] = {(int8_t)nx,(int8_t)ny};

            // --- ENERGY
            if (g.grid.cell[ny][nx] & ENERGY) {
                s.tail = (s.tail - 1 + MAX_BODY) % MAX_BODY;
                ven.push_back({nx, ny});
                s.len++;
                score[id] += params.eat; //10
            }
            /*else if (g.grid.cell[ny][nx] & WALL) {
                s.head = (s.head - 2 + MAX_BODY) % MAX_BODY;
                s.len--;
                if(s.len < 3)s.alive = false;
                score[id] += params.lose_part; //10
            }*/



        }

        for(int i = 0;i < ven.size();++i){
            int nx = ven[i].first, ny = ven[i].second;
            g.grid.cell[ny][nx] &= ~ENERGY;
        }

        for(int id = 0; id < total; ++id){
            // --- CHECK COLLISION AVANT MOVE
            bool collision = false;

            Snake &s = g.snakes[id];
            if(!s.alive)continue;


            Pos h = s.body[s.head];
            int nx = h.x, ny = h.y;
            int ind_other = -1;
            bool head = false;
            for(int j = 0; j < total; ++j){
                int end = 0;
                if(id == j)end = 1;
                Snake &other = g.snakes[j];
                if(!other.alive)continue;

                int cur = other.tail;
                for(int k = 0; k < other.len-end; ++k){
                    Pos b = other.body[cur];
                    if(b.x == nx && b.y == ny){
                        ind_other = j;
                        if(k == other.len -1)head = true;
                        collision = true;
                        break;
                    }
                    cur = (cur + 1) % MAX_BODY;
                }
                if(collision) break;
            }

            if(collision){
                // 👉 rollback : la tête devient segment précédent
                int prev = (s.head - 1 + MAX_BODY) % MAX_BODY;
                s.head = prev;
                s.len--; // shrink
                if (s.len < 3)
                    s.alive = false;
                else
                    score[id] += params.lose_part;
                if(head){
                    Snake &so = g.snakes[ind_other];
                    int prev = (so.head - 1 + MAX_BODY) % MAX_BODY;
                    so.head = prev;
                    so.len--; // shrink
                    if (so.len < 3){
                        so.alive = false;
                        if(s.alive && id < g.snakeCount && ind_other >= g.snakeCount)score[id] += params.kill; 
                        if(s.alive && id < g.snakeCount && ind_other < g.snakeCount)score[id] += params.kill_dude;   
                        if(s.alive && id >= g.snakeCount && ind_other >= g.snakeCount)score[id] += params.kill_dude; 
                        if(s.alive && id >= g.snakeCount && ind_other < g.snakeCount)score[id] += params.kill;  
                    }
                    else if(s.len > so.len){
                        score[id] += params.lose_part * -2.0;
                    }
                    
                    
                }

                
            }
            
            
        }

        

        /*while(true){

            Grid grcurp = g.grid;

            for(int is =  0;is < g.snakeCount+opp_len;++is){
                for(int ids = 0; ids < g.snakes[is].len; ids++){
                    int id = (g.snakes[is].tail + ids) % MAX_BODY;
                    Pos p = g.snakes[is].body[id];

                    if(p.y >=0 && p.y < g.h && p.x >=0 && p.x < g.w)
                        grcurp.cell[p.y][p.x] |= SNAKE;
                }

            }

            bool stop = true;

            for(int i  = 0;i < total;++i){
                int fall = applyGravity(g, g.snakes[i], grcurp);
                if(fall != INT_MAX){
                    stop = false;
                }
            }

            if(stop)break;

        }*/

        doFalls2(g, total);
        /*
        bool needIntercoil = false;

        for(int i=0;i<total;i++){
            for(int j=i+1;j<total;j++){
                if(touchingVertical2(g.snakes[i], g.snakes[j])){
                    needIntercoil = true;
                    break;
                }
            }
            if(needIntercoil) break;
        }

        if(needIntercoil){
            doIntercoiledFalls(g, total);
        }*/
                
        
    }

    int getDirection(const Snake &s) {
 
        Pos head = s.body[s.head];
        int next = (s.head - 1 + MAX_BODY) % MAX_BODY;
        Pos neck = s.body[next];

        int dx = head.x - neck.x;
        int dy = head.y - neck.y;

        if (dx == 0 && dy == -1) return 0; // UP
        if (dx == 0 && dy ==  1) return 1; // DOWN
        if (dx == -1 && dy == 0) return 2; // LEFT
        if (dx ==  1 && dy == 0) return 3; // RIGHT

        return -1;
    }

    vector<vector<vector<int>>> distg;
    vector<vector<vector<pair<int,int>>>> parentg_sim;

    std::vector<int> expandBeam(const Grid &grcurp, Snake &snake, int wt, int ht) {

        std::vector<int> moves;

        // head
        Pos h = snake.body[snake.head];

        for (int i = 0; i < 4; i++) {

            int nx = h.x + DX[i];
            int ny = h.y + DY[i];

            // hors map
            if (nx < 0 || ny < 0 || nx >= wt || ny >= ht)
                continue;

            // mur
            if (grcurp.cell[ny][nx] & WALL)
                continue;

            // règle verticale
            if (i == 0 && grcurp.cell[ny][nx] == EMPTY && isVertical(snake))
                continue;

            // éviter revenir sur le cou
            int curid = (snake.head - 1 + MAX_BODY) % MAX_BODY;
            Pos cur = snake.body[curid];
            if (nx == cur.x && ny == cur.y)
                continue;

            // priorité énergie
            if (grcurp.cell[ny][nx] & ENERGY) {
                return {i};  // retour direct
            }

            moves.push_back(i);
        }

        return moves;
    }

    std::vector<int> expandBeamP(const Grid &grcurp, Snake &snake, int wt, int ht) {

        std::vector<int> moves;

        // head
        Pos h = snake.body[snake.head];

        for (int i = 0; i < 4; i++) {

            int nx = h.x + DX[i];
            int ny = h.y + DY[i];

            // hors map
            if (nx < 0 || ny < 0 || nx >= wt || ny >= ht)
                continue;

            // mur
            if (grcurp.cell[ny][nx] & WALL)
                continue;

            // règle verticale
            if (i == 0 && grcurp.cell[ny][nx] == EMPTY && isVertical(snake))
                continue;

            // éviter revenir sur le cou
            int curid = (snake.head - 1 + MAX_BODY) % MAX_BODY;
            const Pos &cur = snake.body[curid];
            if (nx == cur.x && ny == cur.y)
                continue;

            moves.push_back(i);
        }

        return moves;
    }

    double evaluation(int ind_snake,
                  GameState &game,
                  double score[8],
                  int totalSnake,
                  int power_source_count,
                  //const std::vector<std::vector<int>> &distg,
                  //const vector<std::vector<std::vector<std::pair<int,int>>>> &parentg,
                  int width,
                  int height,Pos energy[MAX_POWER],  double &end) 
    {
        int tot_team = 0;
        int tot_opp = 0;

        // --- compter vivants
        for (int i = 0; i < totalSnake; i++) {
            const Snake &s = game.snakes[i];
            if (s.alive) {
                if (i < game.snakeCount) tot_team++;
                else tot_opp++;
            }
        }

        // --- copy grid
        Grid grcurp = game.grid;

        // --- mark snakes
        for (int i = 0; i < totalSnake; i++) {
            const Snake &s = game.snakes[i];
            for (int k = 0; k < s.len; k++) {
                int idx = (s.tail + k) % MAX_BODY;
                const Pos &p = s.body[idx];

                if (p.y >= 0 && p.y < game.h && p.x >= 0 && p.x < game.w) {
                    grcurp.cell[p.y][p.x] |= SNAKE;
                }
            }
        }

        int i = ind_snake;
        const Snake &s = game.snakes[i];

        int hx = s.body[s.head].x;
        int hy = s.body[s.head].y;

        double sc = score[i];

        // --- mort
        if (!s.alive) {
            return params.death;
        }

        // --- taille
        sc += s.len * params.size;

        // --- distance énergie
        double bestDist = 1e9;
        int count = 0;
        double sumScore = 0.0;

        if (hx >= 0 && hx < game.w && hy >= 0 && hy < game.h) {

            for (int ind = 0; ind < power_source_count; ind++) {
                const Pos &e = energy[ind];

                if (game.grid.cell[e.y][e.x] & ENERGY) {

                    int gap, di=0;
                    gap = compute_max_gap(
                        hx, hy,
                        parentg_sim[e.y * width + e.x],
                        grcurp,
                        game.w, game.h, di
                    );

                    if (gap > s.len - 2)
                        gap += 1000;

                    double d = di + gap;

                    if (d < bestDist)
                        bestDist = d;

                    count++;

                    sumScore += params.dist * 10.0 / (d + 1.0);
                }
            }

            // combo
            if (bestDist < 1e9) {
                sc += params.dist * 30.0 / (bestDist + 1.0);
                sc += 0.3 * sumScore;
            }
        }

        // --- plus d'énergie
        if (count == 0) {
            if (i < game.snakeCount) {
                sc += (tot_team > tot_opp) ? params.win : params.lose;
                end = (tot_team > tot_opp) ? 5000.0 : 0;
            } else {
                sc += (tot_opp > tot_team) ? params.win : params.lose;
                
            }
            end = true;
        }

        // --- flood fill
        int fl = floodFill(hx, hy, grcurp, game, std::min(4, s.len));
        if (fl < std::min(4, s.len)) {
            sc += params.flood;
        }

        return sc;
    }

    int getWidth(int depth) {
        int minW = 8;
        double decay = 0.85; // facteur de décroissance

        int w = (int)(1024 * pow(decay, depth));
        return std::max(minW, w);
    }

    double policyScore(const GameState& game, int player, int m, Pos energy[MAX_POWER]) {
        const Snake& s = game.snakes[player];
        Pos h = s.body[s.head];

        int nx = h.x + DX[m];
        int ny = h.y + DY[m];

        double score = 0.0;

        // --- aller vers énergie
        for(int i = 0; i < game.energyCount; i++) {
            Pos e = energy[i];
            if(game.grid.cell[e.y][e.x] & ENERGY) {
                //int d = abs(nx - e.x) + abs(ny - e.y);
                //score += 10.0 / (d + 1);
                score += 10.0 / (distg[e.y*game.w+e.x][ny][nx] + 1);
                
                if (nx == e.x && ny == e.y)score += 1000.0;
            }
        }

        // --- éviter murs
        if(nx <= 1 || ny <= 1 || nx >= game.w-2 || ny >= game.h-2)
            score -= 5.0;

        // --- éviter cul-de-sac
        int free = get_safe_moves(nx, ny, game.grid, game.w, game.h);
        score += free * 2.0;

        // --- stabilité direction
        //if(m == s.dir)
        //    score += 1.5;

        return score;
    }

    string Play(int height, int width, Grid grid, Snake my_snake[8], int my_snake_len, Snake opp_snake[8], int opp_snake_len, int my_id_snake[8], Pos energy[MAX_POWER], int power_source_count, int time, int TURN){

        auto startm = high_resolution_clock::now();;
        int maxt = 0;
        auto getTime = [&]()-> bool {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - startm);
            //cerr << duration.count() << endl;
            maxt = duration.count();
            return(maxt <= time);
        };

        nodeCount = 0;
        childCount= 0;

        int bestMove[4] = {-1, -1, -1, -1};
        double bestScore[4]= {-1e9, -1e9, -1e9, -1e9};


        int totalSnake = my_snake_len + opp_snake_len;

        int WIDTHB = 512;
        

        int beam[4][WIDTHB];
        int TS[4][WIDTHB];
        int maxTS[4];

        for (int i = 0; i < 4; i++)
            maxTS[i] = 1;

        // --- init root ---
        for (int player = 0; player < my_snake_len; player++) {
            int root = createNode(-1);
            beam[player][0] = root;

            Node &n = nodes[root];
            n.turn = TURN;

            GameState &game = n.game;
            game.w = width;
            game.h = height + BORDERH;
            game.snakeCount = my_snake_len;

            // my snakes
            for (int i = 0; i < my_snake_len; i++) {
                game.snakes[i] = my_snake[i];
                game.snakes[i].dir = UP;
                game.snakes[i].alive = true;
            }

            // opp snakes
            for (int i = 0; i < opp_snake_len; i++) {
                game.snakes[my_snake_len + i] = opp_snake[i];
                game.snakes[my_snake_len + i].dir = UP;
                game.snakes[my_snake_len + i].alive = true;
            }

            // energy
            for (int i = 0; i < power_source_count; i++)
                game.energy[i] = energy[i];

            game.grid = grid;
            game.energyCount = power_source_count;
        }

        clearHashes();

        int depth = -1;
        bool stop = false;

        int MAX_DEPTH = max(20, 200-TURN);
        int collision = 0;

        while (getTime() && depth < MAX_DEPTH) {
            //WIDTHB = getWidth(depth);
            if (depth % 5 == 0)WIDTHB -= 50;
            depth++;
            
            //cerr << depth << endl;
            

            for (int player = 0; getTime() &&player < my_snake_len; player++) {

                int ind_beam = 0;
                int count = 0;

                //cerr << "sz=" << maxTS[ind_beam] << endl;

                while (getTime() && ind_beam < maxTS[player]) {

                    int node_idx = beam[player][ind_beam];
                    Node &node = nodes[node_idx];
                    GameState &game = node.game;

                    std::vector<int> moves = expandBeamP(game.grid, game.snakes[player], game.w, game.h);

                    if (moves.empty()) {
                        ind_beam++;
                        continue;
                    }
                    
                    // --- autres joueurs random / greedy ---
                    for (int iplayer = 0; iplayer < totalSnake; iplayer++) {
                        if(!getTime())break;
                        if (iplayer == player) continue;
                        if (!game.snakes[iplayer].alive) continue;

                        std::vector<int> moveo = expandBeam(game.grid, game.snakes[iplayer], game.w, game.h);

                        if (!moveo.empty()){
                            std::vector<std::pair<double,int>> scored;

                            for(int m : moveo) {
                                double ps = policyScore(game, iplayer, m, energy);
                                scored.emplace_back(ps, m);
                            }

                            std::sort(scored.begin(), scored.end(),
                                [](auto &a, auto &b){
                                    return a.first > b.first;
                                });

                            /*int dir = -1;
                            if (!moveo.empty())
                                dir = moveo[rand() % moveo.size()];
                            else
                                dir = getDirection(game.snakes[iplayer]);*/
                                
                            game.snakes[iplayer].dir = scored[0].second;

                        }
                        else{
                             game.snakes[iplayer].dir = getDirection(game.snakes[iplayer]);
                        }
                    }

                    std::vector<std::pair<double,int>> scored;

                    for(int m : moves) {
                        double ps = policyScore(game, player, m, energy);
                        scored.emplace_back(ps, m);
                    }

                    std::sort(scored.begin(), scored.end(),
                        [](auto &a, auto &b){
                            return a.first > b.first;
                        });

                    int TOPK = 3;
                    if(depth > 0 && scored[0].first > scored[1].first + 1000.0)
                        TOPK--; // greedy direct

                    if(depth > 5)TOPK--;

                    // --- expansion ---
                    for(int k = 0;k < TOPK && k < scored.size();++k){
                    //for (int m : moves) {
                        int m = scored[k].second;
                        if(!getTime())break;
                        Pos h = game.snakes[player].body[game.snakes[player].head];

                        int child = createNode(node_idx);
                        addChild(node_idx, child);

                        if(nodeCount == MAX_NODE)break;

                        int hx = h.x + DX[m];
                        int hy = h.y + DY[m];    
                        
                        nodes[child].turn = node.turn + 1;

                        if (depth == 0)
                            nodes[child].move = m;
                        else
                            nodes[child].move = node.move;

                        double add_energie = 0.0;
                        bool is_energy =false;
                        if (game.grid.cell[hy][hx] & ENERGY){
                            nodes[child].mult = params.eat;
                            is_energy = true;
                            add_energie = 1000.0;
                            if (depth == 0)add_energie= 100000.0;
                        }

                        // --- copy game ---
                        GameState &gamech = nodes[child].game;
                        gamech.w = width;
                        gamech.h = height + BORDERH;

                        gamech.snakeCount = my_snake_len;

                        for (int i = 0; i < my_snake_len; i++)
                            gamech.snakes[i] = game.snakes[i];

                        gamech.snakes[player].dir = m;

                        for (int i = 0; i < opp_snake_len; i++)
                            gamech.snakes[my_snake_len + i] = game.snakes[my_snake_len + i];

                        //for (int i = 0; i < power_source_count; i++)
                        //    gamech.energy[i] = energy[i];

                        gamech.grid = game.grid;
                        gamech.energyCount = power_source_count;

                        // --- simulate ---
                        double score[8] = {0};
                        playMoveTurn(gamech, score, opp_snake_len);

                        double end = 0;
                        double sc = evaluation(player, gamech, score,
                                            totalSnake, power_source_count
                                            , game.w, game.h, energy, end);

                        nodes[child].score = node.score + sc + add_energie + end;

                        uint64_t hs = hashSnake(gamech.snakes[player]);
                        bool not_visited = false;
                        if(!hasVisited(player, hs)) {
                            addHash(player, hs);
                            not_visited = true;
                            
                            // ajouter le node dans le beam
                        }
                        else{
                            collision++;
                        }

                        if(not_visited){
                            TS[player][count++] = child;
                            
                        }

                        if (nodes[child].score > bestScore[player]){
                            bestScore[player] = nodes[child].score;
                            bestMove[player] = nodes[child].move;
                        }

                        if (count >= WIDTHB) break;
                    }

                    if(nodeCount == MAX_NODE)break;
                    if (count >= WIDTHB) break;

                    ind_beam++;
                }

                if(!getTime())break;
                if(nodeCount >= MAX_NODE)break;

                maxTS[player] = count;

                // --- sort ---
                std::sort(TS[player], TS[player] + count,
                    [&](int a, int b) {
                        return nodes[a].score > nodes[b].score;
                    });

                // --- copy to beam ---
                for (int i = 0; i < maxTS[player]; i++)
                    beam[player][i] = TS[player][i];

                
                
            }

            stop = true;
            for(int i = 0;i < my_snake_len;++i){
                if(maxTS[i] > 0){
                    stop = false;
                    break;
                }
            }
            
            if(stop)break;

            if(nodeCount == MAX_NODE)break;
        }

                
      
        cerr <<"TURN=" << depth << " " << "node=" <<  nodeCount << ", child=" << childCount << ", collision=" << collision << endl;

        string ans;

        //direction
        for(int i = 0;i < my_snake_len;++i){
            int indc = -1;
            double maxi = -2e9;
            
            cerr << my_id_snake[i] << endl;
            indc = bestMove[i];
            

            if (indc != -1){
                ans += to_string(my_id_snake[i]) + " " + direction[indc] + ";";
            }
            else{
                ans += "WAIT;";
            }

        }

        if (ans.empty())cerr << "empty ans" << endl;
        
        return ans;


    }

    int compute_max_gap(
        int tx, int ty,
        const vector<vector<pair<int,int>>> &parent,
        const Grid &grid,
        int W, int H, int &d
    ) {
        int max_gap = 0;
        int current_gap = 0;

        int x = tx, y = ty;

        while (x != -1 && y != -1) {
            d++;
            int ny = y + 1;

            bool no_support = false;

            if (ny >= H) {
                no_support = true;
            } else {
                if (!(grid.cell[ny][x] & (WALL | ENERGY | SNAKE))) {
                    no_support = true;
                }
            }

            if(grid.cell[y][x] & SNAKE){
                d += 5;
                //break;
            }

            if (no_support) {
                current_gap++;
                if (current_gap > max_gap)
                    max_gap = current_gap;
            } else {
                current_gap = 0; // 🔥 reset dès qu’il y a support
            }

            auto [px, py] = parent[y][x];
            x = px;
            y = py;
        }

        return max_gap;
    }

    bool somethingSolidUnder(GameState &g, int x, int y,
                         const unordered_set<long long> &meta,
                         int total) {

        int ny = y + 1;

        // --- hors map = solide
        if(x < 0 || x >= g.w || ny < 0)
            return true;

        // --- sol
        if(ny >= g.h)
            return true;

        // --- ignore groupe (clé = x + y*W)
        long long key = (long long)x << 32 | ny;
        if(meta.count(key))
            return false;

        // --- mur
        if(g.grid.cell[ny][x] & WALL)
            return true;

        // --- énergie
        if(g.grid.cell[ny][x] & ENERGY)
            return true;

        // --- autres snakes
        for(int i = 0; i < total; i++){
            Snake &s = g.snakes[i];
            if(!s.alive) continue;

            for(int k = 0; k < s.len; k++){
                int idx = (s.tail + k) % MAX_BODY;
                Pos p = s.body[idx];

                if(p.x == x && p.y == ny)
                    return true;
            }
        }

        return false;
    }

    bool touchingVertical(const Snake &a, const Snake &b){
        for(int i = 0; i < a.len; i++){
            int id1 = (a.tail + i) % MAX_BODY;
            Pos p1 = a.body[id1];

            for(int j = 0; j < b.len; j++){
                int id2 = (b.tail + j) % MAX_BODY;
                Pos p2 = b.body[id2];

                if(p1.x == p2.x && abs(p1.y - p2.y) == 1)
                    return true;
            }
        }
        return false;
    }

    vector<vector<int>> getGroupsor(GameState &g, int total){
        vector<vector<int>> groups;
        vector<bool> visited(total, false);

        for(int i = 0; i < total; i++){
            if(visited[i] || !g.snakes[i].alive) continue;

            vector<int> group;
            stack<int> st;
            st.push(i);

            while(!st.empty()){
                int u = st.top(); st.pop();
                if(visited[u]) continue;

                visited[u] = true;
                group.push_back(u);

                for(int v = 0; v < total; v++){
                    if(v == u || visited[v]) continue;
                    if(!g.snakes[v].alive) continue;

                    if(touchingVertical(g.snakes[u], g.snakes[v]))
                        st.push(v);
                }
            }

            groups.push_back(group);
        }

        return groups;
    }

    vector<vector<int>> getGroups(GameState &g, int total) {
        vector<vector<int>> groups;
        vector<bool> visited(total, false);

        for(int i = 0; i < total; ++i) {
            Snake &si = g.snakes[i];
            if(visited[i] || !si.alive) continue;

            vector<int> group;
            stack<int> st;
            st.push(i);

            while(!st.empty()) {
                int u = st.top(); st.pop();
                if(visited[u]) continue;

                visited[u] = true;
                group.push_back(u);

                Snake &su = g.snakes[u];
                for(int v = 0; v < total; ++v) {
                    if(visited[v] || v == u) continue;
                    Snake &sv = g.snakes[v];
                    if(!sv.alive) continue;

                    if(touchingVertical(su, sv)) {
                        st.push(v);
                    }
                }
            }

            groups.emplace_back(move(group)); // pas de copie
        }

        return groups;
    }

    void doFalls(GameState &g, int total){

        while(true){
            bool somethingFell = false;

            auto groups = getGroups(g, total);

            for(auto &grp : groups){

                // --- meta body
                unordered_set<long long> meta;

                for(int i : grp){
                    Snake &s = g.snakes[i];

                    for(int k = 0; k < s.len; k++){
                        int idx = (s.tail + k) % MAX_BODY;
                        Pos p = s.body[idx];

                        long long key = ((long long)p.x << 32) | (unsigned int)p.y;
                        meta.insert(key);
                    }
                }

                // --- check fall
                bool canFall = true;

                for(auto key : meta){
                    int x = key >> 32;
                    int y = (int)key;

                    if(somethingSolidUnder(g, x, y, meta, total)){
                        canFall = false;
                        break;
                    }
                }

                // --- apply fall
                if(canFall){
                    somethingFell = true;

                    for(int i : grp){
                        Snake &s = g.snakes[i];

                        for(int k = 0; k < s.len; k++){
                            int idx = (s.tail + k) % MAX_BODY;
                            s.body[idx].y += 1;
                        }

                        // --- death
                        bool dead = true;
                        for(int k = 0; k < s.len; k++){
                            int idx = (s.tail + k) % MAX_BODY;

                            if(s.body[idx].y < g.h + 1){
                                dead = false;
                                break;
                            }
                        }

                        if(dead)
                            s.alive = false;
                    }
                }
            }

            if(!somethingFell) break;
        }
    }

    bool isSnakeSupportedOrDead(GameState &g, Snake &s)
    {
        bool hasOutside = false;
        bool hasSupportInside = false;

        for(int k = 0; k < s.len; k++){
            int id = (s.tail + k) % MAX_BODY;
            Pos c = s.body[id];

            // sort sur les côtés
            /*if(c.x < 0 || c.x >= g.w){
                hasOutside = true;
                continue;
            }*/

            // dans la grille
            if(c.y >= 0 && c.y < g.h){

                // sol
                if(c.x >= 0 && c.x < g.w){
                    hasSupportInside = true;
                }
                else if(c.y + 1 >= 0 && c.y + 1 < g.h){
                    if(g.grid.cell[c.y + 1][c.x] & (WALL | ENERGY)){
                        hasSupportInside = true;
                    }
                }
            }
        }

        // 💀 condition de mort
        if(!hasSupportInside){
            return false;
        }

        return true;
    }

    void doFalls2(GameState &g, int total_snakes) {

        vector<int> fallDist(total_snakes, 0);
        vector<bool> alive(total_snakes, true);

        vector<int> airborne;
        vector<int> grounded;

        // init airborne = tous les snakes vivants
        for(int i = 0; i < total_snakes; i++) {
            if(g.snakes[i].alive)
                airborne.push_back(i);
        }

        while(true) {

            bool somethingFell = false;

            // --- propagation grounded ---
            bool changed = true;
            while(changed) {
                changed = false;

                for(int idx = 0; idx < (int)airborne.size(); idx++) {
                    int i = airborne[idx];
                    Snake &s = g.snakes[i];

                    bool isGrounded = false;

                    for(int k = 0; k < s.len; k++) {
                        int id = (s.tail + k) % MAX_BODY;
                        Pos c = s.body[id];

                        // sol
                        if(c.y + 1 >= g.h-1) {
                            isGrounded = true;
                            break;
                        }

                        // mur / pomme
                        if(g.grid.cell[c.y+1][c.x] & (WALL | ENERGY)) {
                            isGrounded = true;
                            break;
                        }

                        // touche snake grounded
                        for(int j : grounded) {
                            Snake &os = g.snakes[j];
                            for(int kk = 0; kk < os.len; kk++) {
                                int id2 = (os.tail + kk) % MAX_BODY;
                                Pos p2 = os.body[id2];

                                if(p2.x == c.x && p2.y == c.y+1) {
                                    isGrounded = true;
                                    break;
                                }
                            }
                            if(isGrounded) break;
                        }
                        if(isGrounded) break;
                    }

                    if(isGrounded) {
                        grounded.push_back(i);
                        airborne[idx] = airborne.back();
                        airborne.pop_back();
                        idx--;
                        changed = true;
                    }
                }
            }

            // --- chute ---
            for(int i : airborne) {
                Snake &s = g.snakes[i];

                bool canFall = true;

                for(int k = 0; k < s.len; k++){
                    int id_ = (s.tail + k) % MAX_BODY;
                    Pos &c = s.body[id_];

                    // touche le sol (hors grille)
                    if(c.y + 1 >= g.h-1){
                        canFall = false;
                        break;
                    }

                    // collision avec mur ou énergie
                    if(/*c.x >= 0 && c.x < g.w &&*/ c.y + 1 >= 0 && c.y + 1 < g.h){
                        if(g.grid.cell[c.y + 1][c.x] & (WALL | ENERGY)){
                            canFall = false;
                            break;
                        }
                    }
                }

                if(!canFall){
                    continue;
                }

                somethingFell = true;

                for(int k = 0; k < s.len; k++) {
                    int id = (s.tail + k) % MAX_BODY;
                    s.body[id].y += 1;
                }

                fallDist[i]++;

                /*if(!isSnakeSupportedOrDead(g, s)){
                    g.snakes[i].alive = false;
                    continue;
                }*/
                

                // out of bounds
                bool dead = true;
                for(int k = 0; k < s.len; k++) {
                    int id = (s.tail + k) % MAX_BODY;
                    if(s.body[id].y <= g.h ) {
                        dead = false;
                        break;
                    }
                }

                if(dead) {
                    s.alive = false;
                }
            }

            if(!somethingFell) break;
        }

        for(int i = 0; i < total_snakes; i++) {
            if(g.snakes[i].alive){
                // 💀 check support horizontal
                if(!isSnakeSupportedOrDead(g, g.snakes[i])){
                    g.snakes[i].alive = false;
                   
                }

            }
                
        }

    }

    bool touchingVertical2(const Snake &a, const Snake &b) {
        for(int i = 0; i < a.len; i++) {
            int id1 = (a.tail + i) % MAX_BODY;
            Pos p1 = a.body[id1];

            for(int j = 0; j < b.len; j++) {
                int id2 = (b.tail + j) % MAX_BODY;
                Pos p2 = b.body[id2];

                if(abs(p1.x - p2.x) == 0 && abs(p1.y - p2.y) == 1)
                    return true;
            }
        }
        return false;
    }

    vector<vector<int>> getIntercoiled(GameState &g, int total) {

        vector<vector<int>> groups;
        vector<bool> vis(total, false);

        for(int i = 0; i < total; i++) {
            if(vis[i] || !g.snakes[i].alive) continue;

            vector<int> group;
            queue<int> q;
            q.push(i);

            while(!q.empty()) {
                int u = q.front(); q.pop();
                if(vis[u]) continue;

                vis[u] = true;
                group.push_back(u);

                for(int v = 0; v < total; v++) {
                    if(u == v || vis[v]) continue;
                    if(!g.snakes[v].alive) continue;

                    if(touchingVertical2(g.snakes[u], g.snakes[v]))
                        q.push(v);
                }
            }

            if(group.size() > 1)
                groups.push_back(group);
        }

        return groups;
    }

    void doIntercoiledFalls(GameState &g, int total) {

        while(true) {

            bool fell = false;

            auto groups = getIntercoiled(g, total);

            for(auto &grp : groups) {

                bool canFall = true;

                for(int id : grp) {
                    Snake &s = g.snakes[id];

                    for(int k = 0; k < s.len; k++) {
                        int idx = (s.tail + k) % MAX_BODY;
                        Pos c = s.body[idx];

                        if(c.y + 1 >= g.h) {
                            canFall = false;
                            break;
                        }

                        if(g.grid.cell[c.y+1][c.x] & WALL) {
                            canFall = false;
                            break;
                        }
                    }

                    if(!canFall) break;
                }

                if(canFall) {
                    fell = true;

                    for(int id : grp) {
                        Snake &s = g.snakes[id];

                        for(int k = 0; k < s.len; k++) {
                            int idx = (s.tail + k) % MAX_BODY;
                            s.body[idx].y += 1;
                        }
                    }
                }
            }

            if(!fell) break;
        }
    }

    int get_safe_moves(int hx, int hy, const Grid &grid, int w, int h) {
         // tête du snake
        int count = 0;

        // DIRS : UP, DOWN, LEFT, RIGHT
        const int DX[4] = {0, 0, -1, 1};
        const int DY[4] = {-1, 1, 0, 0};

        for (int i = 0; i < 4; ++i) {
            int nx = hx + DX[i];
            int ny = hy + DY[i];

            // vérifier qu'on reste dans la grille
            if (nx < 0 || nx >= w || ny < 0 || ny >= h)
                continue;

            // vérifier collision mur ou snake
            if (grid.cell[ny][nx] & (WALL | SNAKE))
                continue;

            
            count++;
        }

        return count;
    }

    

    void backpropor(int node, double result){

        while(node != -1){

            nodes[node].visits++;
            nodes[node].score += result;

            node = nodes[node].parent;
        }
    }

    // Optimisation de backpropagation
    // nodes : tableau contigu de Node
    // node : indice de départ
    // result : score à propager
    void backprop(int start_node, double result) {
        int node = start_node;

        // Boucle simple sur l'arbre
        while (node != -1) {
            Node &n = nodes[node];

            // Accès direct aux champs pour réduire overhead
            n.visits += 1;
            n.score += result;

            // Préfetch du parent pour améliorer la localité mémoire
            int parent = n.parent;
            if (parent != -1) {
                __builtin_prefetch(&nodes[parent], 1, 3); // lecture + haute priorité
            }

            node = parent;
        }
    }

    
    int bestChildIndex(int rootNode) {
        int start = nodes[rootNode].first_child;
        int bestChild = -1;
        double bestScore = -1e30;

        for (int j = 0; j < nodes[rootNode].child_count; j++) {
            int c = children[start + j];  // l’indice réel de l’enfant
            double avg = nodes[c].visits ? nodes[c].score / (double)nodes[c].visits : 0;
            //avg *= nodes[c].mult;
            if (avg > bestScore) {
                bestScore = avg;
                bestChild = c;
            }

            cerr << nodes[c].move
            << " visits=" << nodes[c].visits
            << " score=" << nodes[c].score
            << " avg=" << fixed << setprecision(6)  << avg
            << " mult" << nodes[c].mult
            << endl;

            //cerr << j << "=" << fixed << setprecision(5) << avg << endl;
        }

        return bestChild; // retourne l’indice réel dans nodes[]
    }

    // Pour récupérer la direction finale
    int bestMove(int rootNode) {
        int c = bestChildIndex(rootNode);
        if(c != -1) return nodes[c].move; // utilise la move stockée
        return -1; // pas de coup valide
    }
 
     
    

    
    int applyGravity(GameState &g, Snake &s, Grid &gr)
    {
        int fall = INT_MAX;

        // retirer le snake de la grille
        /*for(int i = 0; i < s.len; i++){
            int id = (s.tail + i) % MAX_BODY;
            Pos p = s.body[id];
            if(p.y >=0 && p.y < g.h && p.x >=0 && p.x < g.w)
                gr.cell[p.y][p.x] &= ~SNAKE;
        }*/

        // calcul de la distance maximale de chute
        for(int i = 0; i < s.len; i++){
            int id = (s.tail + i) % MAX_BODY;
            Pos p = s.body[id];

            int d = 0;
            int y = p.y;

            while(true){
                y++;

                if(y >= g.h) break;

                //if(gr.cell[y][p.x] != EMPTY)
                //    break;

                if(gr.cell[y][p.x] & (WALL|SNAKE|ENERGY))
                    break;

                d++;
            }

            fall = std::min(fall, d);
        }

        if(fall == 0)
            return INT_MAX;

        // appliquer la chute
        for(int i = 0; i < s.len; i++){
            int id = (s.tail + i) % MAX_BODY;
            s.body[id].y += fall;
        }

        return fall;

        // remettre dans la grille
        /*for(int i = 0; i < s.len; i++){
            int id = (s.tail + i) % MAX_BODY;
            Pos p = s.body[id];

            if(p.y >=0 && p.y < g.h && p.x >=0 && p.x < g.w)
                gr.cell[p.y][p.x] |= SNAKE;
        }*/
    }


    int floodFill(int sx, int sy, Grid &gr, GameState &g, int limit = 30) {

        bool vis[50][50] = {};
        queue<pair<int,int>> q;

        q.push({sx, sy});
        vis[sy][sx] = true;

        int cnt = 0;

        while (!q.empty() && cnt < limit) {
            auto [x,y] = q.front(); q.pop();
            cnt++;

            for (int d = 0; d < 4; d++) {
                int nx = x + DX[d];
                int ny = y + DY[d];

                if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
                if (vis[ny][nx]) continue;
                if (gr.cell[ny][nx] & (WALL | SNAKE)) continue;

                vis[ny][nx] = true;
                q.push({nx, ny});
            }
        }

        return cnt;
    }

    
    Snake getSnake(int id, Snake my_snake[8], int len, int &ind){
        for(int i = 0;i < len;++i){
            if(my_snake[i].id == id){
                ind = i;
                return my_snake[i];
            }
        }


    }

    bool isVertical(Snake &s){

        int x0 = s.body[s.head].x;

        for(int i=0;i<s.len;i++){

            int id = (s.tail + i) % MAX_BODY;

            if(s.body[id].x != x0)
                return false;
        }

        return true;
    }


};


void parseGrid(Grid &grid, int height, int width) {
    for(int y = 0; y < height; y++) {
        std::string row;
        std::getline(std::cin, row);
        cerr << row << endl;

        for(int x = 0; x < width; x++) {
            if(row[x] == '#') {
                grid.cell[y+BORDERH][x+BORDER] = WALL;
            } else {
                grid.cell[y+BORDERH][x+BORDER] = EMPTY;
            }
        }
    }
}

bool parseBody(const std::string &s, Snake &snake, Grid &grid) {
    snake.len = 0;
    snake.head = 0;
    snake.tail = 0;
    
    int x=0, y=0;
    int idx = 0;

    Pos temp[MAX_BODY];

    for (size_t i=0; i<s.size(); ) {

        // --- parse x ---
        int sign = 1;
        if(s[i] == '-') {
            sign = -1;
            i++;
            //return false;
        }

        x = 0;
        while(i < s.size() && isdigit(s[i])) {
            x = x*10 + (s[i]-'0');
            i++;
        }
        x *= sign;

        if(i < s.size() && s[i]==',') i++;

        // --- parse y ---
        sign = 1;
        if(s[i] == '-') {
            sign = -1;
            i++;
            //return false;
        }

        y = 0;
        while(i < s.size() && isdigit(s[i])) {
            y = y*10 + (s[i]-'0');
            i++;
        }
        y *= sign;

        temp[idx++] = {(int8_t)(x+BORDER), (int8_t)(y+BORDERH)};

        
        if(i < s.size() && s[i]==':') i++;
    }

    snake.len = idx;

    // inverser pour avoir head à la fin (ton système circulaire)
    for(int i=0;i<snake.len;i++){
        snake.body[i] = temp[snake.len - 1 - i];
        Pos b = snake.body[i];
        //grid.cell[b.y][b.x] = SNAKE;
    }

    snake.head = snake.len - 1;
    snake.tail = 0;

    return true;
}

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
Grid grid;
SM simulation;

int main()
{
    srand(time(NULL));  
    cerr << "FUCK UP" << endl;
    int my_id;
    cin >> my_id; cin.ignore();
    cerr << my_id << endl;
    int width;
    cin >> width; cin.ignore();
    cerr << width << endl;
    int height;
    cin >> height; cin.ignore();
    cerr << height << endl;

    
    parseGrid(grid, height, width);
    /*for (int i = 0; i < height; i++) {
        string row;
        getline(cin, row);

    }*/

    map<int, int> snake_player;
    int snakebots_per_player;
    cin >> snakebots_per_player; cin.ignore();
    cerr << snakebots_per_player << endl;
    for (int i = 0; i < snakebots_per_player; i++) {
        int my_snakebot_id;
        cin >> my_snakebot_id; cin.ignore();
        cerr << my_snakebot_id << endl;
        snake_player[my_snakebot_id] = 0;
    }
    for (int i = 0; i < snakebots_per_player; i++) {
        int opp_snakebot_id;
        cin >> opp_snakebot_id; cin.ignore();
        cerr << opp_snakebot_id << endl;
        snake_player[opp_snakebot_id] = 1;
    }

    Snake last_my_snake[8];
    int last_my_snake_len=0;

    initZobrist();

    int turn = 0;
    // game loop
    while (1) {
        Grid grid_g = grid;
        __builtin_memcpy(&grid_g, &grid, sizeof(Grid));
        Pos energy[MAX_POWER];
        int inde = 0;

        int power_source_count;
        cin >> power_source_count; cin.ignore();
        cerr << "power=" << power_source_count << endl;
        for (int i = 0; i < power_source_count; i++) {
            int x;
            int y;
            cin >> x >> y; cin.ignore();
            cerr << x << " " << y  << endl;
            grid_g.cell[y+BORDERH][x+BORDER] = ENERGY;
            energy[i] = Pos{(int8_t)(x+BORDER), (int8_t)(y+BORDERH)};
            
        }

        //if(turn == 0){
        auto startm = high_resolution_clock::now();
            
            simulation.distg = simulation.compute_distance_mapg(width+BORDER*2, height+BORDERH, grid, energy, power_source_count, simulation.parentg_sim);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - startm);
            cerr <<"DURATION=" << duration.count() << endl;
            int DURATION = duration.count();
        //}

        Snake my_snake[8], opp_snake[8];
        int my_id_snake[8];
        int indids = 0;
        int my_snake_len=0, opp_snake_len=0;

        int snakebot_count;
        cin >> snakebot_count; cin.ignore();
        cerr << snakebot_count << endl;
        for (int i = 0; i < snakebot_count; i++) {
            int snakebot_id;
            string body;
            cin >> snakebot_id >> body; cin.ignore();
            cerr << snakebot_id << " " << body << endl;

            Snake snake;
            snake.id = snakebot_id;
            snake.alive = true;
            bool good = parseBody(body, snake, grid_g);
            if(good){
                if (snake_player[snakebot_id] == 0){
                    my_snake[my_snake_len] = snake;
                    my_snake_len++;
                    my_id_snake[indids] = snakebot_id;
                    indids++;
                }
                else if (snake_player[snakebot_id] == 1){
                    opp_snake[opp_snake_len] = snake;
                    opp_snake_len++;
                }
            }

        }

        /*for(int y = 0; y < height; y++) {
            
            for(int x = 0; x < width; x++) {
                if(grid_g.cell[y][x] & WALL) {
                    cerr << "#";
                }
                else if(grid_g.cell[y][x] & SNAKE) {
                    cerr << "O";
                }
                else if(grid_g.cell[y][x] & ENERGY) {
                    cerr << "e";
                } else {
                    cerr << " ";
                }
            }
            cerr << endl;
        }*/
        
        /*cerr << "len=" << 45.0/(double)my_snake_len << endl;
        string ans;
        for(int player = 0;player < my_snake_len;++player){
            ans += simulation.PlayB(player, height, width, grid_g, my_snake, my_snake_len, opp_snake, opp_snake_len, my_id_snake, energy, power_source_count, 40.0/(double)my_snake_len);

            // Write an action using cout. DON'T FORGET THE "<< endl"
            // To debug: cerr << "Debug messages..." << endl;

            
        }*/

        int time = 60;
        if(turn > 0)time = 45;

        Snake my_snakem[8], my_snakeb[8];
        int my_id_snakeb[8], my_id_snakem[8];
        int my_snake_lenb=0, my_snake_lenm=0;

        int mid = max(1, (my_snake_len / 2));

        cerr << "LEN=" << my_snake_len << endl;

        for(int i = 0;i < my_snake_len;++i){
            my_snakeb[i] = my_snake[i];
            my_snake_lenb++;
            my_id_snakeb[i] = my_id_snake[i];
        }

        /*for(int i = mid;i < my_snake_len;++i){
            opp_snake[opp_snake_len] = my_snake[i];
            opp_snake_len++; 

        }*/

        int j = 0;
        for(int i = 0;i < my_snake_len;++i){
            my_snakem[j] = my_snake[i];
            my_snake_lenm++;
            my_id_snakem[j] = my_id_snake[i];
            ++j;
        }


        string ans = simulation.Play(height, width, grid_g, my_snake, my_snake_len, opp_snake, opp_snake_len, my_id_snake, energy, power_source_count, time-DURATION, turn);
        //string ans2 = simulation.PlayMCTS(height, width, grid_g, my_snakem, my_snake_lenm, opp_snake, opp_snake_len, my_id_snakem, energy, power_source_count, time);

        cout << ans << endl;
        //cout << ans << ans2 << endl;
        cout.flush();

        /*last_my_snake_len = my_snake_lenb;
        for(int i = 0;i < my_snake_lenb;++i){
            last_my_snake[i] = my_snakeb[i];
        }*/

        ++turn;

    }
}