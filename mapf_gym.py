import gym
from gym import spaces
import numpy as np
from collections import OrderedDict
from threading import Lock
import sys
from matplotlib.colors import hsv_to_rgb
import random
import math
import copy
# from od_mstar3 import cpp_mstar
# from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
try:
    from od_mstar3 import cpp_mstar
    USE_CPP_MSTAR = True
except ImportError:
    cpp_mstar = None
    USE_CPP_MSTAR = False

from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
# from gym.envs.classic_control import rendering        

'''
    Observation: (position maps of current agent, current goal, other agents, other goals, obstacles)
        
    Action space: (Tuple)
        agent_id: positive integer
        action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
        5:NE, 6:SE, 7:SW, 8:NW}
    Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
'''
ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD,FINISH_REWARD,BLOCKING_COST = -0.3, -.5, 0.0, -2.,20.,-1.
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
JOINT = False # True for joint estimation of rewards for closeby agents
dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
actionDict={v:k for k,v in dirDict.items()}
class State(object):
    '''
    State.
    Implemented as 2 2d numpy arrays.
    first one "state":
        static obstacle: -1
        empty: 0
        agent = positive integer (agent_id)
    second one "goals":
        agent goal = positive int(agent_id)
    '''
    def __init__(self, world0, goals, diagonal, num_agents=1):
        assert(len(world0.shape) == 2 and world0.shape==goals.shape)
        self.state                    = world0.copy()
        self.goals                    = goals.copy()
        self.num_agents               = num_agents
        self.agents, self.agents_past, self.agent_goals = self.scanForAgents()
        self.diagonal=diagonal
        assert(len(self.agents) == num_agents)

    def scanForAgents(self):
        agents = [(-1,-1) for i in range(self.num_agents)]
        agents_last = [(-1,-1) for i in range(self.num_agents)]        
        agent_goals = [(-1,-1) for i in range(self.num_agents)]        
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if(self.state[i,j]>0):
                    agents[self.state[i,j]-1] = (i,j)
                    agents_last[self.state[i,j]-1] = (i,j)
                if(self.goals[i,j]>0):
                    agent_goals[self.goals[i,j]-1] = (i,j)
        assert((-1,-1) not in agents and (-1,-1) not in agent_goals)
        assert(agents==agents_last)
        return agents, agents_last, agent_goals

    def getPos(self, agent_id):
        return self.agents[agent_id-1]

    def getPastPos(self, agent_id):
        return self.agents_past[agent_id-1]

    def getGoal(self, agent_id):
        return self.agent_goals[agent_id-1]
    
    def diagonalCollision(self, agent_id, newPos):
        '''diagonalCollision(id,(x,y)) returns true if agent with id "id" collided diagonally with 
        any other agent in the state after moving to coordinates (x,y)
        agent_id: id of the desired agent to check for
        newPos: coord the agent is trying to move to (and checking for collisions)
        '''
#        def eq(f1,f2):return abs(f1-f2)<0.001
        def collide(a1,a2,b1,b2):
            '''
            a1,a2 are coords for agent 1, b1,b2 coords for agent 2, returns true if these collide diagonally
            '''
            return np.isclose( (a1[0]+a2[0]) /2. , (b1[0]+b2[0])/2. ) and np.isclose( (a1[1]+a2[1])/2. , (b1[1]+b2[1])/2. )
        assert(len(newPos) == 2);
        #up until now we haven't moved the agent, so getPos returns the "old" location
        lastPos = self.getPos(agent_id)
        for agent in range(1,self.num_agents+1):
            if agent == agent_id: continue
            aPast = self.getPastPos(agent)
            aPres = self.getPos(agent)
            if collide(aPast,aPres,lastPos,newPos): return True
        return False

    #try to move agent and return the status
    def moveAgent(self, direction, agent_id):
        ax=self.agents[agent_id-1][0]
        ay=self.agents[agent_id-1][1]

        # Not moving is always allowed
        if(direction==(0,0)):
            self.agents_past[agent_id-1]=self.agents[agent_id-1]
            return 1 if self.goals[ax,ay]==agent_id else 0

        # Otherwise, let's look at the validity of the move
        dx,dy =direction[0], direction[1]
        if(ax+dx>=self.state.shape[0] or ax+dx<0 or ay+dy>=self.state.shape[1] or ay+dy<0):#out of bounds
            return -1
        if(self.state[ax+dx,ay+dy]<0):#collide with static obstacle
            return -2
        if(self.state[ax+dx,ay+dy]>0):#collide with robot
            return -3
        # check for diagonal collisions
        if(self.diagonal):
            if self.diagonalCollision(agent_id,(ax+dx,ay+dy)):
                return -3
        # No collision: we can carry out the action
        self.state[ax,ay] = 0
        self.state[ax+dx,ay+dy] = agent_id
        self.agents_past[agent_id-1]=self.agents[agent_id-1]
        self.agents[agent_id-1] = (ax+dx,ay+dy)
        if self.goals[ax+dx,ay+dy]==agent_id:
            return 1
        elif self.goals[ax+dx,ay+dy]!=agent_id and self.goals[ax,ay]==agent_id:
            return 2
        else:
            return 0

    # try to execture action and return whether action was executed or not and why
    #returns:
    #     2: action executed and left goal
    #     1: action executed and reached goal (or stayed on)
    #     0: action executed
    #    -1: out of bounds
    #    -2: collision with wall
    #    -3: collision with robot
    def act(self, action, agent_id):
        # 0     1  2  3  4 
        # still N  E  S  W
        direction = self.getDir(action)
        moved = self.moveAgent(direction,agent_id)
        return moved

    def getDir(self,action):
        return dirDict[action]
    def getAction(self,direction):
        return actionDict[direction]

    # Compare with a plan to determine job completion
    def done(self):
        numComplete = 0
        for i in range(1,len(self.agents)+1):
            agent_pos = self.agents[i-1]
            if self.goals[agent_pos[0],agent_pos[1]] == i:
                numComplete += 1
        return numComplete==len(self.agents) #, numComplete/float(len(self.agents))


class MAPFEnv(gym.Env):
    def getFinishReward(self):
        return FINISH_REWARD
    metadata = {"render.modes": ["human", "ansi"]}

    # Initialize env
    def __init__(self, num_agents=1, observation_size=10,world0=None, goals0=None, DIAGONAL_MOVEMENT=False, SIZE=(10,40), PROB=(0,.5), FULL_HELP=False,blank_world=False):
        """
        Args:
            DIAGONAL_MOVEMENT: if the agents are allowed to move diagonally
            SIZE: size of a side of the square grid
            PROB: range of probabilities that a given block is an obstacle
            FULL_HELP
        """
        # Initialize member variables
        self.num_agents        = num_agents 
        #a way of doing joint rewards
        self.individual_rewards           = [0 for i in range(num_agents)]
        self.observation_size  = observation_size
        self.SIZE              = SIZE
        self.PROB              = PROB
        self.fresh             = True
        self.FULL_HELP         = FULL_HELP
        self.finished          = False
        self.mutex             = Lock()
        self.DIAGONAL_MOVEMENT = DIAGONAL_MOVEMENT

        # variables for corridor cache
        self.static_obs_full = None

        self.corridor_id_map = None
        self.corridor_type_map = None
        self.delta_x_full = None
        self.delta_y_full = None

        self.corridor_cells = {}
        self.corridor_endpoints = {}
        self.corridor_stopping_points = {}
        self.corridor_ordered_cells = {}
        self.endpoint_to_corridor = {}
        self.stopping_point_to_corridor = {}
        self.stopping_point_meta = {}

        # Initialize data structures
        self._setWorld(world0,goals0,blank_world=blank_world)
        if DIAGONAL_MOVEMENT:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(9)])
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(5)])
        self.viewer           = None

    def isConnected(self,world0):
        sys.setrecursionlimit(10000)
        world0 = world0.copy()

        def firstFree(world0):
            for x in range(world0.shape[0]):
                for y in range(world0.shape[1]):
                    if world0[x,y]==0:
                        return x,y
        def floodfill(world,i,j):
            sx,sy=world.shape[0],world.shape[1]
            if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                return
            if(world[i,j]==-1):return
            world[i,j] = -1
            floodfill(world,i+1,j)
            floodfill(world,i,j+1)
            floodfill(world,i-1,j)
            floodfill(world,i,j-1)

        i,j = firstFree(world0)
        floodfill(world0,i,j)
        if np.any(world0==0):
            return False
        else:
            return True

    def getObstacleMap(self):
        return (self.world.state==-1).astype(int)
    
    def getGoals(self):
        result=[]
        for i in range(1,self.num_agents+1):
            result.append(self.world.getGoal(i))
        return result
    
    def getPositions(self):
        result=[]
        for i in range(1,self.num_agents+1):
            result.append(self.world.getPos(i))
        return result
    
    def _setWorld(self, world0=None, goals0=None,blank_world=False):
        #blank_world is a flag indicating that the world given has no agent or goal positions 
        def getConnectedRegion(world,regions_dict,x,y):
            sys.setrecursionlimit(1000000)
            '''returns a list of tuples of connected squares to the given tile
            this is memoized with a dict'''
            if (x,y) in regions_dict:
                return regions_dict[(x,y)]
            visited=set()
            sx,sy=world.shape[0],world.shape[1]
            work_list=[(x,y)]
            while len(work_list)>0:
                (i,j)=work_list.pop()
                if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                    continue
                if(world[i,j]==-1):
                    continue#crashes
                if world[i,j]>0:
                    regions_dict[(i,j)]=visited
                if (i,j) in visited:continue
                visited.add((i,j))
                work_list.append((i+1,j))
                work_list.append((i,j+1))
                work_list.append((i-1,j))
                work_list.append((i,j-1))
            regions_dict[(x,y)]=visited
            return visited
        #defines the State object, which includes initializing goals and agents
        #sets the world to world0 and goals, or if they are None randomizes world
        if not (world0 is None):
            if goals0 is None and not blank_world:
                raise Exception("you gave a world with no goals!")
            if blank_world:
                #RANDOMIZE THE POSITIONS OF AGENTS
                agent_counter = 1
                agent_locations=[]
                while agent_counter<=self.num_agents:
                    x,y       = np.random.randint(0,world0.shape[0]),np.random.randint(0,world0.shape[1])
                    if(world0[x,y] == 0):
                        world0[x,y]=agent_counter
                        agent_locations.append((x,y))
                        agent_counter += 1   
                #RANDOMIZE THE GOALS OF AGENTS
                goals0 = np.zeros(world0.shape).astype(int)
                goal_counter = 1
                agent_regions=dict()  
                while goal_counter<=self.num_agents:
                    agent_pos=agent_locations[goal_counter-1]
                    valid_tiles=getConnectedRegion(world0,agent_regions,agent_pos[0],agent_pos[1])#crashes
                    x,y  = random.choice(list(valid_tiles))
                    if(goals0[x,y]==0 and world0[x,y]!=-1):
                        goals0[x,y]    = goal_counter
                        goal_counter += 1
                self.initial_world = world0.copy()
                self.initial_goals = goals0.copy()
                self.world = State(self.initial_world,self.initial_goals,self.DIAGONAL_MOVEMENT,self.num_agents)
                self.static_obs_full = (self.world.state == -1).astype(np.float32)
                self._compute_corridor_cache()
                return
            self.initial_world = world0
            self.initial_goals = goals0
            self.world = State(world0,goals0,self.DIAGONAL_MOVEMENT,self.num_agents)
            self.static_obs_full = (self.world.state == -1).astype(np.float32)
            self._compute_corridor_cache()
            return

        #otherwise we have to randomize the world
        #RANDOMIZE THE STATIC OBSTACLES
        prob=np.random.triangular(self.PROB[0],.33*self.PROB[0]+.66*self.PROB[1],self.PROB[1])
        size=np.random.choice([self.SIZE[0],self.SIZE[0]*.5+self.SIZE[1]*.5,self.SIZE[1]],p=[.5,.25,.25])
        world     = -(np.random.rand(int(size),int(size))<prob).astype(int)

        #RANDOMIZE THE POSITIONS OF AGENTS
        agent_counter = 1
        agent_locations=[]
        while agent_counter<=self.num_agents:
            x,y       = np.random.randint(0,world.shape[0]),np.random.randint(0,world.shape[1])
            if(world[x,y] == 0):
                world[x,y]=agent_counter
                agent_locations.append((x,y))
                agent_counter += 1        
        
        #RANDOMIZE THE GOALS OF AGENTS
        goals = np.zeros(world.shape).astype(int)
        goal_counter = 1
        agent_regions=dict()     
        while goal_counter<=self.num_agents:
            agent_pos=agent_locations[goal_counter-1]
            valid_tiles=getConnectedRegion(world,agent_regions,agent_pos[0],agent_pos[1])
            x,y  = random.choice(list(valid_tiles))
            if(goals[x,y]==0 and world[x,y]!=-1):
                goals[x,y]    = goal_counter
                goal_counter += 1
        self.initial_world = world
        self.initial_goals = goals
        self.world = State(world,goals,self.DIAGONAL_MOVEMENT,num_agents=self.num_agents)
        self.static_obs_full = (self.world.state == -1).astype(np.float32)
        self._compute_corridor_cache()

    # methods used for generating corridor-related maps
    def _compute_corridor_cache(self):
        """
        Recompute all corridor-related caches for the current static map.

        Updates:
            self.corridor_id_map
            self.corridor_type_map
            self.delta_x_full
            self.delta_y_full
            self.corridor_cells
            self.corridor_endpoints
            self.corridor_stopping_points
            self.corridor_ordered_cells
            self.endpoint_to_corridor
            self.stopping_point_to_corridor
            self.stopping_point_meta
        """
        # 1) Clear old cache
        self.corridor_cells = {}
        self.corridor_endpoints = {}
        self.corridor_stopping_points = {}
        self.corridor_ordered_cells = {}
        self.endpoint_to_corridor = {}
        self.stopping_point_to_corridor = {}
        self.stopping_point_meta = {}

        h, w = self.world.state.shape
        self.corridor_id_map = np.full((h, w), -1, dtype=np.int32)
        self.corridor_type_map = np.zeros((h, w), dtype=np.int32)
        self.delta_x_full = np.zeros((h, w), dtype=np.float32)
        self.delta_y_full = np.zeros((h, w), dtype=np.float32)

        # corridor_type_map coding:
        #   -1: obstacle
        #    0: free outside corridor
        #    1: corridor internal cell
        #    2: corridor endpoint / decision point
        #    3: stopping point
        for x in range(h):
            for y in range(w):
                if not self._is_static_free_cell(x, y):
                    self.corridor_type_map[x, y] = -1

        free_degree_map = self._compute_free_degree_map()
        visited = set()
        corridor_id = 0

        # 2) Find all corridor connected components
        for x in range(h):
            for y in range(w):
                if (x, y) in visited:
                    continue
                if not self._is_corridor_seed(x, y, free_degree_map):
                    continue

                cells = self._grow_corridor_component((x, y), free_degree_map, visited)
                if not cells:
                    continue

                endpoints = self._collect_corridor_endpoints(cells, free_degree_map)
                if len(endpoints) == 0:
                    # Components with no decision-side endpoint do not provide entry semantics.
                    continue

                ordered_cells = self._order_corridor_cells(cells)
                if not ordered_cells:
                    ordered_cells = sorted(list(cells))

                self.corridor_cells[corridor_id] = ordered_cells
                self.corridor_ordered_cells[corridor_id] = ordered_cells

                for cx, cy in cells:
                    self.corridor_id_map[cx, cy] = corridor_id
                    self.corridor_type_map[cx, cy] = 1

                self.corridor_endpoints[corridor_id] = endpoints
                for ex, ey in endpoints:
                    if (ex, ey) not in self.endpoint_to_corridor:
                        self.endpoint_to_corridor[(ex, ey)] = corridor_id
                    if self.corridor_type_map[ex, ey] != -1:
                        self.corridor_type_map[ex, ey] = 2

                stopping_points, stopping_meta = self._build_stopping_points_for_corridor(
                    corridor_id, ordered_cells, endpoints
                )
                self.corridor_stopping_points[corridor_id] = stopping_points

                for sp in stopping_points:
                    self.stopping_point_to_corridor[sp] = corridor_id
                    if self.corridor_type_map[sp[0], sp[1]] != -1:
                        self.corridor_type_map[sp[0], sp[1]] = 3

                for sp, meta in stopping_meta.items():
                    self.stopping_point_meta[sp] = meta

                corridor_id += 1

        # 3) Build sparse delta maps from stopping points
        self._build_delta_maps()

    def _is_free_cell(self, x, y):
        if x < 0 or x >= self.world.state.shape[0] or y < 0 or y >= self.world.state.shape[1]:
            return False
        return self.world.state[x, y] != -1

    def _is_static_free_cell(self, x, y):
        h, w = self.world.state.shape
        if x < 0 or x >= h or y < 0 or y >= w:
            return False
        if self.static_obs_full is not None:
            return self.static_obs_full[x, y] == 0
        return self.world.state[x, y] != -1

    def _get_free_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = x + dx
            ny = y + dy
            if self._is_free_cell(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def _get_static_free_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = x + dx
            ny = y + dy
            if self._is_static_free_cell(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def _is_static_blocked_or_oob(self, x, y):
        h, w = self.world.state.shape
        if x < 0 or x >= h or y < 0 or y >= w:
            return True
        return not self._is_static_free_cell(x, y)
    
    def _is_pass_through_corridor_cell(self, x, y, free_degree_map):
        """
        严格 corridor 判定：
        - 恰好两个自由邻居
        - 两邻居必须对向
        - 另外两个侧向必须被堵
        """
        if not self._is_static_free_cell(x, y):
            return False

        neighbors = self._get_static_free_neighbors(x, y)
        if len(neighbors) != 2:
            return False

        n1, n2 = neighbors

        # 水平 corridor：左右自由，上下被堵
        horizontal = (n1[0] == x and n2[0] == x and abs(n1[1] - n2[1]) == 2)
        if horizontal:
            return self._is_static_blocked_or_oob(x - 1, y) and self._is_static_blocked_or_oob(x + 1, y)

        # 垂直 corridor：上下自由，左右被堵
        vertical = (n1[1] == y and n2[1] == y and abs(n1[0] - n2[0]) == 2)
        if vertical:
            return self._is_static_blocked_or_oob(x, y - 1) and self._is_static_blocked_or_oob(x, y + 1)

        return False
    
    def _is_dead_end_terminal_cell(self, x, y, free_degree_map):
        """
        用来补 dead-end 最末端格子（degree == 1）。

        条件：
        1) 当前格恰好只有一个静态自由邻居
        2) 当前格在唯一邻居方向上形成狭窄通道（侧向被堵）
        3) 唯一邻居要么是严格的普通 corridor cell，要么是 decision/endpoint 候选
        """
        if not self._is_static_free_cell(x, y):
            return False

        if free_degree_map[x, y] != 1:
            return False

        neighbors = self._get_static_free_neighbors(x, y)
        if len(neighbors) != 1:
            return False

        nx, ny = neighbors[0]

        # 当前 terminal cell 必须也具有 corridor 的“狭窄性”
        if nx == x:
            # 水平方向连接，要求上下被堵
            if not (self._is_static_blocked_or_oob(x - 1, y) and self._is_static_blocked_or_oob(x + 1, y)):
                return False
        elif ny == y:
            # 垂直方向连接，要求左右被堵
            if not (self._is_static_blocked_or_oob(x, y - 1) and self._is_static_blocked_or_oob(x, y + 1)):
                return False
        else:
            return False

        neighbor_degree = free_degree_map[nx, ny]

        # 情况 A：唯一邻居本身是普通 corridor cell
        if self._is_pass_through_corridor_cell(nx, ny, free_degree_map):
            return True

        # 情况 B：唯一邻居是外部 decision / endpoint 候选（自由度较大）
        # 这允许“只有一个 internal cell 的短死胡同”
        if neighbor_degree >= 3:
            return True

        return False

    def _compute_free_degree_map(self):
        h, w = self.world.state.shape
        degree_map = np.zeros((h, w), dtype=np.int32)
        for x in range(h):
            for y in range(w):
                if not self._is_static_free_cell(x, y):
                    continue
                degree_map[x, y] = len(self._get_static_free_neighbors(x, y))
        return degree_map

    def _is_corridor_internal_candidate(self, x, y, free_degree_map):
        """
        混合判定：
        - degree==2 时，沿用严格 corridor 判定
        - degree==1 时，只允许 dead-end terminal cell
        """
        if not self._is_static_free_cell(x, y):
            return False

        degree = free_degree_map[x, y]

        # 普通 corridor
        if degree == 2:
            return self._is_pass_through_corridor_cell(x, y, free_degree_map)

        # 补 dead-end 末端格
        if degree == 1:
            return self._is_dead_end_terminal_cell(x, y, free_degree_map)

        return False

    def _is_corridor_seed(self, x, y, free_degree_map):
        """
        直接沿用严格 corridor internal 判定。
        只要当前格是合法 corridor internal cell，就可作为 seed。
        """
        return self._is_corridor_internal_candidate(x, y, free_degree_map)

    def _grow_corridor_component(self, seed, free_degree_map, visited):
        stack = [seed]
        cells = set()

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in cells:
                continue
            if not self._is_corridor_internal_candidate(cx, cy, free_degree_map):
                continue

            cells.add((cx, cy))
            visited.add((cx, cy))

            for nx, ny in self._get_static_free_neighbors(cx, cy):
                if (nx, ny) in cells:
                    continue
                if self._is_corridor_internal_candidate(nx, ny, free_degree_map):
                    stack.append((nx, ny))

        return cells

    def _collect_corridor_endpoints(self, cells, free_degree_map):
        endpoints = set()
        for cx, cy in cells:
            for nx, ny in self._get_static_free_neighbors(cx, cy):
                if (nx, ny) in cells:
                    continue
                # Endpoint / decision point candidate should be outside corridor
                # and have branching capacity.
                if free_degree_map[nx, ny] >= 3:
                    endpoints.add((nx, ny))
        return sorted(list(endpoints))

    def _order_corridor_cells(self, cells):
        if not cells:
            return []

        cell_set = set(cells)
        adjacency = {}
        for x, y in cell_set:
            adjacency[(x, y)] = []
            for nx, ny in self._get_static_free_neighbors(x, y):
                if (nx, ny) in cell_set:
                    adjacency[(x, y)].append((nx, ny))

        degree_one_nodes = sorted([c for c in cell_set if len(adjacency[c]) <= 1])
        if degree_one_nodes:
            start = degree_one_nodes[0]
        else:
            start = sorted(list(cell_set))[0]

        order = [start]
        visited_local = set([start])
        prev = None
        current = start

        while True:
            next_candidates = []
            for neighbor in adjacency[current]:
                if neighbor == prev:
                    continue
                if neighbor in visited_local:
                    continue
                next_candidates.append(neighbor)

            if len(next_candidates) == 0:
                break

            next_cell = sorted(next_candidates)[0]
            order.append(next_cell)
            visited_local.add(next_cell)
            prev = current
            current = next_cell

        if len(order) != len(cell_set):
            for cell in sorted(list(cell_set)):
                if cell not in visited_local:
                    order.append(cell)

        return order

    def _build_stopping_points_for_corridor(self, corridor_id, ordered_cells, endpoints):
        cell_to_index = {cell: idx for idx, cell in enumerate(ordered_cells)}

        endpoint_entries = []
        for endpoint in endpoints:
            adjacent_internal = []
            for neighbor in self._get_static_free_neighbors(endpoint[0], endpoint[1]):
                if neighbor in cell_to_index:
                    adjacent_internal.append(neighbor)
            if len(adjacent_internal) == 0:
                continue

            best_stop = min(adjacent_internal, key=lambda c: cell_to_index[c])
            endpoint_entries.append({
                "endpoint": endpoint,
                "stopping_point": best_stop,
                "index": cell_to_index[best_stop]
            })

        if len(endpoint_entries) == 0:
            return [], {}

        endpoint_entries.sort(key=lambda e: e["index"])

        # Most corridors in this formulation are dead-end (1 endpoint)
        # or pass-through (2 endpoints). If there are more, keep the outermost two.
        selected_entries = []
        if len(endpoint_entries) == 1:
            selected_entries = [endpoint_entries[0]]
        else:
            selected_entries = [endpoint_entries[0], endpoint_entries[-1]]

        stopping_points = []
        stopping_meta = {}

        if len(selected_entries) == 2 and \
                selected_entries[0]["endpoint"] != selected_entries[1]["endpoint"] and \
                selected_entries[0]["stopping_point"] != selected_entries[1]["stopping_point"]:
            left = selected_entries[0]
            right = selected_entries[1]
            i_left = left["index"]
            i_right = right["index"]

            if i_left <= i_right:
                path_lr = ordered_cells[i_left:i_right + 1]
            else:
                path_lr = list(reversed(ordered_cells[i_right:i_left + 1]))
                left, right = right, left

            left_stop = left["stopping_point"]
            right_stop = right["stopping_point"]
            left_endpoint = left["endpoint"]
            right_endpoint = right["endpoint"]

            stopping_points.extend([left_stop, right_stop])

            stopping_meta[left_stop] = {
                "corridor_id": corridor_id,
                "endpoint": left_endpoint,
                "paired_endpoint": right_endpoint,
                "paired_stopping_point": right_stop,
                "is_dead_end": False,
                "scan_cells": list(path_lr)
            }
            stopping_meta[right_stop] = {
                "corridor_id": corridor_id,
                "endpoint": right_endpoint,
                "paired_endpoint": left_endpoint,
                "paired_stopping_point": left_stop,
                "is_dead_end": False,
                "scan_cells": list(reversed(path_lr))
            }
        else:
            entry = selected_entries[0]
            stop = entry["stopping_point"]
            idx = entry["index"]

            left_path = list(reversed(ordered_cells[:idx + 1]))
            right_path = ordered_cells[idx:]
            if len(right_path) >= len(left_path):
                scan_cells = list(right_path)
            else:
                scan_cells = list(left_path)

            stopping_points.append(stop)
            stopping_meta[stop] = {
                "corridor_id": corridor_id,
                "endpoint": entry["endpoint"],
                "paired_endpoint": None,
                "paired_stopping_point": None,
                "is_dead_end": True,
                "scan_cells": scan_cells
            }

        # remove duplicates while keeping order
        stopping_points = list(OrderedDict.fromkeys(stopping_points))
        return stopping_points, stopping_meta

    def _build_delta_maps(self):
        self.delta_x_full.fill(0.0)
        self.delta_y_full.fill(0.0)

        for stopping_point, meta in self.stopping_point_meta.items():
            endpoint = meta.get("endpoint")
            paired_endpoint = meta.get("paired_endpoint")
            if endpoint is None or paired_endpoint is None:
                continue

            sx, sy = stopping_point
            ex, ey = endpoint
            ox, oy = paired_endpoint
            self.delta_x_full[sx, sy] = ox - ex
            self.delta_y_full[sx, sy] = oy - ey

    def _extract_fov(self, full_map, top_left, fill_value=0.0):
        obs_shape = (self.observation_size, self.observation_size)
        fov = np.full(obs_shape, fill_value, dtype=np.float32)

        src_x0 = max(0, top_left[0])
        src_y0 = max(0, top_left[1])
        src_x1 = min(full_map.shape[0], top_left[0] + self.observation_size)
        src_y1 = min(full_map.shape[1], top_left[1] + self.observation_size)

        if src_x0 >= src_x1 or src_y0 >= src_y1:
            return fov

        dst_x0 = src_x0 - top_left[0]
        dst_y0 = src_y0 - top_left[1]
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        fov[dst_x0:dst_x1, dst_y0:dst_y1] = full_map[src_x0:src_x1, src_y0:src_y1]
        return fov

    def _get_visible_agents(self, agent_id, top_left):
        visible_agents = []
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        for other_agent in range(1, self.num_agents + 1):
            if other_agent == agent_id:
                continue
            x, y = self.world.getPos(other_agent)
            if x < top_left[0] or x >= bottom_right[0] or y < top_left[1] or y >= bottom_right[1]:
                continue
            visible_agents.append(other_agent)
        return visible_agents

    def _predict_agent_future_positions(self, other_agent_id, horizon=3):
        pos = self.world.getPos(other_agent_id)
        goal = self.world.getGoal(other_agent_id)
        costs = self.getAstarCosts(pos, goal)

        future_positions = []
        current = pos
        for _ in range(horizon):
            best_pos = current
            best_cost = costs[current[0], current[1]]
            for nx, ny in self._get_free_neighbors(current[0], current[1]):
                neighbor_cost = costs[nx, ny]
                if neighbor_cost < best_cost:
                    best_cost = neighbor_cost
                    best_pos = (nx, ny)
            current = best_pos
            future_positions.append(current)
        return future_positions

    def _build_prediction_maps(self, agent_id, visible_agents, top_left, horizon=3):
        pred_maps = [np.zeros((self.observation_size, self.observation_size), dtype=np.float32)
                     for _ in range(horizon)]

        for other_agent_id in visible_agents:
            if other_agent_id == agent_id:
                continue
            future_positions = self._predict_agent_future_positions(other_agent_id, horizon=horizon)
            for step_idx, pos in enumerate(future_positions):
                local_x = pos[0] - top_left[0]
                local_y = pos[1] - top_left[1]
                if local_x < 0 or local_x >= self.observation_size or local_y < 0 or local_y >= self.observation_size:
                    continue
                pred_maps[step_idx][local_x, local_y] = 1.0
        return tuple(pred_maps)

    def _build_blocking_map(self, agent_id, top_left, visible_agents):
        """
        Build blocking_map on stopping points:
            1  -> current side clearly blocked, should not enter
            0  -> no obvious blocking risk from this side
           -1  -> far endpoint occupied by other agent (soft caution)

        visible_agents is intentionally not used here; blocking semantics are
        based on global corridor occupancy and movement trends.
        """
        obs_size = self.observation_size
        blocking_map = np.zeros((obs_size, obs_size), dtype=np.float32)

        if self.corridor_id_map is None or len(self.stopping_point_meta) == 0:
            return blocking_map

        current_pos = self.world.getPos(agent_id)
        current_corridor_id = self.corridor_id_map[current_pos[0], current_pos[1]]

        top_x, top_y = top_left
        for stopping_point, meta in self.stopping_point_meta.items():
            corridor_id = meta["corridor_id"]

            # If the observer is already inside this corridor, this side-entry
            # signal is not informative for immediate decision making.
            if current_corridor_id != -1 and corridor_id == current_corridor_id:
                continue

            blocking_value = self._evaluate_stopping_point_blocking(meta, agent_id)
            spx, spy = stopping_point
            local_x = spx - top_x
            local_y = spy - top_y
            if 0 <= local_x < obs_size and 0 <= local_y < obs_size:
                blocking_map[local_x, local_y] = blocking_value

        return blocking_map

    def _get_last_distinct_pos(self, other_agent_id, current_pos):
        last_pos = self.world.getPastPos(other_agent_id)
        if last_pos == current_pos:
            return None
        return last_pos

    def _evaluate_stopping_point_blocking(self, stopping_meta, observer_agent_id):
        """
        Evaluate blocking value for one stopping point from observer_agent_id's perspective.
        """
        scan_cells = stopping_meta.get("scan_cells", [])
        is_dead_end = stopping_meta.get("is_dead_end", False)

        for idx, pos in enumerate(scan_cells):
            state = self.world.state[pos[0], pos[1]]
            if state <= 0 or state == observer_agent_id:
                continue

            # Dead-end corridor: any other agent in this corridor blocks entry.
            if is_dead_end:
                return 1.0

            # Near-side occupancy blocks immediate entry.
            if idx == 0:
                return 1.0

            # Use one-step movement trend to detect whether the other agent is
            # moving towards this stopping side.
            last_pos = self._get_last_distinct_pos(state, pos)
            if last_pos is None:
                return 1.0

            if idx == len(scan_cells) - 1:
                if idx > 0 and last_pos != scan_cells[idx - 1]:
                    return 1.0
                break

            if last_pos == scan_cells[idx + 1]:
                return 1.0

        # Dual-end corridor: far endpoint occupied gives soft caution (-1).
        if not is_dead_end:
            paired_endpoint = stopping_meta.get("paired_endpoint")
            if paired_endpoint is not None:
                endpoint_state = self.world.state[paired_endpoint[0], paired_endpoint[1]]
                if endpoint_state > 0 and endpoint_state != observer_agent_id:
                    return -1.0

        return 0.0

    # Returns an observation of an agent
    def _observe(self,agent_id):
        assert(agent_id>0)
        top_left=(self.world.getPos(agent_id)[0]-self.observation_size//2,self.world.getPos(agent_id)[1]-self.observation_size//2)
        bottom_right=(top_left[0]+self.observation_size,top_left[1]+self.observation_size)        
        obs_shape=(self.observation_size,self.observation_size)
        goal_map             = np.zeros(obs_shape, dtype=np.float32)
        poss_map             = np.zeros(obs_shape, dtype=np.float32)
        goals_map            = np.zeros(obs_shape, dtype=np.float32)
        obs_map              = np.zeros(obs_shape, dtype=np.float32)
        for i in range(top_left[0],top_left[0]+self.observation_size):
            for j in range(top_left[1],top_left[1]+self.observation_size):
                if i>=self.world.state.shape[0] or i<0 or j>=self.world.state.shape[1] or j<0:
                    #out of bounds, just treat as an obstacle
                    obs_map[i-top_left[0],j-top_left[1]]=1
                    continue
                if self.world.state[i,j]==-1:
                    #obstacles
                    obs_map[i-top_left[0],j-top_left[1]]=1
                if self.world.state[i,j]==agent_id:
                    #agent's position
                    poss_map[i-top_left[0],j-top_left[1]]=1
                if self.world.goals[i,j]==agent_id:
                    #agent's goal
                    goal_map[i-top_left[0],j-top_left[1]]=1
                if self.world.state[i,j]>0 and self.world.state[i,j]!=agent_id:
                    #other agents' positions
                    poss_map[i-top_left[0],j-top_left[1]]=1

        visible_agents = self._get_visible_agents(agent_id, top_left)

        for agent in visible_agents:
            x, y = self.world.getGoal(agent)
            min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.observation_size - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        blocking_map = self._build_blocking_map(agent_id, top_left, visible_agents)
        delta_x_map = self._extract_fov(self.delta_x_full, top_left, fill_value=0.0)
        delta_y_map = self._extract_fov(self.delta_y_full, top_left, fill_value=0.0)
        pred_1, pred_2, pred_3 = self._build_prediction_maps(agent_id, visible_agents, top_left, horizon=3)

        dx=self.world.getGoal(agent_id)[0]-self.world.getPos(agent_id)[0]
        dy=self.world.getGoal(agent_id)[1]-self.world.getPos(agent_id)[1]
        mag=(dx**2+dy**2)**.5
        if mag!=0:
            dx=dx/mag
            dy=dy/mag
        return ([poss_map,goal_map,goals_map,obs_map,
                 blocking_map,delta_x_map,delta_y_map,
                 pred_1,pred_2,pred_3],[dx,dy,mag])




    # Resets environment
    def _reset(self, agent_id,world0=None,goals0=None):
        self.finished = False
        self.mutex.acquire()

        # Initialize data structures
        self._setWorld(world0,goals0)
        self.fresh = True
        
        self.mutex.release()
        if self.viewer is not None:
            self.viewer = None
        on_goal = self.world.getPos(agent_id) == self.world.getGoal(agent_id)
        #we assume you don't start blocking anyone (the probability of this happening is insanely low)
        return self._listNextValidActions(agent_id), on_goal,False

    def _complete(self):
        return self.world.done()
    
    def getAstarCosts(self,start, goal):
        #returns a numpy array of same dims as self.world.state with the distance to the goal from each coord
        def lowestF(fScore,openSet):
            #find entry in openSet with lowest fScore
            assert(len(openSet)>0)
            minF=2**31-1
            minNode=None
            for (i,j) in openSet:
                if (i,j) not in fScore:continue
                if fScore[(i,j)]<minF:
                    minF=fScore[(i,j)]
                    minNode=(i,j)
            return minNode      
        def getNeighbors(node):
            #return set of neighbors to the given node
            n_moves=9 if self.DIAGONAL_MOVEMENT else 5
            neighbors=set()
            for move in range(1,n_moves):#we dont want to include 0 or it will include itself
                direction=self.world.getDir(move)
                dx=direction[0]
                dy=direction[1]
                ax=node[0]
                ay=node[1]
                if(ax+dx>=self.world.state.shape[0] or ax+dx<0 or ay+dy>=self.world.state.shape[1] or ay+dy<0):#out of bounds
                    continue
                if(self.world.state[ax+dx,ay+dy]==-1):#collide with static obstacle
                    continue
                neighbors.add((ax+dx,ay+dy))
            return neighbors
        
        #NOTE THAT WE REVERSE THE DIRECTION OF SEARCH SO THAT THE GSCORE WILL BE DISTANCE TO GOAL
        start,goal=goal,start
        
        # The set of nodes already evaluated
        closedSet = set()
    
        # The set of currently discovered nodes that are not evaluated yet.
        # Initially, only the start node is known.
        openSet =set()
        openSet.add(start)
    
        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, cameFrom will eventually contain the
        # most efficient previous step.
        cameFrom = dict()
    
        # For each node, the cost of getting from the start node to that node.
        gScore =dict()#default value infinity
    
        # The cost of going from start to start is zero.
        gScore[start] = 0
    
        # For each node, the total cost of getting from the start node to the goal
        # by passing by that node. That value is partly known, partly heuristic.
        fScore = dict()#default infinity
    
        #our heuristic is euclidean distance to goal
        heuristic_cost_estimate = lambda x,y:math.hypot(x[0]-y[0],x[1]-y[1])
        
        # For the first node, that value is completely heuristic.        
        fScore[start] = heuristic_cost_estimate(start, goal)
    
        while len(openSet) != 0:
            #current = the node in openSet having the lowest fScore value
            current = lowestF(fScore,openSet)
    
            openSet.remove(current)
            closedSet.add(current)
            for neighbor in getNeighbors(current):
                if neighbor in closedSet:
                    continue		# Ignore the neighbor which is already evaluated.
            
                if neighbor not in openSet:	# Discover a new node
                    openSet.add(neighbor)
                
                # The distance from start to a neighbor
                #in our case the distance between is always 1
                tentative_gScore = gScore[current] + 1
                if tentative_gScore >= gScore.get(neighbor,2**31-1):
                    continue		# This is not a better path.
            
                # This path is the best until now. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + heuristic_cost_estimate(neighbor, goal) 
    
        #parse through the gScores
        costs=self.world.state.copy()
        for (i,j) in gScore:
            costs[i,j]=gScore[i,j]
        return costs
    
    # def astar(self,world,start,goal,robots=[]):
    #     '''robots is a list of robots to add to the world'''
    #     for (i,j) in robots:
    #         world[i,j]=1
    #     try:
    #         path=cpp_mstar.find_path(world,[start],[goal],1,5)
    #     except NoSolutionError:
    #         path=None
    #     for (i,j) in robots:
    #         world[i,j]=0
    #     return path

    def astar(self, world, start, goal, robots=[]):
        for (i, j) in robots:
            world[i, j] = 1
        try:
            if USE_CPP_MSTAR:
                path = cpp_mstar.find_path(world, [start], [goal], 1, 5)
            else:
                path = od_mstar.find_path(world, [start], [goal], 1, 5)
        except (NoSolutionError, OutOfTimeError):
            path = None
        for (i, j) in robots:
            world[i, j] = 0
        return path
    
    def get_blocking_reward(self,agent_id):
        '''calculates how many robots the agent is preventing from reaching goal
        and returns the necessary penalty'''
        #accumulate visible robots
        other_robots=[]
        other_locations=[]
        inflation=10
        top_left=(self.world.getPos(agent_id)[0]-self.observation_size//2,self.world.getPos(agent_id)[1]-self.observation_size//2)
        bottom_right=(top_left[0]+self.observation_size,top_left[1]+self.observation_size)        
        for agent in range(1,self.num_agents):
            if agent==agent_id: continue
            x,y=self.world.getPos(agent)
            if x<top_left[0] or x>=bottom_right[0] or y>=bottom_right[1] or y<top_left[1]:
                continue
            other_robots.append(agent)
            other_locations.append((x,y))
        num_blocking=0
        world=self.getObstacleMap()
        for agent in other_robots:
            other_locations.remove(self.world.getPos(agent))
            #before removing
            path_before=self.astar(world,self.world.getPos(agent),self.world.getGoal(agent),
                                   robots=other_locations+[self.world.getPos(agent_id)])
            #after removing
            path_after=self.astar(world,self.world.getPos(agent),self.world.getGoal(agent),
                                   robots=other_locations)
            other_locations.append(self.world.getPos(agent))
            if (path_before is None and path_after is None):continue
            if (path_before is not None and path_after is None):continue
            if (path_before is None and path_after is not None)\
                or len(path_before)>len(path_after)+inflation:
                num_blocking+=1
        return num_blocking*BLOCKING_COST
            
        
    # Executes an action by an agent
    def _step(self, action_input,episode=0):
        #episode is an optional variable which will be used on the reward discounting
        self.fresh = False
        n_actions = 9 if self.DIAGONAL_MOVEMENT else 5

        # Check action input
        assert len(action_input) == 2, 'Action input should be a tuple with the form (agent_id, action)'
        assert action_input[1] in range(n_actions), 'Invalid action'
        assert action_input[0] in range(1, self.num_agents+1)

        # Parse action input
        agent_id = action_input[0]
        action   = action_input[1]

        # Lock mutex (race conditions start here)
        self.mutex.acquire()

        #get start location of agent
        agentStartLocation = self.world.getPos(agent_id)
        
        # Execute action & determine reward
        action_status = self.world.act(action,agent_id)
        valid_action=action_status >=0
        #     2: action executed and left goal
        #     1: action executed and reached/stayed on goal
        #     0: action executed
        #    -1: out of bounds
        #    -2: collision with wall
        #    -3: collision with robot
        blocking=False
        if action==0:#staying still
            if action_status == 1:#stayed on goal
                reward=GOAL_REWARD
                x=self.get_blocking_reward(agent_id)
                reward+=x
                if x<0:
                    blocking=True
            elif action_status == 0:#stayed off goal
                reward=IDLE_COST
        else:#moving
            if (action_status == 1): # reached goal
                reward = GOAL_REWARD
            elif (action_status == -3 or action_status==-2 or action_status==-1): # collision
                reward = COLLISION_REWARD
            elif (action_status == 2): #left goal
                reward=ACTION_COST
            else:
                reward=ACTION_COST
        self.individual_rewards[agent_id-1]=reward

        if JOINT:
            visible=[False for i in range(self.num_agents)]
            v=0
            #joint rewards based on proximity
            for agent in range(1,self.num_agents+1):
                #tally up the visible agents
                if agent==agent_id:
                    continue
                top_left=(self.world.getPos(agent_id)[0]-self.observation_size//2, \
                          self.world.getPos(agent_id)[1]-self.observation_size//2)
                pos=self.world.getPos(agent)
                if pos[0]>=top_left[0] and pos[0]<top_left[0]+self.observation_size\
                    and pos[1]>=top_left[1] and pos[1]<top_left[1]+self.observation_size:
                        #if the agent is within the bounds for observation
                        v+=1
                        visible[agent-1]=True
            if v>0:
                reward=self.individual_rewards[agent_id-1]/2
                #set the reward to the joint reward if we are 
                for i in range(self.num_agents):
                    if visible[i]:
                        reward+=self.individual_rewards[i]/(v*2)

        # Perform observation
        state = self._observe(agent_id) 

        # Done?
        done = self.world.done()
        self.finished |= done

        # next valid actions
        nextActions = self._listNextValidActions(agent_id, action,episode=episode)

        # on_goal estimation
        on_goal = self.world.getPos(agent_id) == self.world.getGoal(agent_id)
        
        # Unlock mutex
        self.mutex.release()
        return state, reward, done, nextActions, on_goal, blocking, valid_action

    def _listNextValidActions(self, agent_id, prev_action=0,episode=0):
        available_actions = [0] # staying still always allowed

        # Get current agent position
        agent_pos = self.world.getPos(agent_id)
        ax,ay     = agent_pos[0],agent_pos[1]
        n_moves   = 9 if self.DIAGONAL_MOVEMENT else 5

        for action in range(1,n_moves):
            direction = self.world.getDir(action)
            dx,dy     = direction[0],direction[1]
            if(ax+dx>=self.world.state.shape[0] or ax+dx<0 or ay+dy>=self.world.state.shape[1] or ay+dy<0):#out of bounds
                continue
            if(self.world.state[ax+dx,ay+dy]<0):#collide with static obstacle
                continue
            if(self.world.state[ax+dx,ay+dy]>0):#collide with robot
                continue
            # check for diagonal collisions
            if(self.DIAGONAL_MOVEMENT):
                if self.world.diagonalCollision(agent_id,(ax+dx,ay+dy)):
                    continue          
            #otherwise we are ok to carry out the action
            available_actions.append(action)

        if opposite_actions[prev_action] in available_actions:
            available_actions.remove(opposite_actions[prev_action])
                
        return available_actions

    def drawStar(self, centerX, centerY, diameter, numPoints, color):
        outerRad=diameter//2
        innerRad=int(outerRad*3/8)
        #fill the center of the star
        angleBetween=2*math.pi/numPoints#angle between star points in radians
        for i in range(numPoints):
            #p1 and p3 are on the inner radius, and p2 is the point
            pointAngle=math.pi/2+i*angleBetween
            p1X=centerX+innerRad*math.cos(pointAngle-angleBetween/2)
            p1Y=centerY-innerRad*math.sin(pointAngle-angleBetween/2)
            p2X=centerX+outerRad*math.cos(pointAngle)
            p2Y=centerY-outerRad*math.sin(pointAngle)
            p3X=centerX+innerRad*math.cos(pointAngle+angleBetween/2)
            p3Y=centerY-innerRad*math.sin(pointAngle+angleBetween/2)
            #draw the triangle for each tip.
            poly=rendering.FilledPolygon([(p1X,p1Y),(p2X,p2Y),(p3X,p3Y)])
            poly.set_color(color[0],color[1],color[2])
            poly.add_attr(rendering.Transform())
            self.viewer.add_onetime(poly)

    def create_rectangle(self,x,y,width,height,fill,permanent=False):
        ps=[(x,y),((x+width),y),((x+width),(y+height)),(x,(y+height))]
        rect=rendering.FilledPolygon(ps)
        rect.set_color(fill[0],fill[1],fill[2])
        rect.add_attr(rendering.Transform())
        if permanent:
            self.viewer.add_geom(rect)
        else:
            self.viewer.add_onetime(rect)
    def create_circle(self,x,y,diameter,size,fill,resolution=20):
        c=(x+size/2,y+size/2)
        dr=math.pi*2/resolution
        ps=[]
        for i in range(resolution):
            x=c[0]+math.cos(i*dr)*diameter/2
            y=c[1]+math.sin(i*dr)*diameter/2
            ps.append((x,y))
        circ=rendering.FilledPolygon(ps)
        circ.set_color(fill[0],fill[1],fill[2])
        circ.add_attr(rendering.Transform())
        self.viewer.add_onetime(circ)
    def initColors(self):
        c={a+1:hsv_to_rgb(np.array([a/float(self.num_agents),1,1])) for a in range(self.num_agents)}
        return c

    def _render(self, mode='human',close=False,screen_width=800,screen_height=800,action_probs=None):
        if close == True:
            return
        #values is an optional parameter which provides a visualization for the value of each agent per step
        size=screen_width/max(self.world.state.shape[0],self.world.state.shape[1])
        colors=self.initColors()
        if self.viewer==None:
            self.viewer=rendering.Viewer(screen_width,screen_height)
            self.reset_renderer=True
        if self.reset_renderer:
            self.create_rectangle(0,0,screen_width,screen_height,(.6,.6,.6),permanent=True)
            for i in range(self.world.state.shape[0]):
                start=0
                end=1
                scanning=False
                write=False
                for j in range(self.world.state.shape[1]):
                    if(self.world.state[i,j]!=-1 and not scanning):#free
                        start=j
                        scanning=True
                    if((j==self.world.state.shape[1]-1 or self.world.state[i,j]==-1) and scanning):
                        end=j+1 if j==self.world.state.shape[1]-1 else j
                        scanning=False
                        write=True
                    if write:
                        x=i*size
                        y=start*size
                        self.create_rectangle(x,y,size,size*(end-start),(1,1,1),permanent=True)
                        write=False
        for agent in range(1,self.num_agents+1):
            i,j=self.world.getPos(agent)
            x=i*size
            y=j*size
            color=colors[self.world.state[i,j]]
            self.create_rectangle(x,y,size,size,color)
            i,j=self.world.getGoal(agent)
            x=i*size
            y=j*size
            color=colors[self.world.goals[i,j]]
            self.create_circle(x,y,size,size,color)
            if self.world.getGoal(agent)==self.world.getPos(agent):
                color=(0,0,0)
                self.create_circle(x,y,size,size,color)
        if action_probs is not None:
            n_moves=9 if self.DIAGONAL_MOVEMENT else 5
            for agent in range(1,self.num_agents+1):
                #take the a_dist from the given data and draw it on the frame
                a_dist=action_probs[agent-1]
                if a_dist is not None:
                    for m in range(n_moves):
                        dx,dy=self.world.getDir(m)
                        x=(self.world.getPos(agent)[0]+dx)*size
                        y=(self.world.getPos(agent)[1]+dy)*size
                        s=a_dist[m]*size
                        self.create_circle(x,y,s,size,(0,0,0))
        self.reset_renderer=False
        result=self.viewer.render(return_rgb_array = mode=='rgb_array')
        return result

if __name__=='__main__':
    n_agents=8
    env=MAPFEnv(n_agents,PROB=(.3,.5),SIZE=(10,11),DIAGONAL_MOVEMENT=False)
    print(coordinationRatio(env))
