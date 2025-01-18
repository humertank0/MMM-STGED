import math
from collections import defaultdict

class STPoint:
    def __init__(self, lat, lng, time, data):
        self.lat = lat  # 纬度（浮点数）
        self.lng = lng  # 经度（浮点数）
        self.time = time  # 时间戳
        self.data = data  # 其他数据

class Trajectory:
    def __init__(self, oid, tid, pt_list):
        self.oid = oid  # 轨迹ID（字符串）
        self.tid = tid  # 时间戳（整数或字符串，具体取决于你的需求）
        self.pt_list = pt_list  # 轨迹点列表

class RoadNetwork:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

def distance(p1, p2):
    return math.sqrt((p1.lat - p2.lat) ** 2 + (p1.lng - p2.lng) ** 2)

class TimeStep:
    def __init__(self, pt, candidates=None):
        self.observation = pt
        self.candidates = candidates if candidates else []
        self.emission_log_probabilities = defaultdict(float)
        self.transition_log_probabilities = {}

def get_candidates(pt, rn, search_dis):
    # 假设 rn 是一个包含路网节点和边的数据结构
    candidates = []
    for node in rn.nodes:
        if distance(node, pt) <= search_dis:
            candidates.append(node)
    return candidates

class HMMProbabilities:
    def __init__(self, sigma=5.0, beta=2.0):
        self.sigma = sigma
        self.beta = beta

    def emission_log_probability(self, dist):
        return -0.5 * (dist / self.sigma) ** 2

    def transition_log_probability(self, path_dist, linear_dist):
        return -self.beta * abs(path_dist - linear_dist)

class ViterbiAlgorithm:
    def __init__(self, keep_message_history=False):
        self.last_extended_states = None
        self.prev_candidates = []
        self.message = defaultdict(float)
        self.is_broken = False
        self.transition_log_probabilities = {}

    def initialize_state_probabilities(self, observation, candidates, initial_log_probabilities):
        for candi_pt in candidates:
            self.message[candi_pt] = initial_log_probabilities[observation]

    def forward_step(self, observation, prev_candidates, cur_candidates, message,
                     emission_log_probabilities, transition_log_probabilities):
        new_message = defaultdict(float)
        self.last_extended_states = cur_candidates  # Update last extended states
        for cur_state in cur_candidates:
            max_log_probability = float('-inf')
            best_prev_state = None
            for prev_state in prev_candidates:
                log_probability = message[prev_state] + transition_log_probabilities.get((prev_state, cur_state), 0.0)
                if log_probability > max_log_probability:
                    max_log_probability = log_probability
                    best_prev_state = prev_state
            new_message[cur_state] = max_log_probability + emission_log_probabilities[cur_state]
        return new_message

    def compute_most_likely_sequence(self):
        seq = []
        current_state = None
        for state, prob in self.message.items():
            if current_state is None or prob > self.message[current_state]:
                current_state = state
        while current_state is not None:
            seq.append(current_state)
            next_state = None
            max_prob = float('-inf')
            for prev_state in (self.last_extended_states or []):  # Handle None case
                if (prev_state, current_state) in self.transition_log_probabilities and \
                   self.message[prev_state] + self.transition_log_probabilities[(prev_state, current_state)] == self.message[current_state]:
                    next_state = prev_state
                    break
            current_state = next_state
        return seq[::-1]

class TIHMMMapMatcher:
    def __init__(self, rn, search_dis=50, sigma=5.0, beta=2.0):
        self.rn = rn  # 路网数据结构
        self.measurement_error_sigma = search_dis  # 搜索距离
        self.transition_probability_beta = beta  # 转移概率参数
        self.guassian_sigma = sigma  # 高斯分布的σ值
        self.probabilities = HMMProbabilities(sigma, beta)

    def create_time_step(self, pt):
        candidates = get_candidates(pt, self.rn, self.measurement_error_sigma)
        if candidates:
            return TimeStep(pt, candidates)
        return None

    def compute_emission_probabilities(self, time_step):
        for candi_pt in time_step.candidates:
            dist = distance(candi_pt, time_step.observation)
            time_step.emission_log_probabilities[candi_pt] = self.probabilities.emission_log_probability(dist)

    def compute_transition_probabilities(self, prev_time_step, time_step):
        linear_dist = distance(prev_time_step.observation, time_step.observation)
        for prev_candi_pt in prev_time_step.candidates:
            for cur_candi_pt in time_step.candidates:
                path_dist = distance(prev_candi_pt, cur_candi_pt)
                transition_prob = self.probabilities.transition_log_probability(path_dist, linear_dist)
                time_step.transition_log_probabilities[(prev_candi_pt, cur_candi_pt)] = transition_prob

    def match(self, traj):
        viterbi = ViterbiAlgorithm()
        prev_time_step = None
        for pt in traj.pt_list:
            time_step = self.create_time_step(pt)
            if time_step is None:
                continue
            self.compute_emission_probabilities(time_step)
            if prev_time_step is None:
                viterbi.initialize_state_probabilities(time_step.observation, time_step.candidates,
                                                       {time_step.observation: 0.0})
            else:
                self.compute_transition_probabilities(prev_time_step, time_step)
                new_message = viterbi.forward_step(time_step.observation, prev_time_step.candidates,
                                                   time_step.candidates, viterbi.message,
                                                   time_step.emission_log_probabilities,
                                                   time_step.transition_log_probabilities)
                viterbi.message = new_message
            prev_time_step = time_step

        most_likely_sequence = viterbi.compute_most_likely_sequence()
        matched_points = [STPoint(pt.lat, pt.lng, pt.time, {'candi_pt': seq}) for pt, seq in zip(traj.pt_list, most_likely_sequence)]
        return Trajectory(traj.oid, traj.tid, matched_points)

# 示例数据
nodes = [
    STPoint(37.4219086, -122.156369, None, {}),
    STPoint(37.4219295, -122.156372, None, {})
]
edges = [(nodes[0], nodes[1])]
rn = RoadNetwork(nodes, edges)

gps_trajectory_points = [
    STPoint(37.4219086, -122.156369, '2023-10-01T10:00:00', {}),
    STPoint(37.4219295, -122.156372, '2023-10-01T10:01:00', {}),
]

traj = Trajectory('trajectory_1', 1, gps_trajectory_points)

# 创建地图匹配器实例
map_matcher = TIHMMMapMatcher(rn)

# 匹配轨迹并输出结果
mm_traj = map_matcher.match(traj)

# 打印匹配后的轨迹点
for pt in mm_traj.pt_list:
    print(f"Lat: {pt.lat}, Lon: {pt.lng}, Time: {pt.time}, Candidate Point: {pt.data['candi_pt']}")
