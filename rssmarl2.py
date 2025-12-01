"""
RS-SQL 示例（去掉在线预测模块版本）：

功能：
1）构造 S0~S10 的攻击图环境，S6~S10 为 PLC 被攻陷（终止态）
2）定义攻击动作 a0~a13、防御动作 d0~d13，并按题目给定 A0~A5、D0~D5 约束各状态可行动作
3）实现角色切换随机–Stackelberg Q-learning（RS-SQL）
   —— 此处假定情境预测模块已离线完成，给出了固定的预部署防御集合：
        predeployed_defenses = [1, 3, 6]  （分别加固节点1、3、6）
   —— 在环境中，这些预部署防御始终生效：若攻击企图推进到 1/3/6，则被“提前拦截”
4）记录：
   - 状态 s1 下攻击者领导动作的策略演化（各动作被选为 a* 的频率）
   - 状态 s2 下攻击者期望收益的演化 Q_A(s2,a0/a4/a5) 及 J_A(s2)
5）使用 matplotlib 绘图
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import random
import matplotlib.pyplot as plt


# =========================
# 1. 攻击图 + 环境定义
# =========================

@dataclass
class AttackAction:
    """
    攻击动作:
    - name: a0..a13
    - cost: 一次执行该动作的攻击成本
    - transitions: {src_state -> dst_state}, 表示在不同 src 下的转移
    """
    name: str
    cost: float
    transitions: Dict[int, int]


@dataclass
class DefenseAction:
    """
    防御动作:
    - name: d0..d13
    - real_cost: 实时防御成本（Reactive）
    - pre_cost:  预部署防御成本（Proactive），用于计算固定预部署的基础开销
    - target_state: 该动作主要防护/阻断的目标状态；为 None 表示不针对特定状态
    """
    name: str
    real_cost: float
    pre_cost: float
    target_state: Optional[int]


class ICSGameEnv:
    """
    工业网络攻防环境（基于图1攻击图 & 图2动作集合）:

    - 状态 s ∈ {0,1,...,10}
        s0: 互联网侧攻击起点
        s1~s5: 企业 PC / Web / ES / HMI / OS 等关键资产
        s6~s10: PLC1~PLC5 被攻陷 —— 视为攻击成功的终止态
    - 攻击动作 a ∈ {0,...,13}:
        a0: no-op
        a1~a10: 纵向渗透（利用 v1~v10）
        a11~a13: 横向移动（1→2, 3→4, 4→5）
    - 防御动作 d ∈ {0,...,13}:
        d0: no-op
        d1~d10: 修复漏洞 v1~v10、阻止访问对应状态 1~10
        d11~d13: 阻止横向移动到 2、4、5

    说明：
    - 本版本不再包含在线情境预测模块，假定预测模块离线确定了要长期加固的节点集合：
        predeployed_defenses = [1, 3, 6]
      即对节点1、3、6预先部署防御。
    - 在环境中，这些预部署防御始终生效：
        只要攻击试图推进到这些节点，就会被“提前拦截”，状态保持不变。
    - 预部署防御的基础成本在每一步以常数形式计入防御方成本（对策略无影响，只是整体平移）。
    """

    def __init__(
        self,
        num_states: int,
        attack_actions: List[AttackAction],
        defense_actions: List[DefenseAction],
        initial_state: int,
        terminal_states: List[int],
        predeployed_defenses: Optional[List[int]] = None,
    ):
        self.num_states = num_states
        self.attack_actions = attack_actions
        self.defense_actions = defense_actions
        self.initial_state = initial_state
        self.terminal_states = set(terminal_states)

        # 动作空间规模
        self.num_attack_actions = len(attack_actions)
        self.num_defense_actions = len(defense_actions)

        # 假定 index 0 分别为 no-op
        self.attack_noop = 0
        self.defense_noop = 0

        # 成本向量
        self.attack_costs = np.array([a.cost for a in attack_actions], dtype=float)
        self.defense_real_costs = np.array(
            [d.real_cost for d in defense_actions], dtype=float
        )
        self.defense_pre_costs = np.array(
            [d.pre_cost for d in defense_actions], dtype=float
        )

        # 默认：在非终止态所有动作都可选，在终止态只允许 no-op
        self.valid_attack_actions: Dict[int, List[int]] = {}
        self.valid_defense_actions: Dict[int, List[int]] = {}
        for s in range(num_states):
            if s in self.terminal_states:
                self.valid_attack_actions[s] = [self.attack_noop]
                self.valid_defense_actions[s] = [self.defense_noop]
            else:
                self.valid_attack_actions[s] = list(range(self.num_attack_actions))
                self.valid_defense_actions[s] = list(range(self.num_defense_actions))

        # ---------- 固定预部署防御 ----------
        # 给出固定的预部署防御动作索引列表，如 [1,3,6] 对应 d1,d3,d6
        self.predeployed_defenses: List[int] = predeployed_defenses or []

        # 这些预部署动作针对的目标状态集合（例如 {1,3,6}）
        self.predeploy_target_states = set()
        self.base_predeploy_cost = 0.0
        for d_idx in self.predeployed_defenses:
            if 0 <= d_idx < len(self.defense_actions):
                d = self.defense_actions[d_idx]
                if d.target_state is not None:
                    self.predeploy_target_states.add(d.target_state)
                self.base_predeploy_cost += d.pre_cost

    # ---------- 基本接口 ----------

    def reset(self) -> int:
        """复位环境到初始情境状态 s0"""
        return self.initial_state

    # ---------- 带固定预部署的一步博弈演化 ----------

    def step(
        self,
        s: int,
        a_real: int,
        d_real: int,
    ) -> Tuple[int, float, float, bool]:
        """
        执行一轮攻防博弈, 返回:
          - s_next: 下一真实状态
          - R_A:   攻击者即时奖励
          - R_D:   防御者即时奖励
          - done:  是否到达终止态

        流程：
        1）攻击者在状态 s 发起攻击 a_real，得到候选后继 s_raw；
        2）若 s_raw 属于预部署加固的目标状态集合，则视为预部署命中：
            - 攻击被提前拦截，s_next = s；
        3）否则，允许实时防御 d_real 尝试拦截；
        4）若均未拦截成功，攻击推进到 s_raw。
        """

        # ===== 阶段 1: 攻击推进（不考虑防御） =====
        s_raw = s
        attack_used = (a_real != self.attack_noop)

        if attack_used:
            act = self.attack_actions[a_real]
            if s in act.transitions:
                s_raw = act.transitions[s]  # 有效攻击推进
            # 否则：攻击无效，但仍然花费成本

        # ===== 固定预部署防御成本（常数项） =====
        C_pre = self.base_predeploy_cost
        predeploy_hit = False

        # 若攻击有效且目标是预部署加固的节点，则视为“提前拦截”
        if attack_used and (s_raw != s) and (s_raw in self.predeploy_target_states):
            predeploy_hit = True

        # ===== 实时防御 =====
        C_real = 0.0
        defense_hit_real = False

        if predeploy_hit:
            # 预测好的预部署已经生效：攻击被提前拦截
            s_next = s
            attack_success = False
        else:
            # 否则允许实时防御 d_real 进行拦截
            if d_real != self.defense_noop:
                C_real = self.defense_real_costs[d_real]
                target_real = self.defense_actions[d_real].target_state
                if (
                    target_real is not None
                    and s_raw == target_real
                    and attack_used
                    and s_raw != s
                ):
                    defense_hit_real = True

            if defense_hit_real:
                s_next = s
                attack_success = False
            else:
                s_next = s_raw
                attack_success = attack_used and (s_raw != s)

        # ===== 成本与奖励 =====
        C_A = self.attack_costs[a_real] if attack_used else 0.0
        C_D = C_pre + C_real

        R_A = -C_A
        R_D = -C_D

        done = s_next in self.terminal_states
        if done and attack_success:
            # 攻击成功到达终止态: 攻击者 +R, 防御者 -R
            success_reward = 500.0
            R_A += success_reward
            R_D -= success_reward

        return s_next, R_A, R_D, done


# ================================
# 2. 角色切换随机–Stackelberg MARL
# ================================

class RSStackelbergMARL:
    """
    角色切换随机–Stackelberg Q-learning (RS-SQL) —— 无在线预测版本：

    - 仍使用联合 Q_A(s,a,d), Q_D(s,a,d) 近似“攻击者为 Leader”的局部 Stackelberg 均衡
    - 差别在于：
        * 预测模块已从代码中剥离，不再在线调用 INSSP；
        * 预部署防御由 env 内的 predeployed_defenses 固定给定（如 [1,3,6]），不需要算法内再做选择。
    """

    def __init__(
        self,
        env: ICSGameEnv,
        gamma: float = 0.9,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        episodes: int = 1000,
    ):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes

        S = env.num_states
        A = env.num_attack_actions
        D = env.num_defense_actions

        # 联合 Q 表: Q_A, Q_D ∈ R^{S × A × D}
        self.QA = np.random.randn(S, A, D) * 1e-3
        self.QD = np.random.randn(S, A, D) * 1e-3

        # 用于分析的历史记录
        self.hist_s1_a_star: List[int] = []   # 每个episode结束时，s1下攻击者领导动作 a*
        self.hist_s1_d_star: List[int] = []   # 每个episode结束时，s1下防御者最佳响应 d*
        self.hist_s2_Qa0: List[float] = []    # s2 下 Q_A(s2,a0, d_BR)
        self.hist_s2_Qa4: List[float] = []    # s2 下 Q_A(s2,a4, d_BR)
        self.hist_s2_Qa5: List[float] = []    # s2 下 Q_A(s2,a5, d_BR)
        self.hist_s2_JA: List[float] = []     # s2 下 J_A(s2) = max_a Q_A(s2,a,d_BR)

    # ---------- 阶段 1: 真实态势下攻击者为 Leader ----------

    def local_attack_leader(self, s: int):
        """
        在状态 s 上求局部 Stackelberg 解 (攻击者为 Leader):

        1) 防御者对每个 a 的最佳响应:
           d_BR(a) = argmin_d Q_D(s,a,d)
        2) 攻击者在此基础上选:
           a* = argmax_a Q_A(s,a,d_BR(a))
        """
        QA_s = self.QA[s]  # [A,D]
        QD_s = self.QD[s]  # [A,D]

        valid_As = self.env.valid_attack_actions.get(
            s, list(range(self.env.num_attack_actions))
        )
        valid_Ds = self.env.valid_defense_actions.get(
            s, list(range(self.env.num_defense_actions))
        )

        QA_sub = QA_s[np.ix_(valid_As, valid_Ds)]
        QD_sub = QD_s[np.ix_(valid_As, valid_Ds)]

        # 防御者对每个 a 的 best response（在 valid_Ds 内）
        d_br_local = np.argmin(QD_sub, axis=1)  # shape [len(valid_As)]
        d_br_global = [valid_Ds[j] for j in d_br_local]

        # 攻击者在 d_BR 下的价值
        QA_lead = np.array(
            [QA_s[a, d_br_global[i]] for i, a in enumerate(valid_As)]
        )

        best_idx = int(np.argmax(QA_lead))
        a_star = valid_As[best_idx]
        d_star = d_br_global[best_idx]

        W_A = QA_s[a_star, d_star]
        W_D = QD_s[a_star, d_star]
        return a_star, d_star, W_A, W_D

    def select_real_actions(self, s: int) -> Tuple[int, int]:
        """
        在真实态势 s 上选择 (a_real, d_real)，带 ε-greedy 探索，
        但仅在该状态允许的动作子集中采样。
        """
        valid_As = self.env.valid_attack_actions.get(
            s, list(range(self.env.num_attack_actions))
        )
        valid_Ds = self.env.valid_defense_actions.get(
            s, list(range(self.env.num_defense_actions))
        )

        a_star, d_star, _, _ = self.local_attack_leader(s)

        # 攻击动作
        if random.random() < self.epsilon:
            a = random.choice(valid_As)
        else:
            a = a_star if a_star in valid_As else random.choice(valid_As)

        # 防御动作
        if random.random() < self.epsilon:
            d = random.choice(valid_Ds)
        else:
            d = d_star if d_star in valid_Ds else random.choice(valid_Ds)

        return a, d

    # ---------- 记录 s1 策略 & s2 收益 的辅助函数 ----------

    def record_metrics(self):
        """
        在每个 episode 结束后，记录：
        - 状态 s1 的局部 Stackelberg 策略 (a*, d*)
        - 状态 s2 下 Q_A(s2,a0/a4/a5) 及 J_A(s2) 的演化
        """
        # s1: 攻击者为 Leader 的局部 Stackelberg 解
        s1 = 1
        a1_star, d1_star, _, _ = self.local_attack_leader(s1)
        self.hist_s1_a_star.append(a1_star)
        self.hist_s1_d_star.append(d1_star)

        # s2: 各候选攻击动作在防御者 best-response 下的期望收益
        s2 = 2
        QA_s2 = self.QA[s2]
        QD_s2 = self.QD[s2]
        valid_As2 = self.env.valid_attack_actions[s2]  # [0,4,5]
        valid_Ds2 = self.env.valid_defense_actions[s2]  # [0,4,5]

        vals = {}
        for a in valid_As2:
            # 防御者在 valid_Ds2 内对该 a 的最佳响应
            QD_row = QD_s2[a, valid_Ds2]
            j_local = int(np.argmin(QD_row))
            d_br = valid_Ds2[j_local]
            vals[a] = float(QA_s2[a, d_br])

        qa0 = vals.get(0, 0.0)
        qa4 = vals.get(4, 0.0)
        qa5 = vals.get(5, 0.0)
        J = max(vals.values()) if vals else 0.0

        self.hist_s2_Qa0.append(qa0)
        self.hist_s2_Qa4.append(qa4)
        self.hist_s2_Qa5.append(qa5)
        self.hist_s2_JA.append(J)

    # ---------- 训练主循环 ----------

    def train(self):
        for ep in range(self.episodes):
            s = self.env.reset()
            steps = 0

            while True:
                # 1) 在真实态势 s 上，攻击者为 Leader，选 (a_real, d_real)
                a_real, d_real = self.select_real_actions(s)

                # 2) 环境执行一步博弈（内部自动考虑固定预部署）
                s_next, R_A, R_D, done = self.env.step(
                    s, a_real, d_real
                )

                # 3) 下一真实态势上, 再求“攻击者为 Leader”的局部 Stackelberg 价值
                _, _, W_A_next, W_D_next = self.local_attack_leader(s_next)

                # 4) 时序差分更新 (真实态势下的真实动作对)
                self.QA[s, a_real, d_real] = (
                    (1 - self.alpha) * self.QA[s, a_real, d_real]
                    + self.alpha * (R_A + self.gamma * W_A_next)
                )

                self.QD[s, a_real, d_real] = (
                    (1 - self.alpha) * self.QD[s, a_real, d_real]
                    + self.alpha * (R_D + self.gamma * W_D_next)
                )

                s = s_next
                steps += 1

                if done or steps > 100:
                    break

            # 一个 episode 结束后记录一次度量
            self.record_metrics()

    # ---------- 提取最终策略 ----------

    def extract_policies(self) -> Dict[int, Tuple[int, int]]:
        """
        返回每个状态 s 的局部 Stackelberg 策略 (a_R^*(s), d_R^*(s))，
        对应真实态势下的攻防对抗行为。
        """
        S = self.env.num_states
        policies: Dict[int, Tuple[int, int]] = {}
        for s in range(S):
            a_star, d_star, _, _ = self.local_attack_leader(s)
            policies[s] = (a_star, d_star)
        return policies


# =========================
# 3. 基于图 1 & 图 2 的示例环境
# =========================

def build_example_env() -> ICSGameEnv:
    """
    构造与论文图 1/图 2 完全对应的示例环境：
    - 状态 0~10，其中 6~10 为终止态
    - 攻击动作 a0~a13
    - 防御动作 d0~d13
    - 各状态下可用的 A_s, D_s 按题目给定配置
    - 固定预部署防御集合 predeployed_defenses = [1, 3, 6]
      —— 分别对节点 1、3、6 进行提前加固
    """
    num_states = 11  # 0..10

    # ---- 攻击动作定义（索引即为动作编号） ----
    attack_actions: List[AttackAction] = [
        AttackAction("a0_noop", 0.0, {}),                         # 0: no-op
        AttackAction("a1_v1_to_1", 10.0, {0: 1}),                 # 1
        AttackAction("a2_v2_to_2", 10.0, {0: 2}),                 # 2
        AttackAction("a3_v3_to_3", 10.0, {1: 3}),                 # 3
        AttackAction("a4_v4_to_4", 10.0, {1: 4, 2: 4}),           # 4
        AttackAction("a5_v5_to_5", 10.0, {2: 5}),                 # 5
        AttackAction("a6_v6_to_6", 10.0, {3: 6}),                 # 6
        AttackAction("a7_v7_to_7", 10.0, {3: 7, 4: 7}),           # 7
        AttackAction("a8_v8_to_8", 10.0, {4: 8, 5: 8}),           # 8
        AttackAction("a9_v9_to_9", 10.0, {5: 9}),                 # 9
        AttackAction("a10_v10_to_10", 10.0, {5: 10}),             # 10
        AttackAction("a11_move_1_to_2", 5.0, {1: 2}),             # 11
        AttackAction("a12_move_3_to_4", 5.0, {3: 4}),             # 12
        AttackAction("a13_move_4_to_5", 5.0, {4: 5}),             # 13
    ]

    # ---- 防御动作定义 ----
    defense_actions: List[DefenseAction] = [
        DefenseAction("d0_noop", 0.0, 0.0, None),                # 0: no-op
        DefenseAction("d1_fix_v1_block_1", 10.0, 8.0, 1),        # 1
        DefenseAction("d2_fix_v2_block_2", 10.0, 8.0, 2),        # 2
        DefenseAction("d3_fix_v3_block_3", 10.0, 8.0, 3),        # 3
        DefenseAction("d4_fix_v4_block_4", 10.0, 8.0, 4),        # 4
        DefenseAction("d5_fix_v5_block_5", 10.0, 8.0, 5),        # 5
        DefenseAction("d6_fix_v6_block_6", 10.0, 8.0, 6),        # 6
        DefenseAction("d7_fix_v7_block_7", 10.0, 8.0, 7),        # 7
        DefenseAction("d8_fix_v8_block_8", 10.0, 8.0, 8),        # 8
        DefenseAction("d9_fix_v9_block_9", 10.0, 8.0, 9),        # 9
        DefenseAction("d10_fix_v10_block_10", 10.0, 8.0, 10),    # 10
        DefenseAction("d11_block_lateral_to_2", 5.0, 4.0, 2),    # 11
        DefenseAction("d12_block_lateral_to_4", 5.0, 4.0, 4),    # 12
        DefenseAction("d13_block_lateral_to_5", 5.0, 4.0, 5),    # 13
    ]

    # 固定预部署防御集合：d1, d3, d6
    predeployed_defenses = [1, 3, 6]

    env = ICSGameEnv(
        num_states=num_states,
        attack_actions=attack_actions,
        defense_actions=defense_actions,
        initial_state=0,
        terminal_states=[6, 7, 8, 9, 10],
        predeployed_defenses=predeployed_defenses,
    )

    # ---- 将题目给定的 A_s, D_s 塞进环境 ----
    env.valid_attack_actions = {
        0: [0, 1, 2],          # A0 = {a0,a1,a2}
        1: [0, 3, 4, 11],      # A1 = {a0,a3,a4,a11}
        2: [0, 4, 5],          # A2 = {a0,a4,a5}
        3: [0, 6, 7, 12],      # A3 = {a0,a6,a7,a12}
        4: [0, 7, 8, 13],      # A4 = {a0,a7,a8,a13}
        5: [0, 8, 9, 10],      # A5 = {a0,a8,a9,a10}
        6: [0],                # 终止态只允许 no-op
        7: [0],
        8: [0],
        9: [0],
        10: [0],
    }

    env.valid_defense_actions = {
        0: [0, 1, 2],          # D0 = {d0,d1,d2}
        1: [0, 3, 4, 11],      # D1 = {d0,d3,d4,d11}
        2: [0, 4, 5],          # D2 = {d0,d4,d5}
        3: [0, 6, 7, 12],      # D3 = {d0,d6,d7,d12}
        4: [0, 7, 8, 13],      # D4 = {d0,d7,d8,d13}
        5: [0, 8, 9, 10],      # D5 = {d0,d8,d9,d10}
        6: [0],
        7: [0],
        8: [0],
        9: [0],
        10: [0],
    }

    return env


# =========================
# 4. 可视化函数
# =========================

def plot_s1_strategy_evolution(agent: RSStackelbergMARL):
    """
    画出状态 s1 下攻击者领导动作 a* 的演化：
    对 A1={a0,a3,a4,a11}，统计到当前 episode 为止每个动作被选为 a* 的累计频率。
    """
    episodes = np.arange(1, len(agent.hist_s1_a_star) + 1)
    actions_s1 = [0, 3, 4, 11]  # A1
    labels = {0: "a0 (no-op)", 3: "a3 (1→3)", 4: "a4 (1→4)", 11: "a11 (1→2)"}

    plt.figure()
    for a in actions_s1:
        counts = 0
        freq = []
        for i, a_star in enumerate(agent.hist_s1_a_star):
            if a_star == a:
                counts += 1
            freq.append(counts / (i + 1))
        plt.plot(episodes, freq, label=labels[a])
    plt.xlabel("Episode")
    plt.ylabel("Frequency of being leader action at s1")
    plt.title("Strategy evolution at state s1 (attack leader)")
    plt.legend()
    plt.grid(True)


def plot_s2_payoff_evolution(agent: RSStackelbergMARL):
    """
    画出状态 s2 下攻击者期望收益的演化：
    - Q_A(s2,a0,d_BR)
    - Q_A(s2,a4,d_BR)
    - Q_A(s2,a5,d_BR)
    - J_A(s2) = max_{a∈A2} Q_A(s2,a,d_BR)
    """
    episodes = np.arange(1, len(agent.hist_s2_JA) + 1)

    plt.figure()
    plt.plot(episodes, agent.hist_s2_Qa0, label="Q_A(s2,a0)")
    plt.plot(episodes, agent.hist_s2_Qa4, label="Q_A(s2,a4)")
    plt.plot(episodes, agent.hist_s2_Qa5, label="Q_A(s2,a5)")
    plt.plot(episodes, agent.hist_s2_JA, label="J_A(s2)=max_a Q_A(s2,a)", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Expected payoff evolution at state s2 (attack leader)")
    plt.legend()
    plt.grid(True)


# =========================
# 5. 主程序
# =========================

if __name__ == "__main__":
    # 构造环境与智能体
    env = build_example_env()
    agent = RSStackelbergMARL(
        env,
        gamma=0.95,
        alpha=0.05,
        epsilon=0.1,
        episodes=1000,   # 可按需调大
    )

    # 训练
    agent.train()

    # 打印最终局部 Stackelberg 策略
    policies = agent.extract_policies()
    print("学习到的局部 Stackelberg 策略 (state -> (a_R^*(s), d_R^*(s))):")
    for s, (a_star, d_star) in policies.items():
        print(f"  state {s}: a*={a_star}, d*={d_star}")

    # 画图：s1 策略演化 & s2 期望收益演化
    plot_s1_strategy_evolution(agent)
    plot_s2_payoff_evolution(agent)

    plt.show()
