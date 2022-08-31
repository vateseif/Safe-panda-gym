from Safe_RL.safepo.algos.policy_graident import PG
from Safe_RL.safepo.algos.natural_pg import NPG
from Safe_RL.safepo.algos.trpo import TRPO
from Safe_RL.safepo.algos.ppo import PPO
from Safe_RL.safepo.algos.trpo_lagrangian import TRPO_Lagrangian
from Safe_RL.safepo.algos.ppo_lagrangian import PPO_Lagrangian
from Safe_RL.safepo.algos.cpo import CPO
from Safe_RL.safepo.algos.pcpo import PCPO
from Safe_RL.safepo.algos.focops import FOCOPS
from Safe_RL.safepo.algos.p3o import P3O
from Safe_RL.safepo.algos.cup import CUP
from Safe_RL.safepo.algos.sppo import SPPO
from Safe_RL.safepo.algos.lcpo import LCPO
from Safe_RL.safepo.algos.ipo import IPO
from Safe_RL.safepo.algos.cppo_pid import CPPOPid
# import sys
# sys.path.append('/Users/emma/dev/panda-gym/Safe_RL')

REGISTRY = {
    'pg': PG,
    'npg': NPG,
    'trpo': TRPO,
    'ppo': PPO,
    'trpo_lagrangian':TRPO_Lagrangian,
    'ppo_lagrangian': PPO_Lagrangian,
    'cpo': CPO,
    'pcpo': PCPO,
    'focops': FOCOPS,
    'p3o': P3O,
    'cup': CUP,
    'sppo': SPPO,
    'lcpo':LCPO,
    'ipo': IPO,
    'cppo-pid': CPPOPid,
}
