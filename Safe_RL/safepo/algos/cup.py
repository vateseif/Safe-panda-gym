import torch
from Safe_RL.safepo.algos.policy_graident import PG
import Safe_RL.safepo.common.mpi_tools as mpi_tools

class CUP(PG):
    """
        Paper Name: CUP: A Conservative Update Policy Algorithm for Safe Reinforcement Learning
        Paper author: Long Yang, Jiaming Ji, Juntao Dai, Yu Zhang, Pengfei Li, Gang Pan
        Paper URL: https://arxiv.org/abs/2202.07565

        This implementation 
    """
    def __init__(
            self,
            algo: str = 'cup',
            clip: float = 0.2,
            cost_limit: float = 25.0,
            delta: float = 0.02,
            nu_lr: float = 5e-2,
            nu_max: float = 2.0,
            **kwargs
    ):
        super().__init__(algo=algo,
                        cost_limit=cost_limit, 
                        use_cost_value_function=True,
                        use_kl_early_stopping=True, 
                        **kwargs)
        self.clip = clip
        self.cost_limit = cost_limit
        self.delta = delta
        self.nu = 0
        self.nu_lr = nu_lr
        self.nu_max = nu_max

    def compute_loss_pi(self, data: dict):
        # Policy loss
        dist_1, _log_p_1 = self.ac.pi(data['obs'], data['act'])
        ratio_1 = torch.exp(_log_p_1 - data['log_p'])
        ratio_1_clip = torch.clamp(ratio_1, 1-self.clip, 1+self.clip)
        loss_pi_1 = -(torch.min(ratio_1, ratio_1_clip) * data['adv']).mean()
        loss_pi_1 -= self.entropy_coef * dist_1.entropy().mean()

        # Useful extra info
        approx_kl_1 = .5 * (data['log_p'] - _log_p_1).mean().item()
        ent_1 = dist_1.entropy().mean().item()
        pi_info = dict(kl=approx_kl_1, ent=ent_1, ratio=ratio_1.mean().item())

        return loss_pi_1, pi_info

    def post_compute_loss_pi(self, data: dict, ep_costs,p_dist_pre):
        # Policy loss


        dist_2, _log_p_2 = self.ac.pi(data['obs'], data['act'])
        ratio_2 = torch.exp(_log_p_2 - data['log_p'])
        approx_kl_2 = .5 * (data['log_p'] - _log_p_2).mean().item()

        c_loss_coef = (1 - self.buf.gamma * self.buf.lam_c) / (1 - self.buf.gamma)

        self.nu += self.nu_lr * (ep_costs - self.cost_limit)
        if self.nu < 0:
                self.nu = 0
        elif self.nu > self.nu_max:
            self.nu = self.nu_max

        # kl = torch.distributions.kl.kl_divergence(p_dist_pre, dist_2).mean()

        loss_pi_2 = (self.nu * c_loss_coef * ratio_2 * data['cost_adv']).mean() +\
                    torch.distributions.kl.kl_divergence(p_dist_pre, dist_2).mean()

        # Useful extra info

        ent_2= dist_2.entropy().mean().item()
        pi_info = dict(kl=approx_kl_2, ent=ent_2, ratio=ratio_2.mean().item())

        return loss_pi_2, pi_info

    def post_update_policy_net(self, data):
        # get prob. distribution before updates: used to measure KL distance
        ep_costs = self.logger.get_stats('EpCosts')[0]

        with torch.no_grad():
            self.p_dist = self.ac.pi.dist(data['obs']) # for focops, this is ?

        # Get loss and info values before update
        pi_l_old, pi_info_old = self.post_compute_loss_pi(data,ep_costs,p_dist_pre =self.p_dist)
        self.loss_pi_before = pi_l_old.item()
        if self.use_cost_value_function:
            self.loss_c_before = self.compute_loss_c(data['obs'],
                                                     data['target_c']).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iterations):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.post_compute_loss_pi(data,ep_costs,p_dist_pre =self.p_dist)
            loss_pi.backward()
            if self.use_max_grad_norm:  # apply L2 norm
                torch.nn.utils.clip_grad_norm_(
                    self.ac.pi.parameters(),
                    self.max_grad_norm)
            # average grads across MPI processes
            mpi_tools.mpi_avg_grads(self.ac.pi.net)
            self.pi_optimizer.step()
            q_dist = self.ac.pi.dist(data['obs'])
            torch_kl = torch.distributions.kl.kl_divergence(
                self.p_dist, q_dist).mean().item()
            if self.use_kl_early_stopping:
                # average KL for consistent early stopping across processes
                if mpi_tools.mpi_avg(torch_kl) > self.target_kl:
                    self.logger.log(f'Reached ES criterion after {i+1} steps.')
                    break
                    # track when policy iteration is stopped; Log changes from update
        self.logger.store(**{
            'Loss/Pi_2': self.loss_pi_before,
            'Loss/DeltaPi_2': loss_pi.item() - self.loss_pi_before,
            'Misc/StopIter_2': i + 1,
            'Values/Adv_2': data['adv'].numpy(),
            'Entropy_2': pi_info['ent'],
            'KL_2': torch_kl,
            'PolicyRatio_2': pi_info['ratio']
        })

    def update(self):
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # Note that logger already uses MPI statistics across all processes..
        # First update Lagrange multiplier parameter
        # now update policy and value network
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        self.update_policy_net(data=data)
        self.post_update_policy_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)