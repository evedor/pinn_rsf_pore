# src/pinn/back_solver.py
import torch
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from network import Network


class FrictionPINNSolver:
    def __init__(self, config_path: str = "config.yaml",
                 experiment: str = None):
        self.config = self._load_config(config_path)
        self.experiment_key = experiment

        self._override_experiment()
        self._set_seed()
        self.device = torch.device(self.config["training"]["device"])

        self._init_physics_parameters()
        self._init_model()
        self._init_data()
        self._init_optimizers()
        self._init_logging()
        self._init_folder()

    def _init_folder(self):
        self.save_dir = Path('./{}'.format(self.experiment_key))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _set_seed(self):
        seed = self.config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _init_physics_parameters(self):
        cfg = self.config["physics"]
        self.v_0    = cfg["v_0"]
        self.k      = cfg["k"]
        self.beta_1 = cfg["beta_1"]
        self.L_1    = cfg["L_1"]
        self.rho    = cfg["rho"]
        self.a      = cfg["a"]
        self.mu_0   = cfg["mu_0"]
        self.G      = cfg["G"]
        self.rho_v  = cfg["rho_v"]
        self.p_0    = cfg["p_0"]
        self.beta_a = cfg["beta_a"]
        self.beta_m = cfg["beta_m"]
        self.phi_0  = cfg["phi_0"]
        self.epsilon= cfg["epsilon"]
        self.c_0    = cfg["c_0"]

        # 派生参数
        self.beta = self.phi_0*(self.beta_a+self.beta_m)
        self.L_2   = self.rho * self.L_1
        self.lam   = self.a / self.mu_0
        self.tau_0 = self.mu_0 * self.sigma_n0
        self.kappa = (self.k * self.L_1) / (self.a * self.sigma_n0)
        self.nu    = (self.G / (2 * np.sqrt(self.G / self.rho_v))) * self.v_0 / (self.a * self.sigma_n0)
        self.beta_2= -self.epsilon / (self.lam * self.beta * self.sigma_n0)
        self.alpha = (self.c_0 * self.p_0 * self.L_1) / (self.v_0 * self.lam * self.sigma_n0)
        self.gamma = self.c_0 * self.L_1 / self.v_0

    def _init_model(self):
        mcfg = self.config["training"]["model"]
        self.model = Network(
            input_size=mcfg["input_size"],
            hidden_size=mcfg["hidden_size"],
            output_size=mcfg["output_size"],
            depth=mcfg["depth"],
            act=torch.nn.Tanh if mcfg["activation"] == "Tanh" else torch.nn.Tanh
        ).to(self.device)

    def _override_experiment(self):
        self.experiment = self.config.get("experiments", {})
        exp = self.experiment[self.experiment_key]
        self.sigma_n0 = exp["sigma_n0"]
        self.t0_values =  exp["t0_values"]

    def _init_data(self):
        tcfg = self.config["training"]["time"]
        self.dt = tcfg["step"]
        t = torch.arange(tcfg["start"], tcfg["end"], self.dt, device=self.device)
        self.X_inside = t.reshape(1, -1).T

        # 使用实验指定的初始条件
        ic = t[0].reshape(1, -1).T  # t=0边界
        self.X_boundary = ic

        u_ic = torch.tensor(self.t0_values)  # t=0边界处采用第一类边界条件u=-sin(pi*x)
        self.U_boundary = torch.cat([u_ic])

        self.X_inside = self.X_inside.to(self.device)
        self.X_boundary = self.X_boundary.to(self.device)
        self.U_boundary = self.U_boundary.to(self.device)
        self.X_inside.requires_grad = True  # 设置：需要计算对X的梯度

        self.criterion = torch.nn.MSELoss()

    def _init_optimizers(self):
        ocfg = self.config["training"]["optimizer"]

        self.adam = torch.optim.Adam(
            self.model.parameters(),
            lr=ocfg["adam"]["lr"]
        )

        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=ocfg["lbfgs"]["lr"],
            max_iter=ocfg["lbfgs"]["max_iter"],
            max_eval=ocfg["lbfgs"]["max_eval"],
            history_size=ocfg["lbfgs"]["history_size"],
            tolerance_grad=ocfg["lbfgs"]["tolerance_grad"],
            tolerance_change=ocfg["lbfgs"]["tolerance_change"],
            line_search_fn=ocfg["lbfgs"]["line_search_fn"]
        )

    def _init_logging(self):
        self.iter = 1
        self.min_loss = float('inf')
        self.best_model_state = None
        self.loss_equation0_history = []
        self.loss_boundary_history = []

    # ── 核心残差计算 ────────────────────────────────────────
    def compute_residuals(self):
        U_inside = self.model(self.X_inside)
        x, y, z, u = U_inside[:, 0].squeeze(), U_inside[:, 1].squeeze(), U_inside[:, 2].squeeze(), U_inside[:,
                                                                                                   3].squeeze()

        du_dX_all = []
        for i in range(4):
            du_dX_i = torch.autograd.grad(
                inputs=self.X_inside,
                outputs=U_inside[:, i],
                grad_outputs=torch.ones_like(U_inside[:, i]),
                retain_graph=True,
                create_graph=True
            )[0]
            du_dX_all.append(du_dX_i.squeeze())

        x_t, y_t, z_t, u_t = du_dX_all[0], du_dX_all[1], du_dX_all[2], du_dX_all[3]

        residual_z = z_t - (-self.rho * (self.beta_2 * x + z) * torch.exp(x))
        residual_u = u_t - (-self.alpha - self.gamma * u + z_t)
        residual_x = x_t - (
                    torch.exp(x) * ((self.beta_1 - 1) * x * (1 + self.lam * u) + y - u) + self.kappa * (1 - torch.exp(x)) - u_t * (
                    1 + self.lam * y) / (1 + self.lam * u)) / (1 + self.lam * u + self.nu * torch.exp(x))
        residual_y = y_t - (self.kappa * (1 - torch.exp(x)) - self.nu * torch.exp(x) * x_t)

        return torch.stack([residual_x, residual_y, residual_z, residual_u], dim=1)

    def loss_func(self):
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        # 第一部分loss: 边界条件不吻合产生的loss
        U_pred_boundary = self.model(self.X_boundary)  # 使用当前模型计算u在边界处的预测值
        loss_boundary = self.criterion(
            U_pred_boundary, self.U_boundary)  # 计算边界处的MSE

        # 第二部分loss:内点非物理产生的loss
        U_inside = self.model(self.X_inside)  # 使用当前模型计算内点处的预测值

        x, y, z, u = U_inside[:, 0].squeeze(), U_inside[:, 1].squeeze(), U_inside[:, 2].squeeze(), U_inside[:,
                                                                                                   3].squeeze()

        # 使用自动求导方法得到U对X的导数
        du_dX_all = []
        for i in range(4):
            du_dX_i = torch.autograd.grad(
                inputs=self.X_inside,
                outputs=U_inside[:, i],
                grad_outputs=torch.ones_like(U_inside[:, i]),
                retain_graph=True,
                create_graph=True
            )[0]
            du_dX_all.append(du_dX_i.squeeze())

        x_t, y_t, z_t, u_t = du_dX_all[0], du_dX_all[1], du_dX_all[2], du_dX_all[3]

        loss_z = self.criterion(z_t, - self.rho * (self.beta_2 * x + z) * torch.exp(x))
        loss_u = self.criterion(u_t, - self.alpha - self.gamma * u + z_t)
        loss_x = self.criterion(x_t, (
                    torch.exp(x) * ((self.beta_1 - 1) * x * (1 + self.lam * u) + y - u) + self.kappa * (1 - torch.exp(x)) - u_t * (
                        1 + self.lam * y) / (1 + self.lam * u)) / (1 + self.lam * u + self.nu * torch.exp(x)))
        loss_y = self.criterion(y_t, self.kappa * (1 - torch.exp(x)) - self.nu * torch.exp(x) * x_t)

        residuals = self.compute_residuals()
        weights = torch.clamp(torch.abs(residuals).mean(dim=0), self.config["training"]["loss"]["residual_clamp_min"],
                              self.config["training"]["loss"]["residual_clamp_max"])
        loss_equation0 = (loss_x * weights[0] + loss_y * weights[1] + loss_z * weights[2] + loss_u * weights[
            3]) / weights.sum()  # 加权求和并归一化

        loss = loss_equation0 + self.config["training"]["loss"]["boundary_weight"] * loss_boundary

        # logging & best model
        self.loss_equation0_history.append(loss_equation0.item())
        self.loss_boundary_history.append(loss_boundary.item())

        if loss.item() < self.min_loss:
            self.min_loss = loss.item()
            self.best_model_state = self.model.state_dict()

        loss.backward()

        if self.iter % self.config["training"]["logging"]["print_every"] == 0:
            print(f"Iter {self.iter:6d} | loss {loss.item():.4e}")

        self.iter += 1
        return loss

    def train(self):
        print("Phase 1: Adam ...")
        for _ in range(self.config["training"]["optimizer"]["adam"]["steps"]):
            self.adam.step(self.loss_func)

        # if self.best_model_state is not None:
        #     torch.save(self.best_model_state, "temp_best.pth")

        print("Phase 2: L-BFGS ...")
        # self.model.load_state_dict(torch.load("temp_best.pth"))
        self.lbfgs.step(self.loss_func)

        self.save_best_model()

    def save_best_model(self):
        path1 = self.config["training"]["logging"]["best_model_path"]
        path = self.experiment_key+'/'+path1
        if self.best_model_state:
            torch.save(self.best_model_state, path)
            print(f"Best model saved: {path} (loss={self.min_loss:.2e})")

    def plot_loss(self):
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 14

        fig, ax = plt.subplots(figsize=(10, 6))

        iterations = range(1, len(self.loss_equation0_history) + 1)

        ax.plot(iterations, self.loss_equation0_history, label='ODE Equation Loss', color='blue')
        ax.plot(iterations, self.loss_boundary_history, label='Initial Condition Loss', color='red')

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Across Epochs')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')

        plt.tight_layout()

        # 保存（建议加上实验标识，避免覆盖）
        save_path = self.experiment_key+'/'+self.config["training"]["logging"].get("loss_plot_path", "loss_plot.pdf")



        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {save_path}")

        plt.show()

    def evaluate(self):
        model_path = self.experiment_key+'/'+self.config["training"]["logging"]["best_model_path"]
        self.model.load_state_dict(torch.load(model_path))


        tcfg = self.config["evaluate"]["time"]
        t = torch.arange(tcfg["start"], tcfg["end"], tcfg["step"], device=self.device)
        t = t.reshape(1, -1).T

        with torch.no_grad():
            U_pred = self.model(t).cpu().numpy()

        t_real = t * self.L_1 / self.v_0
        t_real = t_real.cpu().numpy().reshape(-1)

        x = U_pred[:, 0]
        y = U_pred[:, 1]
        z = U_pred[:, 2]
        u = U_pred[:, 3]
        plt.scatter(t_real, y, c='red')
        figsave_path = self.experiment_key + '/' + self.config["evaluate"]["logging"].get("y_plot_path",
                                                                                       "y_plot.pdf")
        plt.savefig(figsave_path, dpi=300, bbox_inches='tight')

        txtsave_path = self.experiment_key + '/' + self.config["evaluate"]["logging"].get("results_path")
        np.savetxt(txtsave_path,U_pred)