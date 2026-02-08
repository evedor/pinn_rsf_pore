
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

        self._set_seed()
        self.device = torch.device(self.config["training"]["device"])
        self._override_experiment()

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
        self.v_0 = cfg["v_0"]
        self.k = cfg["k"]
        self.beta_1 = cfg["beta_1"]
        self.L_1 = cfg["L_1"]
        self.rho = cfg["rho"]
        self.a = cfg["a"]
        self.mu_0 = cfg["mu_0"]
        self.G = cfg["G"]
        self.rho_v = cfg["rho_v"]
        self.p_0 = cfg["p_0"]
        self.beta_a = cfg["beta_a"]
        self.beta_m = cfg["beta_m"]
        self.phi_0 = cfg["phi_0"]
        self.epsilon = cfg["epsilon"]
        self.c_0 = cfg["c_0"]

        # 派生参数
        self.beta = self.phi_0 * (self.beta_a + self.beta_m)
        self.L_2 = self.rho * self.L_1
        self.lam = self.a / self.mu_0

    def _override_experiment(self):
        self.experiment = self.config.get("experiments", {})
        exp = self.experiment[self.experiment_key]

        init_val_mpa = 30.0
        self.sigma_n_param = torch.nn.Parameter(torch.tensor(init_val_mpa, dtype=torch.float32).to(self.device))
        self.t0_values = exp["t0_values"]

    def _update_dynamic_parameters(self):
        self.sigma_n0 = torch.abs(self.sigma_n_param) * 1e6

        # 根据 train.py 中的公式动态计算
        self.tau_0 = self.mu_0 * self.sigma_n0
        self.kappa = (self.k * self.L_1) / (self.a * self.sigma_n0)
        self.nu = (self.G / (2 * np.sqrt(self.G / self.rho_v))) * self.v_0 / (self.a * self.sigma_n0)
        self.beta_2 = -self.epsilon / (self.lam * self.beta * self.sigma_n0)
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

    def _init_data(self):
        tcfg = self.config["training"]["time"]
        self.dt = tcfg["step"]
        t = torch.arange(tcfg["start"], tcfg["end"], self.dt, device=self.device)
        self.X_inside = t.reshape(1, -1).T

        self.data_path = "./experiment_data/{}.txt".format(self.experiment_key)
        raw_data = np.loadtxt(self.data_path)

        target_val = raw_data[:, 1] # * self.a * 17.003 + self.mu_0 * 17.003
        self.X_inside_data = torch.tensor(target_val, dtype=torch.float32).to(self.device)

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

        # 优化器参数列表中使用新的 sigma_n_param
        # self.adam = torch.optim.Adam(
        #     list(self.model.parameters()) + [self.sigma_n_param],
        #     lr=ocfg["adam"]["lr"]
        # )

        # 显式指定 line_search_fn='strong_wolfe' 有助于处理刚性问题
        self.lbfgs = torch.optim.LBFGS(
            list(self.model.parameters()) + [self.sigma_n_param],
            lr=0.1,  # 稍微降低 LBFGS 学习率，防止震荡
            max_iter=ocfg["lbfgs"]["max_iter"],
            max_eval=ocfg["lbfgs"]["max_eval"],
            history_size=ocfg["lbfgs"]["history_size"],
            tolerance_grad=ocfg["lbfgs"]["tolerance_grad"],
            tolerance_change=ocfg["lbfgs"]["tolerance_change"],
            line_search_fn="strong_wolfe"
        )

    def _init_logging(self):
        self.iter = 1
        self.min_loss = float('inf')
        self.best_model_state = None
        self.loss_equation0_history = []
        self.loss_boundary_history = []
        self.loss_data_history = []
        self.sigma_history = []

    # ── 核心残差计算 ────────────────────────────────────────
    def compute_residuals(self):
        self._update_dynamic_parameters()

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
                torch.exp(x) * ((self.beta_1 - 1) * x * (1 + self.lam * u) + y - u) + self.kappa * (
                    1 - torch.exp(x)) - u_t * (
                        1 + self.lam * y) / (1 + self.lam * u)) / (1 + self.lam * u + self.nu * torch.exp(x))
        residual_y = y_t - (self.kappa * (1 - torch.exp(x)) - self.nu * torch.exp(x) * x_t)

        return torch.stack([residual_x, residual_y, residual_z, residual_u], dim=1)

    def loss_func(self):
        # self.adam.zero_grad()
        self.lbfgs.zero_grad()

        self._update_dynamic_parameters()

        # 1. 边界 Loss
        U_pred_boundary = self.model(self.X_boundary)
        loss_boundary = self.criterion(U_pred_boundary, self.U_boundary)

        # 2. ODE Loss (内点)
        U_inside = self.model(self.X_inside)
        x, y, z, u = U_inside[:, 0].squeeze(), U_inside[:, 1].squeeze(), U_inside[:, 2].squeeze(), U_inside[:,
                                                                                                   3].squeeze()

        # 自动求导
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

        # 物理残差 MSE
        loss_z = self.criterion(z_t, - self.rho * (self.beta_2 * x + z) * torch.exp(x))
        loss_u = self.criterion(u_t, - self.alpha - self.gamma * u + z_t)
        loss_x_term = (torch.exp(x) * ((self.beta_1 - 1) * x * (1 + self.lam * u) + y - u) + self.kappa * (
                    1 - torch.exp(x)) - u_t * (
                               1 + self.lam * y) / (1 + self.lam * u)) / (1 + self.lam * u + self.nu * torch.exp(x))
        loss_x = self.criterion(x_t, loss_x_term)
        loss_y = self.criterion(y_t, self.kappa * (1 - torch.exp(x)) - self.nu * torch.exp(x) * x_t)

        loss_equation = loss_x + loss_y + loss_z + loss_u

        # 3. Data Loss
        # 预测的物理量 (注意单位转换)
        y_pred_phys = y * self.a * self.sigma_n0 / 1e6 + self.mu_0 * self.sigma_n0 / 1e6
        loss_data = self.criterion(y_pred_phys, self.X_inside_data)

        # 赋予 Data Loss 较大的固定权重 (例如 100.0)，强迫参数向数据靠拢
        w_boundary = self.config["training"]["loss"]["boundary_weight"]
        w_data = 100.0  # 手动设定一个较大的权重

        loss = loss_equation + w_boundary * loss_boundary + w_data * loss_data

        # Logging & Best Model
        # 记录时稍微处理一下，方便绘图
        self.loss_equation0_history.append(loss_equation.item())
        self.loss_boundary_history.append(loss_boundary.item())
        self.loss_data_history.append(loss_data.item() * w_data)

        # 记录当前的 MPa 值
        current_sigma_val = torch.abs(self.sigma_n_param).item()
        self.sigma_history.append(current_sigma_val)

        if loss.item() < self.min_loss:
            self.min_loss = loss.item()
            self.best_model_state = self.model.state_dict()
            self.best_sigma_n = current_sigma_val  # 直接存 MPa 值

        loss.backward()

        if self.iter % self.config["training"]["logging"]["print_every"] == 0:
            print(f"Iter {self.iter:6d} | Loss: {loss.item():.4e} | Sigma_n: {current_sigma_val:.4f} MPa")

        self.iter += 1
        return loss

    def train(self):
        # print("Phase 1: Adam ...")
        # for _ in range(self.config["training"]["optimizer"]["adam"]["steps"]):
        #     self.adam.step(self.loss_func)

        print("Phase 1 L-BFGS ...")
        self.lbfgs.step(self.loss_func)

        self.save_best_model()
        print(f"Final Inverted Sigma_n: {self.best_sigma_n:.4f} MPa")

    def save_best_model(self):
        path1 = self.config["training"]["logging"]["best_model_path"]
        path = self.experiment_key + '/' + path1
        if self.best_model_state:
            torch.save(self.best_model_state, path)
            print(f"Best model saved: {path} (loss={self.min_loss:.2e})")

    def plot_loss(self):
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 14

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        iterations = range(1, len(self.loss_equation0_history) + 1)

        # Plot 1: Losses
        ax1.plot(iterations, self.loss_equation0_history, label='ODE Loss', color='blue')
        ax1.plot(iterations, self.loss_boundary_history, label='IC Loss', color='red')
        ax1.plot(iterations, self.loss_data_history, label='Data Loss (x100)', color='orange')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True)
        ax1.set_yscale('log')

        self.experiment = self.config.get("experiments", {})
        exp = self.experiment[self.experiment_key]
        self.sigma_n0 = exp["sigma_n0"]/1e6

        # Plot 2: Parameter Inversion
        sigma_values_mpa = np.array(self.sigma_history)
        ax2.plot(iterations, sigma_values_mpa, color='black', label=r'Inverted $\sigma_{n0}$')
        ax2.axhline(y=self.sigma_n0, color='red', linestyle='--', label='True Value ({} MPa)'.format(self.sigma_n0))

        # 动态调整 Y 轴范围以适应数据
        ax2.set_ylim(min(15, min(sigma_values_mpa)), max(32, max(sigma_values_mpa)))

        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Sigma_n (MPa)')
        ax2.set_title('Parameter Inversion')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        save_path = self.save_dir / "inversion_plot.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Inversion plot saved to: {save_path}")
        plt.show()