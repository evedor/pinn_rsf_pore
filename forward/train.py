import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from network import Network

seed = 41  # 你可以选择任意整数作为种子
torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
np.random.seed(seed)     # 设置 NumPy 的随机种子

v_0 = 10e-6     # Loading velocity [m/s]
k = 14.8*1e9    # Spring stiffness [Pa/m]
beta_1 = 1.2    # b₁/a₀ [non-dim]
L_1 = 3*1e-6    # Critical distance for θ₁ [m]
rho = 0.1
L_2 = rho*L_1

a = 0.01        # Friction direct effect [non-dim]
mu_0 = 0.64       # Static friction coefficient [non-dim]
lam = a/mu_0

G = 31e9         # Rigidity modulus of quartz [Pa]
rho_v = 2.65e3   # Density of quartz [kg/m^3]
c_s = np.sqrt(G/rho_v)
eta_0 = G/(2*c_s)
eta = 1*eta_0

p_0 = 1.01325e5  # Reference surrounding pressure (atmospheric pressure) [Pa]

beta_a = 1e-9    #[0.5-4]*1e-9 (David et al., 1994; see Segall and Rice, 1995)
beta_m = 1e-11       # Compressibility of Quartz (Pimienta et al., 2017, JGR, fig. 12)
phi_0 = 0.075       # Reference porosity
beta = phi_0*(beta_a+beta_m)
epsilon = -0.017*1e-3  # Dilatancy/Compressibility coefficient
#
c_0 = 10          # Diffusivity [1/s]
gamma = c_0*L_1/v_0

sigma_n0 = 17.003*1e6

# 简化参数
tau_0 = mu_0*sigma_n0  # (10)
kappa = (k*L_1) / (a*sigma_n0) #26(a)
nu = eta*v_0 / (a*sigma_n0) #26(b)
rho = rho
beta_2 = -epsilon/(lam*beta*sigma_n0) #26(d)
alpha = (c_0*p_0*L_1) / (v_0*lam*sigma_n0)  #26(e)
gamma = gamma

# 定义一个类，用于实现PINN(Physics-informed Neural Networks)
class PINN:
    # 构造函数
    def __init__(self):
        # 选择使用GPU还是CPU
        device = torch.device("mps")
            # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # 定义神经网络
        self.model = Network(
            input_size=1,  # 输入层神经元数
            hidden_size=200,  # 隐藏层神经元数
            output_size=4,  # 输出层神经元数
            depth=4,  # 隐藏层数xc
            act=torch.nn.Tanh  # 输入层和隐藏层的激活函数
        ).to(device)  # 将这个神经网络存储在GPU上（若GPU可用）

        self.k = 0.1  # 设置时间步长
        t = torch.arange(0, 6, self.k)  # 在[0,1]区间上均匀取值，记为t

        # 将x和t组合，形成时间空间网格，记录在张量X_inside中
        self.X_inside = t.reshape(1, -1).T

        # 边界处的时空坐标
        ic = t[0].reshape(1, -1).T # t=0边界
        self.X_boundary = ic  # 将所有边界处的时空坐标点整合为一个张量

        # 边界处的u值
        u_ic = torch.tensor([[0.0014851490120946023,-0.3807304456678973,0.0001626260574953936,-0.3814333184281347]])  # t=0边界处采用第一类边界条件u=-sin(pi*x)
        self.U_boundary = torch.cat([u_ic])  # 将所有边界处的u值整合为一个张量

        # 将数据拷贝到GPU
        self.X_inside = self.X_inside.to(device)
        self.X_boundary = self.X_boundary.to(device)
        self.U_boundary = self.U_boundary.to(device)
        self.X_inside.requires_grad = True  # 设置：需要计算对X的梯度

        # 设置准则函数为MSE，方便后续计算MSE
        self.criterion = torch.nn.MSELoss()

        # 定义迭代序号，记录调用了多少次loss
        self.iter = 1

        # 初始化最小损失和最佳模型状态
        self.min_loss = float('inf')
        self.best_model_state = None

        # 初始化损失记录列表
        self.loss_equation0_history = []
        self.loss_boundary_history = []

        # 设置lbfgs优化器
        # self.lbfgs = torch.optim.LBFGS(
        #     self.model.parameters(),
        #     lr=0.001,
        #     max_iter=100000,
        #     max_eval=100000,
        #     history_size=50,
        #     tolerance_grad=1e-9,
        #     tolerance_change=1.0 * np.finfo(float).eps,
        #     line_search_fn="strong_wolfe",
        # )

        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=0.001,
            max_iter=400000,
            max_eval=400000,
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-12, #1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

        # 设置adam优化器
        self.adam = torch.optim.Adam(self.model.parameters(),lr=0.0003)#3e-3

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

        residual_z = z_t - (-rho * (beta_2 * x + z) * torch.exp(x))
        residual_u = u_t - (-alpha - gamma * u + z_t)
        residual_x = x_t - (torch.exp(x) * ((beta_1 - 1) * x * (1 + lam * u) + y - u) + kappa * (1 - torch.exp(x)) - u_t * (
                        1 + lam * y) / (1 + lam * u)) / (1 + lam * u + nu * torch.exp(x))
        residual_y = y_t - (kappa * (1 - torch.exp(x)) - nu * torch.exp(x) * x_t)


        return torch.stack([residual_x, residual_y, residual_z, residual_u], dim=1)

    # 损失函数
    def loss_func(self):
        # 将导数清零
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        # 第一部分loss: 边界条件不吻合产生的loss
        U_pred_boundary = self.model(self.X_boundary)  # 使用当前模型计算u在边界处的预测值
        loss_boundary = self.criterion(
            U_pred_boundary, self.U_boundary)  # 计算边界处的MSE

        # 第二部分loss:内点非物理产生的loss
        U_inside = self.model(self.X_inside)  # 使用当前模型计算内点处的预测值

        x, y, z, u = U_inside[:,0].squeeze(), U_inside[:,1].squeeze(), U_inside[:,2].squeeze(), U_inside[:,3].squeeze()

        # 使用自动求导方法得到U对X的导数
        du_dX_all = []
        for i in range(4):
            du_dX_i = torch.autograd.grad(
                inputs=self.X_inside,
                outputs=U_inside[:,i],
                grad_outputs=torch.ones_like(U_inside[:,i]),
                retain_graph=True,
                create_graph=True
            )[0]
            du_dX_all.append(du_dX_i.squeeze())

        x_t, y_t, z_t, u_t = du_dX_all[0], du_dX_all[1], du_dX_all[2], du_dX_all[3]

        loss_z = self.criterion(z_t, - rho * (beta_2 * x + z) * torch.exp(x))
        loss_u = self.criterion(u_t, - alpha - gamma * u + z_t)
        loss_x = self.criterion(x_t,(torch.exp(x)*((beta_1-1)*x*(1+lam*u)+y-u)+kappa*(1-torch.exp(x)) - u_t*(1+lam*y)/(1+lam*u))/(1+lam*u+nu*torch.exp(x)))
        loss_y = self.criterion(y_t, kappa*(1 - torch.exp(x)) - nu*torch.exp(x)*x_t)

        # loss_x_0 = x_t - (torch.exp(x)*((beta_1-1)*x*(1+lam*u)+y-u)+kappa*(1-torch.exp(x)) - u_t*(1+lam*y)/(1+lam*u))/(1+lam*u+kappa*torch.exp(x))
        # loss_y_0 = y_t - (kappa*(1 - torch.exp(x)) - nu*torch.exp(x)*x_t)
        # loss_z_0 = z_t - (- rho*(beta_2*x + z)*torch.exp(x))
        # loss_u_0 = u_t - (- alpha - gamma*u + z_t)
        #
        # loss_all = (loss_x_0**2+loss_y_0**2+loss_z_0**2+loss_u_0**2)

        # loss_equation0 = loss_x+loss_y+loss_z+loss_u  # 计算物理方程的MSE
        # 自适应权重：根据残差调整权重
        residuals = self.compute_residuals()
        eps = 1e-6  # 防止除零
        mean_residuals = torch.abs(residuals).mean(dim=0)
        weight = 1.0 / (mean_residuals + eps)  # 倒数加权
        weights = torch.clamp(weight, min=0.1, max=10.0)  # 限制权重范围
        # weights = torch.clamp(torch.abs(residuals).mean(dim=0), min=0.1, max=10.0)  # 计算每个方程的平均残差，并限制范围
        loss_equation0 = (loss_x * weights[0] + loss_y * weights[1] + loss_z * weights[2] + loss_u * weights[
            3]) / weights.sum()  # 加权求和并归一化


        # loss_equation = loss_all[0]*1/2+loss_all[-1]*1/2 +torch.sum(loss_all[1:-1]*1)

        # 最终的loss由两项组成
        loss = loss_equation0 + 0.5*loss_boundary

        self.loss_equation0_history.append(loss_equation0.item())
        self.loss_boundary_history.append(loss_boundary.item())

        if loss.item() < self.min_loss:
            self.min_loss = loss.item()
            self.best_model_state = self.model.state_dict()

        # # loss反向传播，用于给优化器提供梯度信息
        loss.backward()

        # 每计算100次loss在控制台上输出消息
        if self.iter % 100 == 0:
            print(self.iter, loss.item())
        self.iter = self.iter + 1
        return loss

    # 训练
    def train(self):
        self.model.train()  # 设置模型为训练模式

        # 首先运行5000步Adam优化器
        print("采用Adam优化器")
        for i in range(6000):
            self.adam.step(self.loss_func)
        if self.best_model_state is not None:
            torch.save(self.best_model_state, 'model.pth')

        # 然后运行lbfgs优化器
        self.model.load_state_dict(torch.load('model.pth'))
        print("采用L-BFGS优化器")
        self.lbfgs.step(self.loss_func)

        # 保存最佳模型
    def save_best_model(self, path='model.pth'):
        if self.best_model_state is not None:
            torch.save(self.best_model_state, path)
            print(f"Best model saved with loss {self.min_loss} at {path}")
        else:
            print("No best model state found to save.")

    def plot_loss(self):
        plt.rcParams['font.family'] = 'Arial'  # 设置字体为 Arial（你也可以选择其他字体）
        plt.rcParams['font.size'] = 14

        plt.figure(figsize=(10, 6))
        iterations = range(1, len(self.loss_equation0_history) + 1)
        plt.plot(iterations, self.loss_equation0_history, label='ODE Equation Loss', color='blue')
        plt.plot(iterations, self.loss_boundary_history, label='Initial Condition Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Across Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.yscale('log')  # 使用对数刻度以更好地显示损失变化
        plt.savefig('loss_plot.pdf', dpi=300)
        plt.show()

# 实例化PINN
pinn = PINN()

# 开始训练
pinn.train()

# 将模型保存到文件
pinn.save_best_model()
pinn.plot_loss()