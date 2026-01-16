import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from network import Network

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

sigma_n0 = 13.600*1e6

# 简化参数
tau_0 = mu_0*sigma_n0  # (10)
kappa = (k*L_1) / (a*sigma_n0) #26(a)
nu = eta*v_0 / (a*sigma_n0) #26(b)
rho = rho
beta_2 = -epsilon/(lam*beta*sigma_n0) #26(d)
alpha = (c_0*p_0*L_1) / (v_0*lam*sigma_n0)  #26(e)
gamma = gamma

# 选择GPU或CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Network(
    input_size=1,
    hidden_size=200,
    output_size=4,
    depth=4,
    act=torch.nn.Tanh
).to(device)

# # 从文件加载已经训练完成的模型
# model_loaded = torch.load('model.pth', map_location=device)
# model_loaded.eval()  # 设置模型为evaluation状态

# 从文件加载状态字典
state_dict = torch.load('model.pth', map_location=device)

# 将状态字典加载到模型
model.load_state_dict(state_dict)

# 设置模型为evaluation状态
model.eval()

# 生成时空网格
k = 0.1
t = torch.arange(0, 6, k)
t1 = torch.arange(0, 6, 0.1)
X = t.reshape(1, -1).T
X = X.to(device)

# 计算该时空网格对应的预测值
with torch.no_grad():
    U_pred = model(X).cpu().numpy()

# 绘制计算结果
plt.figure(figsize=(5, 3), dpi=300)
xnumpy = t.numpy()
# t = torch.arange(0, 67, 1.0)
t_real = t*L_1/v_0
t_real1 = t1*L_1/v_0
x1 = U_pred[:,0]
y1 = U_pred[:,1]
z1 = U_pred[:,2]
u1 = U_pred[:,3]
data = np.loadtxt('data1.txt')
x = data[:,0]
y = data[:,1]
z = data[:,2]
u = data[:,3]

# ShearStressobs1 = (tau_0 + a * sigma_n0 * y1) / 1e6

plt.scatter(t_real1,y)
plt.scatter(t_real,y1,c='red')
# plt.scatter(t_real1,u)
# plt.scatter(t_real,u1,c='red')
# plt.plot(xnumpy[220001:], U_pred[220001:, 1])
plt.show()
