import numpy as np

class GazeKalmanFilter:
    def __init__(self, state_dim=4, measure_dim=2, 
                 dt=1/30, process_noise=0.01, measurement_noise=0.1,
                 coeff_Q=0.5, coeff_P=10):
        """
        初始化卡尔曼滤波器
        参数：
        dt: 时间间隔（秒），默认1/30秒（对应30fps）
        process_noise: 过程噪声强度（影响状态估计的灵敏度）
        measurement_noise: 测量噪声强度（影响对观测值的信任程度）
        """
        # 状态维度：4（pitch, pitch速度, yaw, yaw速度）
        self.state_dim = state_dim
        
        # 观测维度：2（pitch, yaw）
        self.measure_dim = measure_dim

        # 状态转移矩阵（动力学模型）
        self.F = np.array([
            [1, dt, 0, 0],    # pitch更新
            [0, 1, 0, 0],     # pitch速度保持
            [0, 0, 1, dt],    # yaw更新
            [0, 0, 0, 1]      # yaw速度保持
        ])

        # 观测矩阵（只观测角度，不观测角速度）
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        # 过程噪声协方差矩阵
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[1,1] *= coeff_Q  # 角度速度噪声系数
        self.Q[3,3] *= coeff_Q

        # 测量噪声协方差矩阵
        self.R = np.eye(self.measure_dim) * measurement_noise

        # 状态协方差矩阵（初始不确定性）
        self.P = np.eye(self.state_dim) * coeff_P

        # 初始状态（首次更新时初始化）
        self.x = None

    def predict(self):
        """仅执行预测步骤"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        支持缺失值处理（measurement为None时仅预测）
        """
        
        # 首次观测初始化
        if self.x is None:
            self.x = np.zeros(self.state_dim)
            if measurement is not None:
                self.x[0] = measurement[0]
                self.x[2] = measurement[1]
            return

        if measurement is None:
            self.predict()
            return


        # 常规更新流程
        self.predict()  # 先预测
        S = self.H @ self.P @ self.H.T + self.R
        
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        ## accelerate matrix inversion
        # L = np.linalg.cholesky(S)  # S必须正定
        # K = P @ H.T @ np.linalg.inv(L.T) @ np.linalg.inv(L)


        self.x = self.x + K @ (measurement - self.H @ self.x)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
    
    def get_estimate(self):
        """
        获取当前平滑估计值
        返回：
        (pitch, yaw) 的平滑估计值（弧度）
        """
        return [self.x[0], self.x[2]]