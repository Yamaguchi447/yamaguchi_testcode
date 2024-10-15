import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.importance import PedAnovaImportanceEvaluator
import pandas as pd
from optunahub import load_module
from optuna.distributions import FloatDistribution
from sklearn.decomposition import PCA
from src.sampler.uniform_design import UniformDesignSampler

# N次元空間の回転行列を生成する関数
def generate_full_rotation_matrix(N, angle):
    R = np.eye(N)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    for i in range(N):
        for j in range(i + 1, N):
            R_ij = np.eye(N)
            R_ij[i, i] = cos_theta
            R_ij[j, j] = cos_theta
            R_ij[i, j] = -sin_theta
            R_ij[j, i] = sin_theta
            R = np.dot(R, R_ij)
    return R

def rotated_ellipsoid(x, R):
    w = np.logspace(0, EllipsoidCondition, base=10, num=x.shape[0], endpoint=True)
    y = np.dot(R, x)
    return np.sqrt(np.sum(w * y**2))

EllipsoidCondition = 1
N = 2
R_initial = np.eye(N)
dx, dy = 0.05, 0.05
y, x = np.mgrid[slice(-1, 1 + dy, dy), slice(-1, 1 + dx, dx)]
importance_results = []

# 目的関数値に基づくPCAの第一主成分軸を描く関数
def objective_value_pca(sampled_points, objective_values, ax, color="blue"):
    # サンプル点と目的関数値を特徴量としてPCAを適用
    features = np.hstack([sampled_points, objective_values.reshape(-1, 1)])  # 座標と目的関数値を結合
    pca = PCA(n_components=1)  # 第一主成分のみ計算
    pca.fit(features)
    
    # 第一主成分ベクトル（固有ベクトル）を取得
    first_pc = pca.components_[0]
    
    # 中心点をプロット
    mean_point = sampled_points.mean(axis=0)
    
    # PCAの結果に基づいて、目的関数値の分散方向を描画
    scale = 1  # 矢印のスケール
    ax.arrow(mean_point[0], mean_point[1], first_pc[0] * scale, first_pc[1] * scale,
             head_width=0.05, head_length=0.1, fc=color, ec=color)
    return first_pc

# 0度から180度まで回転しながら探索空間を描画
for theta in np.arange(0, 180, 10):  # 0度から180度まで回転
    z = np.zeros(x.shape)
    R_rotation = generate_full_rotation_matrix(N, np.deg2rad(theta))
    R = np.dot(R_rotation, R_initial)

    def objective(trial):
        param1 = trial.suggest_uniform("param1", -1, 1)
        param2 = trial.suggest_uniform("param2", -1, 1)
        # param1 = 2 * param1 - 1
        # param2 = 2 * param2 - 1
        point = np.array([param1, param2])
        point = np.pad(point, (0, N - 2))
        return rotated_ellipsoid(point, R)

    search_space = {
        "param1": FloatDistribution(-1, 1),
        "param2": FloatDistribution(-1, 1),
    }

    discretization_level = 50
    sampler = UniformDesignSampler(search_space, discretization_level)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=50, n_jobs=12)

    evaluator = PedAnovaImportanceEvaluator()
    importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
    importance["angle"] = theta
    importance_results.append(importance)

    # 探索空間の図を作成
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = np.zeros(N)
            point[:2] = [x[i, j], y[i, j]]
            z[i, j] = rotated_ellipsoid(point, R)

    fig, ax = plt.subplots()
    CS = ax.contour(x, y, z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.grid()
    
    # 回転後のサンプル点を取得してプロット
    sampled_points = np.array([[trial.params["param1"], trial.params["param2"]] for trial in study.trials])
    objective_values = np.array([trial.value for trial in study.trials])
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], color="red", label="Sampled Points", zorder=5)

    # 目的関数値とサンプル点に基づくPCAによる主成分軸を描画
    objective_value_pca(sampled_points, objective_values, ax)

    ax.set_title(f"Rotation Angle: {theta}°")
    ax.legend()
    ax.axis("equal")
    plt.show()

# 結果をCSVとして保存
importance_df = pd.DataFrame(importance_results)
importance_df.to_csv("importance_results.csv", index=False)

# グラフとしてプロット
importance_df.plot(x="angle", y=["param1", "param2"], kind="line", marker="o")
plt.xlabel("Rotation Angle")
plt.ylabel("Importance")
plt.title("Parameter Importance")
plt.grid(True)
plt.savefig("importance_results_plot3.pdf")
plt.show()
