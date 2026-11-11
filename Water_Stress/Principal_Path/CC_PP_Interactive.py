import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from openTSNE import TSNE
import sys

# linear_utilities.py and principalpath.py live next to this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import linear_utilities as lu
import principalpath as pp

def load_data(base_path):
    conditions = ["Early", "Control", "Late"]
    plants = ["plant3", "plant0", "plant1"]
    indices = [0, 1, 2, 3]
  #  indices = [198, 199, 187]
    all_data, data_labels, plant_ids = [], [], []
    for condition in conditions:
        for plant in plants:
            folder_path = os.path.join(base_path, condition, plant)
            if not os.path.exists(folder_path): continue
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        data = np.genfromtxt(file_path, delimiter=',')
                        if data.shape[0] >= 3 and data.shape[1] >= 5:
                            flattened = data[indices, 3:5].flatten()
                            all_data.append(flattened)
                            data_labels.append(condition)
                            plant_ids.append(plant)
                    except:
                        continue
    return np.array(all_data), np.array(data_labels), np.array(plant_ids)

def project_to_path(point, path):
    min_dist, proj_point, proj_idx = float('inf'), None, 0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ab = b - a
        t = np.dot(point - a, ab) / np.dot(ab, ab)
        t_clamped = np.clip(t, 0, 1)
        proj = a + t_clamped * ab
        dist = np.linalg.norm(point - proj)
        if dist < min_dist:
            min_dist, proj_point, proj_idx, proj_t = dist, proj, i, t_clamped
    norm_val = (proj_idx + proj_t) / (len(path) - 1)
    return proj_point, norm_val

def draw_colored_gauge(ax):
    colors = ['red'] * 10 + ['yellow'] * 10 + ['green'] * 20
    start = 0
    for c in colors:
        end = start + 180 / len(colors)
        ax.add_patch(Wedge(center=(0, 0), r=1.0, theta1=start, theta2=end, width=0.5, facecolor=c))
        start = end
    ax.add_patch(Circle((0, 0), 0.7, color='white'))
    ax.add_patch(Circle((0, 0), 0.05, color='black'))

# Load and normalize
data_path = "Dataset/Water-Stress/"
X_raw, labels, plant_ids = load_data(data_path)
plant_list = ["plant3", "plant0", "plant1"]
condition_colors = {'Early': 'blue', 'Control': 'red', 'Late': 'green'}

for idx, test_plant in enumerate(plant_list):
    if test_plant in ['plant0', 'plant1']: continue

    train_plants = [p for p in plant_list if p != test_plant]
    print(f"\nFold {idx + 1}: Train = {train_plants}, Test = {test_plant}")

    train_mask = np.isin(plant_ids, train_plants)
    test_mask = (plant_ids == test_plant)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_raw[train_mask])
    X_test = scaler.transform(X_raw[test_mask])

    X_combined = np.zeros_like(X_raw)
    X_combined[train_mask] = X_train
    X_combined[test_mask] = X_test

    # Principal Path
    boundary_ids = lu.getMouseSamples2D(X_train, 2)
    NC = 21 ###make sure that this parameter does not exceed 24-25 for both experiments, otherwise the path will not
    ###be constructed smoothly
    waypoint_ids = lu.initMedoids(X_train, NC, 'kpp', boundary_ids)
    waypoint_ids = np.hstack([boundary_ids[0], waypoint_ids, boundary_ids[1]])
    W_init = X_train[waypoint_ids, :]
    s_span = np.hstack([np.logspace(5, -5), 0])
    models = np.zeros((s_span.size, NC + 2, X_combined.shape[1]))
    for i, s in enumerate(s_span):
        W, _ = pp.rkm(X_train, W_init, s)
        W_init = W
        models[i] = W
    best_s_id = np.argmax(pp.rkm_MS_evidence(models, s_span, X_train))
    print(best_s_id)
    W_path = models[11, :, :] ###keep it between 10-11-12
    # t-SNE fit on second training plant, then transform [W_path + test]
    train_mask_2 = (plant_ids == train_plants[1])
    tsne = TSNE(perplexity=30, learning_rate='auto',
                metric='cosine',
                n_iter= 1000, ##do not select smaller values, as it requires 1000 and above to converge
                random_state=42, 
                initialization='pca',
                verbose= True,
                n_jobs=-1)
    embedding_train_2 = tsne.fit(X_combined[train_mask_2])

    X_stack = np.vstack([W_path, X_combined[test_mask]])
    embedding_stack = embedding_train_2.transform(X_stack)
    embedding_W_path = embedding_stack[:W_path.shape[0]]
    embedding_test = embedding_stack[W_path.shape[0]:]

    # Plot & Interactivity
    fig, (ax_tsne, ax_gauge) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
    for condition in ['Early', 'Control', 'Late']:
        c_mask = labels[test_mask] == condition
        if np.any(c_mask):
            ax_tsne.scatter(embedding_test[c_mask, 0], embedding_test[c_mask, 1], color=condition_colors[condition], label=condition, alpha=0.6)
    ax_tsne.plot(embedding_W_path[:, 0], embedding_W_path[:, 1], 'k*-', label="Principal Path")
    ax_tsne.set_title(f"{test_plant} (Test) + Principal Path")
    ax_tsne.set_xlabel("t-SNE 1")
    ax_tsne.set_ylabel("t-SNE 2")
    ax_tsne.legend()

    ax_gauge.set_xlim(-1.2, 1.2)
    ax_gauge.set_ylim(-1.2, 1.2)
    ax_gauge.axis('off')
    ax_gauge.set_title("Stress Level Gauge", fontsize=12)
    draw_colored_gauge(ax_gauge)
    needle, = ax_gauge.plot([], [], 'r-', lw=3)

    state = {"clicked": None, "proj": None, "line": None}

    def update_gauge(stress):
        angle = 180 * stress
        x = np.cos(np.radians(180 - angle)) * 0.85
        y = np.sin(np.radians(180 - angle)) * 0.85
        needle.set_data([0, x], [0, y])
        ax_gauge.set_title(f"Stress Level: {stress:.2f}", fontsize=12)

    def onclick(event):
        if event.inaxes == ax_tsne:
            clicked = np.array([event.xdata, event.ydata])
            proj, stress = project_to_path(clicked, embedding_W_path)
            for key in ['clicked', 'proj', 'line']:
                if state[key]: state[key].remove()
            state["clicked"] = ax_tsne.plot(clicked[0], clicked[1], 'k*', markersize=12)[0]
            state["proj"] = ax_tsne.plot(proj[0], proj[1], 'ko', markersize=8)[0]
            state["line"] = ax_tsne.plot([clicked[0], proj[0]], [clicked[1], proj[1]], 'k--')[0]
            update_gauge(stress)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()
