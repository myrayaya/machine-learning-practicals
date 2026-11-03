import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

# dataset prep
data = pd.DataFrame({
    "Point": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "x1":    [4, 3, 5, 2, 8, 7],
    "x2":    [3, 3, 5, 4, 8, 2],
    "Class": ["B", "A", "A", "A", "B", "B"],
})
print(data.to_string(index=False))

Q = np.array([4, 4])                     
print("\nNew point Q =", tuple(Q.tolist()))
print()

# distance from Q to each point
points = data[["x1", "x2"]].values
data["Distance"] = np.sqrt(((points - Q) ** 2).sum(axis=1))

# sort so the nearest point comes first
ranked = data.sort_values("Distance").reset_index(drop=True)
ranked.index = ranked.index + 1          
ranked.index.name = "Rank"
print(ranked.round(3).to_string())
print()

# classification for different k
results = []
for k in [1, 3, 5]:
    nearest = ranked.head(k)             
    votes = Counter(nearest["Class"])    
    predicted = votes.most_common(1)[0][0]

    print(f"K = {k}")
    print("  Nearest points :", list(nearest["Point"]))
    print("  Their classes  :", list(nearest["Class"]))
    print("  Votes          :", dict(votes))
    print("  Prediction     :", predicted)
    print()

    results.append({"K": k, "Votes": dict(votes), "Prediction": predicted})

# double check with sklearn's KNN
for k in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(points, data["Class"])
    print(f"K = {k} -> sklearn predicts:", knn.predict([Q])[0])
print()

# results and summary
print(pd.DataFrame(results).to_string(index=False))
print()

# plotting
plt.figure(figsize=(7, 7))

colors = {"A": "tab:blue", "B": "tab:red"}
for cls in ["A", "B"]:
    subset = data[data["Class"] == cls]
    plt.scatter(subset["x1"], subset["x2"], color=colors[cls], s=80, label="Class " + cls)

for _, row in data.iterrows():           # label each point
    plt.text(row["x1"] + 0.1, row["x2"] + 0.1, row["Point"])

plt.scatter(Q[0], Q[1], color="black", marker="*", s=250, label="Q (new point)")

for k, style in zip([1, 3, 5], [":", "--", "-."]):
    radius = ranked["Distance"].iloc[k - 1]
    circle = plt.Circle(Q, radius, fill=False, linestyle=style, color="grey")
    plt.gca().add_patch(circle)
    plt.text(Q[0], Q[1] + radius + 0.08, f"K={k}", fontsize=9, color="dimgrey", ha="center")

plt.gca().set_aspect("equal")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("KNN: effect of K on classifying Q = (4, 4)")
plt.legend(loc = "upper left")
plt.grid(True)
plt.show()
