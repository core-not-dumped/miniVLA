import matplotlib.pyplot as plt
import numpy as np

values = [1, 2, 2, 5, 2, 3, 1, 5, 7, 3, 1, 2, 5, 7 , 4]

values = np.array(values)

# x축: 1M, 2M, 3M, ...
x = np.arange(1, len(values) + 1)

plt.figure(figsize=(8, 4))
plt.plot(x, values)

# x축을 5M 단위로 표시
xticks = np.arange(5, len(values) + 1, 5)
plt.xticks(xticks, [f"{i}M" for i in xticks])

plt.xlabel("Steps")
plt.ylabel("Value")
plt.title("Training Curve")

plt.grid(True)
plt.tight_layout()

plt.savefig("curve.png", dpi=150)
plt.close()