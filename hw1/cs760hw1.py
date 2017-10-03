import numpy as np
import matplotlib.pyplot as plt

# 7.1.1
x1_1 = np.arange(-1,0,0.01)
x1_2 = np.arange(0,1,0.01)
y1_1 = x1_1 + 1
y1_2 = -x1_1 - 1
y1_3 = x1_2 - 1
y1_4 = -x1_2 + 1
plt.plot(x1_1, y1_1, "b", x1_1, y1_2, "b", x1_2, y1_3, "b", x1_2, y1_4, "b")
plt.show()

# 7.1.2
x2_1 = np.arange(-1,0,0.01)
x2_2 = np.arange(0,1,0.01)
y2_1 = np.sqrt(1-x2_1**2)
y2_2 = -np.sqrt(1-x2_1**2)
y2_3 = np.sqrt(1-x2_2**2)
y2_4 = -np.sqrt(1-x2_2**2)
plt.plot(x2_1, y2_1, "b", x2_1, y2_2, "b", x2_2, y2_3, "b", x2_2, y2_4, "b")
plt.show()

# 7.1.3
plt.axvline(x = -1, ymin = -1, ymax = 1, color = "b")
plt.axvline(x = 1, ymin = -1, ymax = 1, color = "b")
plt.axhline(y = -1, xmin = -1, xmax = 1, color = "b")
plt.axhline(y = 1, xmin = -1, xmax = 1, color = "b")
plt.show()

# 8.1
v1 = np.random.multivariate_normal([0,0],[[1,0],[0,1]],100)
x1 = v1[0:100,0]
y1 = v1[0:100,1]
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.scatter(x1,y1)
plt.savefig("hw1_8_1.png")

# 8.2
v2 = np.random.multivariate_normal([1,-1],[[1,0],[0,1]],100)
x2 = v2[0:100,0]
y2 = v2[0:100,1]
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.scatter(x2,y2)
plt.savefig("hw1_8_2.png")

# 8.3
v3 = np.random.multivariate_normal([0,0],[[2,0],[0,2]],100)
x3 = v3[0:100,0]
y3 = v3[0:100,1]
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.scatter(x3,y3)
plt.savefig("hw1_8_3.png")

# 8.4
v4 = np.random.multivariate_normal([0,0],[[2,0.2],[0.2,2]],100)
x4 = v4[0:100,0]
y4 = v4[0:100,1]
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.scatter(x4,y4)
plt.savefig("hw1_8_4.png")

# 8.5
v5 = np.random.multivariate_normal([0,0],[[2,-0.2],[-0.2,2]],100)
x5 = v5[0:100,0]
y5 = v5[0:100,1]
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.scatter(x5,y5)
plt.savefig("hw1_8_5.png")

