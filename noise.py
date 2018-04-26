import numpy as np
from matplotlib import pyplot as plt

# creates data set for 2 lines
# returns y1, y2 for x [0, 500)
def createLinearTestData():
    x1 = []
    y1 = []

    x2 = []
    y2 = []

    x_intercept = 0

    while 0 <= x_intercept <= 500:
        m1 = np.random.uniform(-20, 20)
        b1 = np.random.uniform(-5000, 5000)

        m2 = np.random.normal(m1, .5)
        b2 = np.random.uniform(-5000, 5000)

        m_diff = m2-m1
        b_diff = b1-b2
        x_intercept = b_diff - m_diff

    height = abs(b2 - b1)

    for i in range(0, 500):
        # first corner's line

        noise_x1 = np.random.normal(0, .01 * height)
        noise_y1 = np.random.normal(0, .1 * height)
        x1.append(i)
        y1.append(m1 * (x1[-1] + noise_x1) + (b1 + noise_y1))

        # second corner's line
        noise_x2 = np.random.normal(0, .01 * height)
        noise_y2 = np.random.normal(0, .1 * height)
        x2.append(i)
        y2.append(m2 * (x2[-1] + noise_x2) + (b2 + noise_y2))

    print("m1 ", m1, '\t', "|| b1 ", b1)
    print("m2 ", m2, '\t', "|| b2 ", b2)

    #plt.plot(x1, y1, '1', x2, y2, '2')
    #plt.show()

    return y1, y2




# creates data set for one sinusoid line
def createSinusoidTestData() :
    x1 = []
    y1 = []

    x2 = []
    y2 = []

    a1 = np.random.uniform(-1, 1)
    b1 = np.random.uniform(-5, 5)

    for i in range(-250, 250):
        # first corner's line
        noise_x1 = np.random.normal(0.0, .1)
        noise_y1 = np.random.normal(0.0, .5)
        x1.append(i)
        y1.append(a1 * np.sin(b1 * x1[-1] + noise_x1) + noise_y1)

    plt.plot(x1, y1)
    plt.show()

def createMultipleLinearDataSets() :
    for i in range(20):
        createLinearTestData()
