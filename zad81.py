import math

def forwardPass (wiek, waga, wzrost):
    w1 = (wiek * (-0.46122)) + (waga * 0.97314) + (wzrost * (-0.39203)) + 0.80109
    w1 = 1 / (1 + math.exp(-w1))
    w2 = (wiek * 0.78548) + (waga * 2.10584) + (wzrost * (-0.57847)) + 0.43529
    w2 = 1 / (1 + math.exp(-w2))
    w3 = (w1 * (-0.81546)) + (w2 * 1.03775) - 0.2368
    return w3


print(forwardPass(23, 75, 176))