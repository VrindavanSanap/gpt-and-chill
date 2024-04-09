#!/usr/bin/python3

#https://en.wikipedia.org/wiki/Symmetric_derivative
def get_minimizer(iterations: int, learning_rate: float, init: int)->float:
  def f(x):
    return x**2
  for i in range(iterations):
    h = 0.001
    slope = (f(init + h) - f(init - h))/(2 * h)
    init -= slope * learning_rate
  return round(init, 5)
res = get_minimizer(10, 0.01, 5)
print(res)
