import numpy as np

# Newtonverfahren

def newton(f, df, x0, tol, max_iter):
	x = x0
	for i in range(max_iter):
		x = x - f(x) / df(x)
		print("x = ", x, "f(x) = ", f(x))
		if abs(f(x)) < tol:
			return x
	return x


def main():
	f = np.poly1d([1, 0, -2, 2])
	df = f.deriv()
	x0 = 2
	tol = 1e-10
	max_iter = 50
	x = newton(f, df, x0, tol, max_iter)
	print(x)



if __name__ == "__main__":
	main()