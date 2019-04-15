# "High-frequency trading in a limit order book"
# by Marco Avellaneda and Sasha Stoikov"

# https://math.nyu.edu/faculty/avellane/HighFrequencyTrading.pdf
# https://quant.stackexchange.com/questions/36400/avellaneda-stoikov-market-making-model

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import time

while True:
	# Parameters for mid price simulation:
	S0 = 100. 			# initial price
	T = 1.0  			# time
	sigma = 2 			# volatility
	M = 20000 			# number of time steps
	dt = T/M 			# time step
	Sim = 1000  		# number of simulations
	gamma = 0.1 		# risk aversion
	k = 1.5				#
	A = 140				# order probability
	I = 1				# stocks
	M_float = float(M)
	sqrt_dt = math.sqrt(dt)

	# Results:
	AverageSpread = []
	Profit = []
	Std = []
	for i in range(1, Sim+1):
		# reservation price:
		# 	r(s,t) = s - q * gamma * sigma**2 * (T-t)
		S = np.zeros((M+1, I))
		Bid = np.zeros((M+1, I))
		Ask = np.zeros((M+1, I))
		ReservPrice = np.zeros((M+1, I))
		spread = np.zeros((M+1, I))
		deltaB = np.zeros((M+1, I))
		deltaA = np.zeros((M+1, I))
		q = np.zeros((M+1, I))
		w = np.zeros((M+1, I))
		equity = np.zeros((M+1, I))

		S[0] = S0
		ReservPrice[0] = S0
		Bid[0] = S0
		Ask[0] = S0
		spread[0] = 0
		deltaB[0] = 0
		deltaA[0] = 0
		q[0] = 0 		# position
		w[0] = 0 		# wealth
		equity[0] = 0
		for t in range(1, M+1):

			z = np.random.standard_normal(I)			# dW
			S[t] = S[t-1] + sigma * sqrt_dt * z			# new S value
			ReservPrice[t] = S[t] - q[t-1] * gamma * (sigma ** 2) * (T - t/M_float)						 # Mid Price estimation
			spread[t] = gamma * (sigma ** 2) * (T - t/M_float) + (2/gamma) * math.log(1 + (gamma/k))	 # Spread estimation
			Bid[t] = ReservPrice[t] - spread[t]/2.		# bid
			Ask[t] = ReservPrice[t] + spread[t]/2.		# ask

			deltaB[t] = S[t] - Bid[t]					# difference to price
			deltaA[t] = Ask[t] - S[t]					# difference to price

			lambdaA = A * np.exp(-k * deltaA[t])
			ProbA = 1 - np.exp(-lambdaA * dt)
			fa = random.random()

			lambdaB = A * np.exp(-k * deltaB[t])
			ProbB = 1 - np.exp(-lambdaB * dt)
			fb = random.random()

			if ProbB > fb and ProbA < fa:				# buy market order filling our limit order
				q[t] = q[t-1] + 1						# position
				w[t] = w[t-1] - Bid[t]					# wealth

			if ProbB < fb and ProbA > fa:				# sell market order filling our limit order
				q[t] = q[t-1] - 1						# position
				w[t] = w[t-1] + Ask[t]					# wealth

			if ProbB < fb and ProbA < fa:				# no order
				q[t] = q[t-1]							# position
				w[t] = w[t-1]							# wealth

			if ProbB > fb and ProbA > fa:				# both sides order
				q[t] = q[t-1]							# position
				w[t] = w[t-1] - Bid[t]					# wealth
				w[t] = w[t] + Ask[t]

			equity[t] = w[t] + q[t] * S[t]
		AverageSpread.append(spread.mean())
		Profit.append(equity[-1])
		Std.append(np.std(equity))

	print("                   Results              ")
	print("----------------------------------------")
	print("%14s %21s" % ('statistic', 'value'))
	print(40 * "-")
	print("%14s %20.5f" % ("Average spread :", np.array(AverageSpread).mean()))
	print("%16s %20.5f" % ("Profit :", np.array(Profit).mean()))
	print("%16s %20.5f" % ("Std(Profit) :", np.array(Std).mean()))

	# Plots:
	x = np.linspace(0., T, num=(M+1))

	fig = plt.figure(figsize=(10, 8))
	plt.subplot(2, 2, 1)   						# number of rows, number of  columns, number of the subplot
	plt.plot(x, S[:], lw=1., label='S')
	plt.plot(x, Ask[:], lw=1., label='Ask')
	plt.plot(x, Bid[:], lw=1., label='Bid')
	plt.grid(True)
	plt.legend(loc=0)
	plt.ylabel('P')
	plt.title('Prices')

	plt.subplot(2, 2, 2)
	plt.plot(x, q[:], 'g', lw=1., label='q') 	# plot 2 lines
	plt.grid(True)
	plt.legend(loc=0)
	plt.axis('tight')
	plt.xlabel('Time')
	plt.ylabel('Position')
	#plt.show()


	plt.subplot(2, 2, 4)
	plt.plot(x, equity[:], 'b', lw=1., label='equity')
	plt.grid(True)
	plt.legend(loc=0)
	plt.axis('tight')
	plt.xlabel('Time')
	plt.ylabel('Position')

	# Histogram of profit:
	#plt.figure(figsize=(7, 5))
	plt.subplot(2, 2, 3)
	plt.hist(np.array(Profit), label=['Inventory strategy'], bins=100)
	plt.legend(loc=0)
	plt.grid(True)
	plt.xlabel('pnl')
	plt.ylabel('number of values')
	plt.title('Histogram')
	plt.show()

	time.sleep(1)
