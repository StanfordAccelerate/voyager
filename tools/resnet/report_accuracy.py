total_samples = 0.0
gold_total_correct = 0.0
sim_total_correct = 0.0
for i in range(36):
	f = open("pybuild/accuracy" + str(i))
	lines = f.readlines()

	samples = len(lines)
	
	sim_start = lines[-1].find("sim: ") +  len("sim: ")	
	sim_end = lines[-1].find(",")	

	gold_start = lines[-1].find("gold: ") + len("gold:" )
	gold_end = lines[-1].find(",", gold_start)

	sim_acc = float(lines[-1][sim_start: sim_end])
	gold_acc = float(lines[-1][gold_start: gold_end])

	sim_total_correct += sim_acc * float(samples)
	gold_total_correct += gold_acc * float(samples)
	
	print(i, sim_acc, gold_acc, samples)
	total_samples += samples

print("Overall accuracy is sim: "+str(sim_total_correct / total_samples)+" and gold: "+str(gold_total_correct / total_samples)+" with "+str(total_samples)+" samples")

print("\n Complete Blacklist:")
for i in range(36):
	f = open("pybuild/blacklist" + str(i))
	lines = f.readlines()
	
	for line in lines:
		print(line)

