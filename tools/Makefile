all:
all:
	scl enable devtoolset-10 'g++ decode.cc -o decode -std=c++17 -I..'
	scl enable devtoolset-10 'g++ interpreter.cc -o interpreter -std=c++17 -I..'

mobilebert:
	rm -rf ../data/mobilebert/*
	mkdir ../data/mobilebert/activations
	mkdir ../data/mobilebert/weights
	python3 pkl_parser_resnet.py -i ../data/mobilebert_activations.pkl -o ../data/mobilebert/activations/ -t posit8
	python3 pkl_parser_resnet.py -i ../data/mobilebert_weights.pkl -o ../data/mobilebert/weights/ -t posit8