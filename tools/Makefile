all:
all:
	scl enable devtoolset-10 'g++ decode.cc -o decode -std=c++17 -I..'
	scl enable devtoolset-10 'g++ interpreter.cc -o interpreter -std=c++17 -I..'

mobilebert:
	rm -rf ../data/mobilebert/*
	python3 pkl_parser_resnet.py -i ../data/mobilebert_activations.pkl -o ../data/mobilebert -t posit8
	python3 pkl_parser_resnet.py -i ../data/mobilebert_weights.pkl -o ../data/mobilebert -t posit8