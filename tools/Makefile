all:
all:
	scl enable devtoolset-10 'g++ decode.cc -o decode -std=c++17 -I..'
	scl enable devtoolset-10 'g++ interpreter.cc -o interpreter -std=c++17 -I..'

mobilebert:
	rm -rf ../data/mobilebert/*
	python3 pkl_parser.py -i ../data/mobilebert_activations.pkl -o ../data/mobilebert/activations -t posit8
	python3 pkl_parser.py -i ../data/mobilebert_weights.pkl -o ../data/mobilebert/weights -t posit8
	python3 pkl_parser.py -i ../data/mobilebert_weights_scaled.pkl -o ../data/mobilebert/weights_scaled -t posit8
	python3 pkl_parser.py -i ../data/mobilebert_errors.pkl -o ../data/mobilebert/errors -t posit8
	python3 pkl_parser.py -i ../data/mobilebert_gradients.pkl -o ../data/mobilebert/gradients -t posit8