all:
all:
	scl enable devtoolset-10 'g++ decode.cc -o decode -std=c++17 -I..'
	scl enable devtoolset-10 'g++ interpreter.cc -o interpreter -std=c++17 -I..'

mobilebert:
	rm -rf ../data/mobilebert/*
	python3 pkl_parser_resnet.py -i ../data/mobilebert_activations.pkl -o ../data/mobilebert -t posit8
	python3 pkl_parser_resnet.py -i ../data/mobilebert_weights.pkl -o ../data/mobilebert -t posit8

mobilebert_scaled:
	rm -rf ../data/mobilebert_scaled/*
	python3 pkl_parser_resnet.py -i ../data/mobilebert_activations.pkl -o ../data/mobilebert_scaled -t posit8
	python3 pkl_parser_resnet.py -i ../data/mobilebert_weights_scaled.pkl -o ../data/mobilebert_scaled -t posit8

mobilebert_errors:
	rm -rf ../data/mobilebert_errors/*
	python3 pkl_parser_resnet.py -i ../data/mobilebert_errors.pkl -o ../data/mobilebert_errors -t posit8
	python3 pkl_parser_resnet.py -i ../data/mobilebert_weights.pkl -o ../data/mobilebert_errors -t posit8