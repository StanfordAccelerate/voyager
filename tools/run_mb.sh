make clean && make build/MobilebertTest

python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 0 --end 25 --id 0 > logs/run0.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 25 --end 50 --id 1 > logs/run1.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 50 --end 75 --id 2 > logs/run2.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 75 --end 100 --id 3 > logs/run3.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 100 --end 125 --id 4 > logs/run4.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 125 --end 150 --id 5 > logs/run5.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 150 --end 175 --id 6 > logs/run6.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 175 --end 200 --id 7 > logs/run7.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 200 --end 225 --id 8 > logs/run8.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 225 --end 250 --id 9 > logs/run9.log &

python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 250 --end 275 --id 10 > logs/run10.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 275 --end 300 --id 11 > logs/run11.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 300 --end 325 --id 12 > logs/run12.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 325 --end 350 --id 13 > logs/run13.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 350 --end 375 --id 14 > logs/run14.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 375 --end 400 --id 15 > logs/run15.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 400 --end 425 --id 16 > logs/run16.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 425 --end 450 --id 17 > logs/run17.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 450 --end 475 --id 18 > logs/run18.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 475 --end 500 --id 19 > logs/run19.log &

python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 500 --end 525 --id 20 > logs/run20.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 525 --end 550 --id 21 > logs/run21.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 550 --end 575 --id 22 > logs/run22.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 575 --end 600 --id 23 > logs/run23.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 600 --end 625 --id 24 > logs/run24.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 625 --end 650 --id 25 > logs/run25.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 650 --end 675 --id 26 > logs/run26.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 675 --end 700 --id 27 > logs/run27.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 700 --end 725 --id 28 > logs/run28.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 725 --end 750 --id 29 > logs/run29.log &

python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 750 --end 775 --id 30 > logs/run30.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 775 --end 800 --id 31 > logs/run31.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 800 --end 825 --id 32 > logs/run32.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 825 --end 850 --id 33 > logs/run33.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 850 --end 875 --id 34 > logs/run34.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 875 --end 900 --id 35 > logs/run35.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 900 --end 925 --id 36 > logs/run36.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 925 --end 950 --id 37 > logs/run37.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 950 --end 975 --id 38 > logs/run38.log &
python3 -u tools/mobilebert_inference_sub.py --model_name_or_path models/mobilebert --start 975 --end 1000 --id 39 > logs/run39.log &