import subprocess
import os

TESTS = {
  'mobilebert':
    [
      # 'inputBottleneck',
      # 'qkvProjection',
      # 'qkAttention',
      # 'vAttention',
      # 'wProjection',
      # 'ffn1',
      # 'ffn2',
      # 'outputBottleneck'
    ],
  'resnet':
    [
      "conv1",
      "layer1_0_conv1",
      "layer1_0_conv2",
      "layer1_1_conv1",
      "layer1_1_conv2",
      "layer2_0_downsample",
      "layer2_0_conv1",
      "layer2_0_conv2",
      "layer2_1_conv1",
      "layer2_1_conv2",
      "layer3_0_downsample",
      "layer3_0_conv1",
      "layer3_0_conv2",
      "layer3_1_conv1",
      "layer3_1_conv2",
      "layer4_0_downsample",
      "layer4_0_conv1",
      "layer4_0_conv2",
      "layer4_1_conv1",
      "layer4_1_conv2",
      # "fc"
    ]
}

os.makedirs('test_outputs', exist_ok=True)

test_results = []


for group in TESTS:
  for test in TESTS[group]:
    file = open('test_outputs/' + group + '.' + test + '.out', 'w')
    test_results.append(
      [
        group + "." +test, 
        subprocess.Popen(['make', 'sim', 'GROUP='+group, 'TESTS='+test, 'SIMS=accelerator,customposit'], stdout=file, stderr=subprocess.STDOUT)
      ])

passing_tests = []
failing_tests = []

for test in test_results:
  test[1].communicate()
  if test[1].wait() == 0:
    print(test[0])
    file = 'test_outputs/'+test[0] + '.out'
    output = subprocess.Popen(['tail', '-n' , '14', file], stdout=subprocess.PIPE).stdout.readlines()
    for line in output:
      print(line.decode('utf-8').rstrip())
    

    passing_tests.append(test[0])
  else:
    failing_tests.append(test[0])
  
print("Passing Tests:")
print("--------------")
print(*passing_tests, sep='\n')

print("\n")

print("Failing Tests:")
print("--------------")
print(*failing_tests, sep='\n')
