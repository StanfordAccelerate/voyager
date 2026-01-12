import argparse
import os
import csv

def main():
    parser = argparse.ArgumentParser(description="Compare two FSDB files")
    parser.add_argument("file1", help="First FSDB file")
    parser.add_argument("file1_hierarchical_path", help="Hierarchical path to accelerator module" )
    parser.add_argument("file2", help="Second FSDB file")
    parser.add_argument("file2_hierarchical_path", help="Hierarchical path to accelerator module" )
    parser.add_argument("-type", help="Type of comparison (top|PE)", default="top")
    args = parser.parse_args()

    top_level_interface_signals = [
        "inputDataResponse",
        "weightDataResponse",
        "vectorFetch0DataResponse",
        "vectorFetch1DataResponse",
        "vectorFetch2DataResponse",
        "vectorOutput"
    ]
    pe_interface_signals = [
        "weightIn",
        "weightOut",
        "inputIn",
        "inputOut",
        "psumIn",
        "psumOut"
    ]
    all_pe_interface_signals = [f"pe_{i}.{signal}" for i in range(32*32) for signal in pe_interface_signals]

    signals_to_compare = top_level_interface_signals if args.type == "top" else all_pe_interface_signals

    compare_signals(args.file1, args.file1_hierarchical_path, args.file2, args.file2_hierarchical_path, signals_to_compare)

def compare_signals(file1, file1_hierarchical_path, file2, file2_hierarchical_path, signals):
    for signal in signals:
        values1 = get_value_changes(file1, file1_hierarchical_path, signal)
        values2 = get_value_changes(file2, file2_hierarchical_path, signal)

        if(len(values1) == 0 or len(values2) == 0):
            print(f"Warning: No values found for signal {signal}. Skipping comparison.")
            continue

        # iterate through both and compare the data values
        # report time of first difference
        for i in range(min(len(values1), len(values2))):
            if values1[i][1] != values2[i][1]:
                print(f"***Difference in signal {signal}***")
                print(f"file 1: {values1[i][1]} (time: {values1[i][0]})")
                print(f"file 2: {values2[i][1]} (time: {values2[i][0]})")
                break


def get_value_changes(file, hierarchical_path, signal):
    # if hierachical path does not end with a /, add it
    if hierarchical_path[-1] != '/':
        hierarchical_path += '/'

    data_signal = f"{hierarchical_path}{signal}_dat"
    ready_signal = f"{hierarchical_path}{signal}_rdy"
    valid_signal = f"{hierarchical_path}{signal}_vld"
    cmd = f"fsdbreport {file} -s '{data_signal}' -exp \"{ready_signal}==1 && {valid_signal}==1 && {hierarchical_path}clk==1\"  -csv -of h"

    # run cmd and suppress output
    os.system(f"{cmd} > /dev/null")

    with open('report.txt', newline='') as csvfile:
        reader = csv.reader(csvfile)
        values = [row for row in reader]

    # remove header row
    values = values[1:]
    return values

if __name__ == "__main__":
    main()