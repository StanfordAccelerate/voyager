import sys
import re
import os

def parse_and_annotate(log_files):
    # Regex to match standard clang-tidy output: "filepath:line:col: level: message"
    pattern = re.compile(r"^(.*?):(\d+):(\d+):\s+(warning|error|note):\s+(.*)$")
    
    for filepath in log_files:
        if not os.path.exists(filepath):
            print(f"::error::Could not find log file {filepath}")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    file, line_num, col, level, msg = match.groups()
                    
                    # Map clang-tidy severity to GitHub Actions severity
                    if level == "error":
                        gh_level = "error"
                    elif level == "warning":
                        gh_level = "warning"
                    else:
                        gh_level = "notice"
                        
                    # Print the GitHub Actions workflow command
                    # Format: ::warning file={name},line={line},col={col}::{message}
                    print(f"::{gh_level} file={file},line={line_num},col={col}::{msg}")

if __name__ == "__main__":
    # Pass all clang_tidy_output_*.txt files as arguments to this script
    if len(sys.argv) < 2:
        print("Usage: python clang_tidy_to_github.py <log_file1.txt> <log_file2.txt> ...")
        sys.exit(1)
        
    parse_and_annotate(sys.argv[1:])
