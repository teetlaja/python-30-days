# Create a script that reads a log file line by line.
# Use regular expressions (re module) to extract data like status codes and URLs.
# Write the extracted data into a summary report file.
# Implement error handling for common issues like missing files or incorrect formats.

import re
import sys



def read_log_file(file_path):
    try:
        with open(file_path, 'r') as log_file:
            return log_file.readlines()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)


log_file_path = 'server.log'
log_content = read_log_file(log_file_path)



http_response_pattern = re.compile(r'(?<=\s)\d{3}(?=\s)')

log_data = {
    'Requests by code': {}
}

for line in log_content:
    match = http_response_pattern.search(line)
    log_data['Total requests'] = log_data.get('Total requests', 0) + 1
    if match:
        http_response_code = match.group(0)

        if http_response_code in log_data['Requests by code']:
            log_data['Requests by code'][http_response_code] += 1
        else:
            log_data['Requests by code'][http_response_code] = 1
            print(f"HTTP Response Code: {http_response_code}")


print(log_data)
