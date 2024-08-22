import random
from datetime import datetime

def generate_log_entry():
    ips = ["127.0.0.1", "192.168.1.1", "10.0.0.2"]
    methods = ["GET", "POST", "DELETE"]
    urls = ["/index.html", "/login", "/register"]
    statuses = [200, 401, 404, 500]
    sizes = [2326, 525, 1042, 512]

    ip = random.choice(ips)
    method = random.choice(methods)
    url = random.choice(urls)
    status = random.choice(statuses)
    size = random.choice(sizes)
    timestamp = datetime.now().strftime("%d/%b/%Y:%H:%M:%S")

    return f'{ip} - - [{timestamp}] "{method} {url} HTTP/1.1" {status} {size}'

def generate_log_file(file_name, num_entries=100):
    with open(file_name, 'w') as f:
        for _ in range(num_entries):
            f.write(generate_log_entry() + "\n")

generate_log_file('server.log', 100)
