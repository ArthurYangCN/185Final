import os
import ssl
import urllib.request
import tarfile

# Solved------------- SL that doesn't verify 
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def download_cifar10():
    
    os.makedirs('./data/cifar-10-batches-py', exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "./data/cifar-10-python.tar.gz"
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(filename, 'wb') as f:
            f.write(response.read())
    
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall('./data/')
    os.remove(filename)
    

if __name__ == "__main__":
    download_cifar10()