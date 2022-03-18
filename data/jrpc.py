import requests
import json

headers = {
    'content-type':'application/json'
}

url = 'http://localhost:8545'

def eth_jrpc(method, params=[], id=1):
    data = {
        "jsonrpc":"2.0",
        "method":method,
        "params":params,
        "id":id
    }
    data = json.dumps(data)
    response = requests.post(url, headers=headers, data=data)
    return response

if __name__=="__main__":

    import sys

    if len(sys.argv)<2:
        print("Usage: jrpc method [params]")
        sys.exit(0)

    method = sys.argv[1]
    params = []
    if len(sys.argv)>2:
        for param in sys.argv[2:]:
            params.append(eval(param))

    response = eth_jrpc(method, params)
    
    print(response)
    print(response.status_code)
    print(json.loads(response.content))

