from agent import ClientAgent

# server_url = 'http://192.168.0.209:11110'
server_url = 'http://192.168.0.191:11110'

def main():
    agent = ClientAgent(server_url)
    agent.start()

if __name__ == "__main__":
    main()
