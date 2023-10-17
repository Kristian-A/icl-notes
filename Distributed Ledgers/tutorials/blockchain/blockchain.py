import gzip
import json

class Transaction:

    @staticmethod
    def generate_transactions(json_transactions):
        return [Transaction(json_transaction) for json_transaction in json_transactions]

    def __init__(self, json_obj):
        self.amount = json_obj['amount']
        self.lock_time = json_obj['lock_time']
        self.receiver = json_obj['receiver']
        self.sender = json_obj['sender']
        self.signature = json_obj['signature']
        self.transaction_fee = json_obj['transaction_fee']

class Block:
    def __init__(self, json_obj):
        self.header = json_obj['header']
        self.transactions = Transaction.generate_transactions(json_obj['transactions'])

path_blockchain = 'data/blockchain.json.gz'
path_mempool = 'data/mempool.json.gz'

with gzip.open(path_blockchain, 'r') as f:
    initial_blocks = json.load(f)

with gzip.open(path_mempool, 'r') as f:
    future_blocks = json.load(f)

print(future_blocks[-50])

latest_block = Block(initial_blocks[-1])

print(latest_block.transactions[0].sender)
