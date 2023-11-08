import gzip
import json
from hashlib import sha256

class Hashable:
    empty = '\x00'

    @staticmethod
    def sha256(string):
        return sha256(string.encode()).hexdigest()

    def __init__(self, json_obj, hashable_fields):
        self.json_obj = json_obj
        self.hashable_fields = sorted(hashable_fields)

    def hash(self):
        hash_string = ','.join([str(self.json_obj[field]) for field in self.hashable_fields])
        return Hashable.sha256(hash_string)


class Transaction(Hashable):
    hashable_fields = [
        'amount',
        'lock_time',
        'receiver',
        'sender',
        'signature',
        'transaction_fee',
    ]

    def __init__(self, json_obj):
        super().__init__(json_obj, Transaction.hashable_fields)
        self.amount = json_obj['amount']
        self.lock_time = json_obj['lock_time']
        self.json_obj = json_obj

    def to_json(self):
        return self.json_obj

class TransactionFactory:
    def __init__(self, json_transactions):
        self.transactions = [Transaction(json_transaction) for json_transaction in json_transactions]
        self.transactions.sort(key=lambda transaction: transaction.amount, reverse=True)

    def pop_top_transactions_before(self, timestamp, n=100):
        ret = [transaction for transaction in self.transactions if transaction.lock_time < timestamp][:n]
        [self.transactions.remove(x) for x in ret]
        return ret

    @staticmethod
    def merkle_root(transactions):
        hashes = [transaction.hash() for transaction in transactions]
        while len(hashes) > 1:
            new_hashes = []
            if len(hashes) % 2 == 1:
                hashes.append(Hashable.empty)
            for h1, h2 in zip(hashes[::2], hashes[1::2]):
                if h1 < h2:
                    new_hash_string = h1 + h2
                else:
                    new_hash_string = h2 + h1
                new_hashes.append(Hashable.sha256(new_hash_string))
            hashes = new_hashes
        return hashes[0]

class Block(Hashable):
    hashable_fields = [
        "difficulty",
        "height",
        "transactions_merkle_root",
        "miner",
        "previous_block_header_hash",
        "timestamp",
        "transactions_count",
        "nonce",
    ]

    @staticmethod
    def from_args(
        difficulty,
        height,
        transactions_merkle_root,
        miner,
        previous_block_header_hash,
        timestamp,
        transactions_count,
        nonce,
        transactions,
    ):
        json_obj = {
            "header": {
                "difficulty": difficulty,
                "height": height,
                "transactions_merkle_root": transactions_merkle_root,
                "miner": miner,
                "previous_block_header_hash": previous_block_header_hash,
                "timestamp": timestamp,
                "transactions_count": transactions_count,
                "nonce": nonce,
            },
            "transactions": transactions,
        }
        return Block(json_obj)

    def __init__(self, block_json):
        super().__init__(block_json["header"], Block.hashable_fields)
        self.header = block_json["header"]
        self.transactions = block_json["transactions"]

        # Helper fields
        self.difficulty = self.header["difficulty"]
        self.height = self.header["height"]
        self.miner = self.header["miner"]
        self.transactions_merkle_root = self.header["transactions_merkle_root"]
        self.previous_block_header_hash = self.header["previous_block_header_hash"]
        self.timestamp = self.header["timestamp"]

    def try_mine(self):
        current_hash = self.hash()
        if current_hash[:self.difficulty] == '0' * self.difficulty:
            self.header['nonce'] = self.json_obj['nonce']
            self.header['hash'] = "0x" + current_hash
            return True

        self.json_obj['nonce'] += 1
        return False

    def to_json(self):
        transactions_json = [transaction.to_json() for transaction in self.transactions]

        return {
            "header": self.header,
            "transactions": transactions_json,
        }

class BlockFactory:
    def __init__(self, block_json, transactions_json, start_difficulty=1):
        self.current_block = Block(block_json)
        self.transaction_factory = TransactionFactory(transactions_json)
        self.mined_blocks_count = (start_difficulty-1)*10 + 1

    def prepare_block(self):
        transactions = self.transaction_factory.pop_top_transactions_before(self.current_block.timestamp)

        difficulty = self.mined_blocks_count // 10 + 1
        height = self.current_block.height + 1
        transactions_merkle_root = TransactionFactory.merkle_root(transactions)
        miner = self.current_block.miner
        previous_block_header_hash = self.current_block.transactions_merkle_root
        timestamp = self.current_block.timestamp + 10
        transactions_count = len(transactions)
        nonce = 0

        block = Block.from_args(
            difficulty,
            height,
            transactions_merkle_root,
            miner,
            previous_block_header_hash,
            timestamp,
            transactions_count,
            nonce,
            transactions,
        )
        return block

    def mine(self):
        block = self.prepare_block()
        while not block.try_mine():
            pass

        self.mined_blocks_count += 1
        self.current_block = block
        return block

    mine_n = lambda self, n: [self.mine() for _ in range(n)]

    def save_blocks(self, blocks, path='blocks.json'):
        with open(path, 'w') as f:
            json.dump([block.to_json() for block in blocks], f)

path_blockchain = 'data/blockchain.json.gz'
path_mempool = 'data/mempool.json.gz'

# Try out difficulty 6
if __name__ == '__main__':

    with gzip.open(path_blockchain, 'r') as f:
        test_blocks = json.load(f)

    with gzip.open(path_mempool, 'r') as f:
        transactions_json = json.load(f)

    block_factory = BlockFactory(test_blocks[0], transactions_json, start_difficulty=5)
    blocks = block_factory.mine_n(1)

    block_factory.save_blocks(blocks)
