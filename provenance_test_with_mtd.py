import hashlib
import json
import numpy as np
import pandas as pd
import struct
import time
from typing import Dict, List, Any, Optional, Tuple

# Constants
SHUFFLE_SEED = 42

class MerkleTree:
    """Implementation of a Merkle Tree for data integrity verification."""
    
    def __init__(self, data_blocks: List[Any]):
        """Initialize the Merkle Tree with data blocks."""
        # Convert data blocks to strings using JSON serialization for consistency
        self.leaves = [self._hash(json.dumps(item, sort_keys=True)) for item in data_blocks]
        self.root = self._build_tree(self.leaves)
        self.data_blocks = data_blocks
        
    def _hash(self, data: str) -> str:
        """Create a SHA-256 hash of the input data."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _build_tree(self, leaves: List[str]) -> str:
        """Recursively build the Merkle Tree and return the root hash."""
        if len(leaves) == 1:
            return leaves[0]
        
        # Handle odd number of leaves by duplicating the last one
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
            
        # Pair leaves and hash them together
        new_level = []
        for i in range(0, len(leaves), 2):
            combined = leaves[i] + leaves[i+1]
            new_level.append(self._hash(combined))
            
        # Recursively build the next level
        return self._build_tree(new_level)
    
    def get_root(self) -> str:
        """Return the Merkle root hash."""
        return self.root
    
    def get_leaf_hash(self, index: int) -> str:
        """Get the hash of a specific leaf."""
        if index >= len(self.leaves):
            raise ValueError("Index out of range")
        return self.leaves[index]
    
    def generate_proof(self, index: int) -> List[Tuple[str, str]]:
        """Generate a proof path for a specific leaf node."""
        if index >= len(self.leaves):
            raise ValueError("Index out of range")
            
        proof = []
        current_index = index
        current_level = self.leaves.copy()
        
        while len(current_level) > 1:
            proof_element = None
            if current_index % 2 == 0:  # Left node
                if current_index + 1 < len(current_level):
                    proof_element = ('right', current_level[current_index + 1])
            else:  # Right node
                proof_element = ('left', current_level[current_index - 1])
                
            proof.append(proof_element)
            
            # Move to next level up
            new_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i+1]
                    new_level.append(self._hash(combined))
                else:
                    new_level.append(current_level[i])
            
            current_index = current_index // 2
            current_level = new_level
            
        return proof
    
    def verify_proof(self, leaf_hash: str, proof: List[Tuple[str, str]]) -> bool:
        """Verify a proof path against the stored root hash."""
        current_hash = leaf_hash
        
        for direction, hash_value in proof:
            if direction == 'left':
                current_hash = self._hash(hash_value + current_hash)
            else:
                current_hash = self._hash(current_hash + hash_value)
                
        return current_hash == self.root


class Sender:
    """Module to read data, create a Merkle tree, and send shuffled data."""
    
    def __init__(self, csv_file):
        """Initialize the sender with a CSV file."""
        self.csv_file = csv_file
        self.data = None
        self.merkle_tree = None
        self.shuffled_data = None
        self.position_to_id_map = None
        
    def read_data(self):
        """Read data from the CSV file."""
        # Read data from CSV file using pandas
        df = pd.read_csv(self.csv_file)
        
        # Ensure the dataframe has the expected columns
        if 'id' not in df.columns or 'measurement' not in df.columns:
            # If columns have different names, rename them
            df.columns = ['id', 'measurement']
        
        # Store the data
        self.data = df
        
        # Create a list of tuples for consistent hashing
        data_tuples = [(int(row.id), float(row.measurement)) for _, row in df.iterrows()]
        
        # Create a Merkle tree for the original data
        self.merkle_tree = MerkleTree(data_tuples)
        
        return df
    
    def shuffle_data(self):
        """Shuffle the data while keeping track of original positions."""
        if self.data is None:
            raise ValueError("No data to shuffle. Call read_data() first.")
        
        # Set the random seed for reproducibility
        np.random.seed(SHUFFLE_SEED)
        
        # Create a copy of the original dataframe with an index column
        # This index will help us restore the original order
        df_with_index = self.data.copy()
        df_with_index['original_position'] = range(len(self.data))
        
        # Shuffle the dataframe while keeping track of original positions
        shuffled_df = df_with_index.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)
        
        # Create a mapping from original positions to shuffled IDs
        self.position_to_id_map = dict(zip(shuffled_df['original_position'], self.data['id']))
        
        # Replace original IDs with shuffled IDs while keeping measurements in place
        shuffled_df['shuffled_id'] = shuffled_df['original_position'].map(self.position_to_id_map)
        
        # Sort by shuffled ID
        sorted_df = shuffled_df.sort_values('shuffled_id').reset_index(drop=True)
        
        # Store the shuffled data
        self.shuffled_data = sorted_df
        
        return sorted_df
    
    def get_merkle_root(self):
        """Get the Merkle root hash of the original data."""
        if self.merkle_tree is None:
            raise ValueError("No Merkle tree created. Call read_data() first.")
        
        return self.merkle_tree.get_root()
    
    def send_data(self):
        """Simulate sending the shuffled data."""
        if self.shuffled_data is None:
            raise ValueError("No shuffled data to send. Call shuffle_data() first.")
        
        print("Sending shuffled data:")
        print("Shuffled ID\tMeasurement")
        
        sent_data = []
        # In a real implementation, this would send over a network
        for _, row in self.shuffled_data.iterrows():
            shuffled_id = int(row['shuffled_id'])
            measurement = float(row['measurement'])
            
            print(f"{shuffled_id}\t{measurement:.2f}")
            sent_data.append((shuffled_id, measurement))
            
            # Simulate network delay
            time.sleep(0.1)
        
        # Also send the Merkle root
        print(f"\nSending Merkle root: {self.get_merkle_root()[:16]}...")
        
        return sent_data, self.get_merkle_root()


class Receiver:
    """Module to receive data, unshuffle it, and verify integrity with Merkle tree."""
    
    def __init__(self):
        """Initialize the receiver."""
        self.received_data = {}
        self.restored_data = []
        self.merkle_root = None
        
    def receive_data(self, data, merkle_root):
        """Simulate receiving data and the Merkle root."""
        print("\nReceiving data...")
        
        # Store the received data in a dictionary keyed by shuffled ID
        for shuffled_id, measurement in data:
            self.received_data[shuffled_id] = measurement
            
        # Store the received Merkle root
        self.merkle_root = merkle_root
        
        print(f"Received {len(self.received_data)} data points")
        print(f"Received Merkle root: {self.merkle_root[:16]}...")
        
        return self.received_data
    
    def unshuffle_data(self):
        """Unshuffle the received data."""
        if not self.received_data:
            raise ValueError("No data received. Call receive_data() first.")
        
        # Create a sequence of original IDs (1 to n) to match the expected output
        original_ids = list(range(1, len(self.received_data) + 1))
        
        # Set the random seed for reproducibility (same as sender)
        np.random.seed(SHUFFLE_SEED)
        
        # Shuffle the original IDs the same way the sender did
        shuffled_ids = np.random.permutation(original_ids)
        
        # Create a mapping from shuffled positions to original IDs
        position_to_original_id = {pos: orig_id for pos, orig_id in enumerate(shuffled_ids, 1)}
        
        # Apply the mapping to get original IDs
        self.restored_data = []
        for shuffled_id, measurement in self.received_data.items():
            original_id = position_to_original_id.get(shuffled_id, shuffled_id)
            self.restored_data.append((original_id, measurement))
        
        # Sort by original ID
        self.restored_data.sort(key=lambda x: x[0])
        
        print("\nUnshuffled data (restored and sorted by ID):")
        print("ID\tMeasurement")
        for id_val, meas in self.restored_data:
            print(f"{id_val}\t{meas:.2f}")
        
        return self.restored_data
    
    def verify_integrity(self):
        """Verify the integrity of the received data using the Merkle root."""
        if not self.restored_data or not self.merkle_root:
            raise ValueError("No data or Merkle root. Call receive_data() and unshuffle_data() first.")
            
        # IMPORTANT: Create data tuples in EXACTLY the same format as the sender
        # The order matters - must be (id, measurement) to match sender
        data_tuples = []
        for id_val, meas in self.restored_data:
            # Convert to exact same types as sender used
            data_tuple = (int(id_val), float(meas))
            data_tuples.append(data_tuple)
        
        # Sort by ID to ensure same order as original data
        data_tuples.sort(key=lambda x: x[0])
        
        # Create a new Merkle tree with the exact same format
        new_merkle_tree = MerkleTree(data_tuples)
        
        # Get the root hash of the new Merkle tree
        new_root = new_merkle_tree.get_root()
        
        # Compare with the received root hash
        is_valid = (new_root == self.merkle_root)
        
        print("\nVerifying data integrity...")
        print(f"Original Merkle root: {self.merkle_root[:16]}...")
        print(f"Computed Merkle root: {new_root[:16]}...")
        print(f"Data integrity verification: {'Passed' if is_valid else 'Failed'}")
        
        return is_valid


def run_demo():
    """Demonstrate the full system."""
    print("=== Data Integrity Verification with Shuffling Demo ===\n")
    
    # 1. Sender reads and shuffles data
    sender = Sender('measurements.csv')
    sender.read_data()
    sender.shuffle_data()
    
    # 2. Sender sends data and Merkle root
    sent_data, merkle_root = sender.send_data()
    
    # 3. Receiver gets data
    receiver = Receiver()
    receiver.receive_data(sent_data, merkle_root)
    
    # 4. Receiver unshuffles data
    receiver.unshuffle_data()
    
    # 5. Receiver verifies data integrity
    integrity_verified = receiver.verify_integrity()
    
    # 6. Report the overall result
    if integrity_verified:
        print("\n✓ Success: Data transmission and reshuffling completed without any tampering!")
    else:
        print("\n✗ Error: Data integrity verification failed. Possible tampering detected!")
    
    return integrity_verified


def run_demo_with_tampering():
    """Demonstrate the system with deliberate tampering."""
    print("=== Data Integrity Verification with Tampering Demo ===\n")
    
    # 1. Sender reads and shuffles data
    sender = Sender('measurements.csv')
    sender.read_data()
    sender.shuffle_data()
    
    # 2. Sender sends data and Merkle root
    sent_data, merkle_root = sender.send_data()
    
    # 3. Simulate tampering by modifying one measurement
    if sent_data:
        tampered_index = np.random.randint(0, len(sent_data))
        original_value = sent_data[tampered_index][1]
        sent_data[tampered_index] = (sent_data[tampered_index][0], original_value+0.02)
        
        print(f"\n! Simulating tampering: Modified measurement at index {tampered_index}")
        print(f"  Original value: {original_value:.2f}")
        print(f"  Tampered value: {sent_data[tampered_index][1]:.2f}")
    
    # 4. Receiver gets data
    receiver = Receiver()
    receiver.receive_data(sent_data, merkle_root)
    
    # 5. Receiver unshuffles data
    receiver.unshuffle_data()
    
    # 6. Receiver verifies data integrity
    integrity_verified = receiver.verify_integrity()
    
    # 7. Report the overall result
    if integrity_verified:
        print("\n✗ Problem: Tampering was not detected!")
    else:
        print("\n✓ Success: Tampering was correctly detected by Merkle tree verification!")
    
    return integrity_verified


if __name__ == "__main__":
    # Run the normal demo first
    print("Running demo without tampering:")
    run_demo()
    
    print("\n" + "="*50 + "\n")
    
    # Then run the demo with tampering
    print("Running demo with tampering:")
    run_demo_with_tampering() 