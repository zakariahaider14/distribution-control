import hashlib
import json
import numpy as np
import pandas as pd
import struct
import time
import socket
import threading
import queue
import networkx as nx
import matplotlib.pyplot as plt
import sys
from io import StringIO
from typing import Dict, List, Any, Optional, Tuple, Union
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

# Constants
SHUFFLE_SEED = 42
DEFAULT_PORT = 8000
DEFAULT_HOST = "localhost"
CHUNK_SIZE = 1024  # bytes
TIMEOUT = 3  # seconds


class MerkleTree:
    """Fixed implementation of a Merkle Tree for data integrity verification."""
    
    def __init__(self, data_blocks: List[Any]):
        """Initialize the Merkle Tree with data blocks."""
        # Convert data blocks to strings using JSON serialization for consistency
        self.data_blocks = data_blocks
        self.leaves = [self._hash(json.dumps(item, sort_keys=True)) for item in data_blocks]
        
        # Build all levels of the tree explicitly for consistent access
        self.levels = [self.leaves]
        self._build_levels()
        
        # The root is the single node at the top level
        self.root = self.levels[-1][0]
        
        # For visualization
        self.nodes = {}
        self.node_positions = {}
        self.graph = nx.DiGraph()
        self._build_visualization_graph()
    
    def _hash(self, data: str) -> str:
        """Create a SHA-256 hash of the input data."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _build_levels(self):
        """Build all levels of the Merkle tree iteratively."""
        current_level = self.leaves
        
        while len(current_level) > 1:
            next_level = []
            # Process pairs of nodes at this level
            for i in range(0, len(current_level), 2):
                # Handle odd number of elements by duplicating the last one
                if i + 1 >= len(current_level):
                    next_level.append(current_level[i])
                else:
                    # Hash the pair of nodes
                    left = current_level[i]
                    right = current_level[i + 1]
                    combined = left + right
                    parent_hash = self._hash(combined)
                    next_level.append(parent_hash)
            
            # Add this level to our tree
            self.levels.append(next_level)
            current_level = next_level
    
    def _build_visualization_graph(self):
        """Build a graph representation for visualization."""
        # Add all nodes to the graph
        for level_idx, level in enumerate(self.levels):
            for node_idx, node_hash in enumerate(level):
                node_id = f"L{level_idx}_{node_idx}"
                self.nodes[node_id] = node_hash
                self.graph.add_node(node_id, hash=node_hash[:8], level=level_idx)
                # Set position for visualization
                self.node_positions[node_id] = (node_idx * (2**(len(self.levels) - level_idx - 1)), -level_idx)
        
        # Add edges between levels
        for level_idx in range(len(self.levels) - 1):
            current_level = self.levels[level_idx]
            for node_idx in range(0, len(current_level), 2):
                # Find parent in next level (index is halved)
                parent_idx = node_idx // 2
                parent_id = f"L{level_idx + 1}_{parent_idx}"
                left_id = f"L{level_idx}_{node_idx}"
                
                # Add edge from parent to left child
                self.graph.add_edge(parent_id, left_id)
                
                # Add edge from parent to right child if it exists
                if node_idx + 1 < len(current_level):
                    right_id = f"L{level_idx}_{node_idx + 1}"
                    self.graph.add_edge(parent_id, right_id)
    
    def get_root(self) -> str:
        """Return the Merkle root hash."""
        return self.root
    
    def get_leaf_hash(self, index: int) -> str:
        """Get the hash of a specific leaf."""
        if index >= len(self.leaves):
            raise ValueError(f"Index {index} out of range (0-{len(self.leaves)-1})")
        return self.leaves[index]
    
    def generate_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Generate a proof path for a specific leaf node.
        
        Args:
            index: The index of the leaf node
            
        Returns:
            A list of tuples (direction, hash) representing the proof path
        """
        if index >= len(self.leaves):
            raise ValueError(f"Index {index} out of range (0-{len(self.leaves)-1})")
        
        # For debugging
        print(f"Generating proof for leaf index {index}")
        print(f"Leaf hash: {self.leaves[index][:16]}...")
        
        proof = []
        idx = index
        
        # Start from the bottom level (leaves) and go up
        for level_idx, level in enumerate(self.levels[:-1]):  # Skip the root level
            print(f"  Level {level_idx}, index {idx}, leaves: {len(level)}")
            
            # Determine if current node is left or right in its pair
            is_right = idx % 2 == 1
            
            if is_right:
                # If right, we need its left sibling
                sibling_idx = idx - 1
                sibling_hash = level[sibling_idx]
                proof.append(('left', sibling_hash))
                print(f"  Adding left sibling at index {sibling_idx}: {sibling_hash[:16]}...")
            else:
                # If left, we need its right sibling
                sibling_idx = idx + 1
                
                # Check if sibling exists (for odd number of nodes)
                if sibling_idx < len(level):
                    sibling_hash = level[sibling_idx]
                    proof.append(('right', sibling_hash))
                    print(f"  Adding right sibling at index {sibling_idx}: {sibling_hash[:16]}...")
                else:
                    # For the last node with no sibling, we use itself as the "sibling"
                    # but this actually doesn't affect the proof path since this node will be used as-is
                    # in the next level, as per our tree construction logic
                    sibling_hash = level[idx]
                    proof.append(('right', sibling_hash))
                    print(f"  Adding right sibling (self) at index {idx}: {sibling_hash[:16]}...")
            
            # Update index for the next level up
            idx = idx // 2
        
        print(f"  Root hash: {self.root[:16]}...")
        return proof
    
    def verify_proof(self, leaf_hash: str, proof: List[Tuple[str, str]]) -> bool:
        """
        Verify a proof path against the stored root hash.
        
        Args:
            leaf_hash: The hash of the leaf node
            proof: A list of tuples (direction, hash) representing the proof path
            
        Returns:
            True if the proof is valid, False otherwise
        """
        print(f"Verifying proof for leaf hash: {leaf_hash[:16]}...")
        print(f"Merkle root: {self.root[:16]}...")
        
        current_hash = leaf_hash
        
        for i, (direction, sibling_hash) in enumerate(proof):
            if direction == 'left':
                # Left sibling comes first in the concatenation
                combined = sibling_hash + current_hash
            else:
                # Right sibling comes second
                combined = current_hash + sibling_hash
            
            current_hash = self._hash(combined)
            print(f"  Step {i+1}: {direction} + hash -> {current_hash[:16]}...")
        
        # Check if the computed hash matches the root
        root_match = current_hash == self.root
        print(f"  Final hash: {current_hash[:16]}...")
        print(f"  Root hash:  {self.root[:16]}...")
        print(f"  Match: {root_match}")
        
        return root_match
    
    def visualize(self, filename: Optional[str] = None) -> None:
        """
        Visualize the Merkle tree using NetworkX and matplotlib.
        Optionally save to a file if filename is provided.
        """
        plt.figure(figsize=(12, 8))
        
        try:
            # Draw the graph with custom positions
            nx.draw(self.graph, pos=self.node_positions, with_labels=False, 
                    node_color='lightblue', node_size=700, arrows=True)
            
            # Add labels with shortened hash values
            node_labels = {node: data['hash'] for node, data in self.graph.nodes(data=True)}
            nx.draw_networkx_labels(self.graph, self.node_positions, labels=node_labels, font_size=9)
            
            plt.title('Merkle Tree Visualization')
            plt.axis('off')
            
            if filename:
                plt.savefig(filename)
            plt.show()
        except Exception as e:
            print(f"Error visualizing Merkle tree: {e}")
            print("Skipping visualization and continuing with the demo...")

class NetworkProtocol:
    """Protocol for sending/receiving data over a network."""
    
    @staticmethod
    def serialize_data(data: Any) -> bytes:
        """Serialize data to JSON and convert to bytes."""
        json_data = json.dumps(data)
        return json_data.encode('utf-8')
    
    @staticmethod
    def deserialize_data(data_bytes: bytes) -> Any:
        """Deserialize JSON bytes back to Python objects."""
        json_data = data_bytes.decode('utf-8')
        return json.loads(json_data)
    
    @staticmethod
    def send_data(sock: socket.socket, data: Any) -> bool:
        """Send data over a socket with size prefix."""
        try:
            serialized = NetworkProtocol.serialize_data(data)
            # Prefix with 4-byte length
            length_prefix = struct.pack('!I', len(serialized))
            sock.sendall(length_prefix + serialized)
            return True
        except (socket.error, struct.error) as e:
            print(f"Error sending data: {e}")
            return False
    
    @staticmethod
    def receive_data(sock: socket.socket) -> Any:
        """Receive data from a socket with size prefix."""
        try:
            # First receive the 4-byte length prefix
            length_bytes = sock.recv(4)
            if not length_bytes:
                return None
                
            # Unpack the length
            length = struct.unpack('!I', length_bytes)[0]
            
            # Now receive the actual data
            data_bytes = b''
            remaining = length
            while remaining > 0:
                chunk = sock.recv(min(CHUNK_SIZE, remaining))
                if not chunk:
                    return None
                data_bytes += chunk
                remaining -= len(chunk)
                
            return NetworkProtocol.deserialize_data(data_bytes)
        except (socket.error, struct.error, json.JSONDecodeError) as e:
            print(f"Error receiving data: {e}")
            return None


class CryptoTools:
    """Cryptographic utilities for digital signatures and verification."""
    
    @staticmethod
    def generate_key_pair():
        """Generate a new RSA key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        return private_key, public_key
    
    @staticmethod
    def serialize_public_key(public_key):
        """Serialize a public key to bytes."""
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    @staticmethod
    def deserialize_public_key(serialized_key):
        """Deserialize bytes back to a public key."""
        return serialization.load_pem_public_key(serialized_key)
    
    @staticmethod
    def sign_data(private_key, data: str) -> bytes:
        """Create a digital signature for data using a private key."""
        signature = private_key.sign(
            data.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    @staticmethod
    def verify_signature(public_key, data: str, signature: bytes) -> bool:
        """Verify a digital signature using a public key."""
        try:
            public_key.verify(
                signature,
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False


class ServerSocket:
    """Server socket for handling network communication."""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """Initialize the server socket."""
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.thread = None
        self.clients = []
        self.queue = queue.Queue()
        
    def start(self):
        """Start the server in a separate thread."""
        if self.running:
            return
            
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.socket.settimeout(0.5)  # Short timeout for checking stop flag
            self.running = True
            
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()
            
            print(f"Server started on {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Server start error: {e}")
            self.socket.close()
            self.socket = None
            return False
    
    def _run(self):
        """Main server loop running in a thread."""
        while self.running:
            try:
                # Accept new connections
                client_socket, addr = self.socket.accept()
                print(f"New connection from {addr}")
                
                # Start a thread to handle this client
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr)
                )
                client_thread.daemon = True
                client_thread.start()
                self.clients.append((client_socket, addr, client_thread))
            
            except socket.timeout:
                # This is expected due to the timeout we set
                continue
            except Exception as e:
                if self.running:  # Only print if not deliberately stopping
                    print(f"Server error: {e}")
                break
        
        # Clean up when stopping
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def _handle_client(self, client_socket, addr):
        """Handle communication with a connected client."""
        client_socket.settimeout(TIMEOUT)
        
        try:
            while self.running:
                # Receive data from the client
                data = NetworkProtocol.receive_data(client_socket)
                if data is None:
                    break
                
                # Put the received data in the queue for processing
                self.queue.put((addr, data))
        except socket.timeout:
            print(f"Client {addr} timed out")
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()
            # Remove this client from our list
            self.clients = [(s, a, t) for s, a, t in self.clients if a != addr]
            print(f"Connection from {addr} closed")
    
    def stop(self):
        """Stop the server."""
        self.running = False
        
        # Close all client connections
        for client_socket, _, _ in self.clients:
            try:
                client_socket.close()
            except:
                pass
        
        # Wait for the server thread to end
        if self.thread and self.thread.is_alive():
            self.thread.join(2.0)  # Wait up to 2 seconds
        
        print("Server stopped")
    
    def send_to_all(self, data):
        """Send data to all connected clients."""
        for client_socket, addr, _ in self.clients:
            try:
                NetworkProtocol.send_data(client_socket, data)
            except Exception as e:
                print(f"Error sending to {addr}: {e}")
    
    def get_next_message(self, block=False, timeout=None):
        """Get the next message from the queue."""
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None


class ClientSocket:
    """Client socket for network communication."""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """Initialize the client socket."""
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.thread = None
        self.running = False
        self.queue = queue.Queue()
    
    def connect(self):
        """Connect to the server."""
        if self.connected:
            return True
            
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(TIMEOUT)
        
        try:
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.running = True
            
            # Start thread to receive messages
            self.thread = threading.Thread(target=self._receive_loop)
            self.thread.daemon = True
            self.thread.start()
            
            print(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            self.socket.close()
            self.socket = None
            return False
    
    def _receive_loop(self):
        """Background thread to receive messages."""
        self.socket.settimeout(0.5)  # Short timeout for checking stop flag
        
        while self.running:
            try:
                # Receive data from the server
                data = NetworkProtocol.receive_data(self.socket)
                if data is None:
                    break
                
                # Put received data in the queue
                self.queue.put(data)
            except socket.timeout:
                # This is expected due to the timeout we set
                continue
            except Exception as e:
                if self.running:  # Only print if not deliberately disconnecting
                    print(f"Receive error: {e}")
                break
        
        # If we exited the loop but should be running, connection was lost
        if self.running:
            print("Connection to server lost")
            self.connected = False
            self.socket.close()
            self.socket = None
    
    def disconnect(self):
        """Disconnect from the server."""
        self.running = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        if self.thread and self.thread.is_alive():
            self.thread.join(2.0)  # Wait up to 2 seconds
        
        self.connected = False
        self.socket = None
        print("Disconnected from server")
    
    def send(self, data):
        """Send data to the server."""
        if not self.connected:
            print("Not connected to server")
            return False
        
        return NetworkProtocol.send_data(self.socket, data)
    
    def get_next_message(self, block=False, timeout=None):
        """Get the next message from the queue."""
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None


class Sender:
    """Enhanced module to read data, create a Merkle tree, and send data securely."""
    
    def __init__(self, csv_file, host=DEFAULT_HOST, port=DEFAULT_PORT):
        """Initialize the sender with a CSV file and network parameters."""
        self.csv_file = csv_file
        self.data = None
        self.merkle_tree = None
        self.shuffled_data = None
        self.position_to_id_map = None
        self.client = ClientSocket(host, port)
        
        # Generate cryptographic keys
        self.private_key, self.public_key = CryptoTools.generate_key_pair()
        self.serialized_public_key = CryptoTools.serialize_public_key(self.public_key)
        
    def read_data(self):
        """Read data from the CSV file."""
        # Handle file not found or other IO errors
        try:
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
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Error reading CSV file: {e}")
            
            # Create a sample dataset for demonstration if file doesn't exist
            if isinstance(e, FileNotFoundError):
                print("Creating sample data instead")
                sample_data = {
                    'id': list(range(1, 11)),
                    'measurement': np.random.normal(50, 10, 10).tolist()
                }
                self.data = pd.DataFrame(sample_data)
                
                # Create list of tuples
                data_tuples = [(int(row.id), float(row.measurement)) for _, row in self.data.iterrows()]
                self.merkle_tree = MerkleTree(data_tuples)
                return self.data
            
            # For other errors, re-raise
            raise
    
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
    
    def connect_to_receiver(self):
        """Connect to the receiver server."""
        return self.client.connect()
    
    def send_data(self, use_network=True):
        """Send the shuffled data to the receiver."""
        if self.shuffled_data is None:
            raise ValueError("No shuffled data to send. Call shuffle_data() first.")
        
        # Prepare the data for sending
        sent_data = []
        for _, row in self.shuffled_data.iterrows():
            shuffled_id = int(row['shuffled_id'])
            measurement = float(row['measurement'])
            sent_data.append((shuffled_id, measurement))
        
        # Create a digital signature of the Merkle root
        merkle_root = self.get_merkle_root()
        signature = CryptoTools.sign_data(self.private_key, merkle_root)
        
        # Create a message to send
        message = {
            'type': 'data',
            'data': sent_data,
            'merkle_root': merkle_root,
            'signature': signature.hex(),  # Convert bytes to hex string for JSON
            'public_key': self.serialized_public_key.decode('utf-8')  # PEM is already base64, convert to string
        }
        
        if use_network:
            # Send over the network
            if not self.client.connected:
                print("Not connected to receiver. Attempting to connect...")
                if not self.connect_to_receiver():
                    print("Failed to connect. Cannot send data.")
                    return sent_data, merkle_root, signature
            
            print("Sending data over network...")
            success = self.client.send(message)
            if success:
                print(f"Sent {len(sent_data)} data points and Merkle root")
            else:
                print("Failed to send data over network")
        else:
            # Print locally
            print("Simulating data sending (network disabled):")
            print("Shuffled ID\tMeasurement")
            for shuffled_id, measurement in sent_data:
                print(f"{shuffled_id}\t{measurement:.2f}")
                # Simulate network delay
                time.sleep(0.001)
            
            print(f"\nSending Merkle root: {merkle_root[:16]}...")
        
        return sent_data, merkle_root, signature
    
    def visualize_merkle_tree(self, filename=None):
        """Visualize the Merkle tree."""
        if self.merkle_tree is None:
            raise ValueError("No Merkle tree created. Call read_data() first.")
        
        self.merkle_tree.visualize(filename)
    
    def disconnect(self):
        """Disconnect from the receiver."""
        if self.client.connected:
            self.client.disconnect()


class Receiver:
    """Enhanced module to receive data, unshuffle it, and verify integrity with Merkle tree."""
    
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        """Initialize the receiver with network parameters."""
        self.server = ServerSocket(host, port)
        self.received_data = {}
        self.restored_data = []
        self.merkle_root = None
        self.signature = None
        self.sender_public_key = None
        self.verification_result = None
    
    def start_server(self):
        """Start the server to receive data."""
        return self.server.start()
    
    def wait_for_data(self, timeout=30):
        """Wait for data from the sender."""
        print(f"Waiting for data (timeout: {timeout} seconds)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            message = self.server.get_next_message(block=True, timeout=1.0)
            if message:
                sender_addr, data = message
                print(f"Received message from {sender_addr}")
                
                if isinstance(data, dict) and data.get('type') == 'data':
                    # Extract and process the data
                    self.receive_data(
                        data['data'], 
                        data['merkle_root'],
                        bytes.fromhex(data['signature']),  # Convert hex string back to bytes
                        data['public_key'].encode('utf-8')  # Convert string back to bytes
                    )
                    return True
            
            # Check if we should continue waiting
            elapsed = time.time() - start_time
            if elapsed > timeout / 2 and not self.server.clients:
                print("No clients connected. Still waiting...")
        
        print("Timeout waiting for data")
        return False
    
    def receive_data(self, data, merkle_root, signature, public_key_pem):
        """Process received data."""
        print("\nReceiving data...")
        
        # Store the received data in a dictionary keyed by shuffled ID
        for shuffled_id, measurement in data:
            self.received_data[shuffled_id] = measurement
            
        # Store the received Merkle root and signature
        self.merkle_root = merkle_root
        self.signature = signature
        self.sender_public_key = CryptoTools.deserialize_public_key(public_key_pem)
        
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
        """Verify the integrity of the received data using the Merkle root and signature."""
        if not self.restored_data or not self.merkle_root:
            raise ValueError("No data or Merkle root. Call receive_data() and unshuffle_data() first.")
        
        # First verify the signature
        signature_valid = False
        if self.signature and self.sender_public_key:
            signature_valid = CryptoTools.verify_signature(
                self.sender_public_key, 
                self.merkle_root, 
                self.signature
            )
            print("\nVerifying digital signature...")
            print(f"Signature verification: {'Passed' if signature_valid else 'Failed'}")
        else:
            print("\nNo signature or public key provided, skipping signature verification")
        
        # Create data tuples in the same format as the sender
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
        merkle_valid = (new_root == self.merkle_root)
        
        print("\nVerifying data integrity...")
        print(f"Original Merkle root: {self.merkle_root[:16]}...")
        print(f"Computed Merkle root: {new_root[:16]}...")
        print(f"Data integrity verification: {'Passed' if merkle_valid else 'Failed'}")
        
        # Store and return the final verification result
        self.verification_result = merkle_valid and (not self.signature or signature_valid)
        
        if self.verification_result:
            print("\n✓ SUCCESS: Data integrity verified!")
        else:
            print("\n✗ ERROR: Data integrity verification failed!")
            if not merkle_valid:
                print("  - Merkle root mismatch: Data may have been tampered with")
            if self.signature and not signature_valid:
                print("  - Invalid signature: Data may not be from the expected sender")
        
        # Visualize the Merkle tree we constructed
        print("\nGenerating Merkle Tree visualization...")
        new_merkle_tree.visualize("receiver_merkle_tree.png")
        
        return self.verification_result
    
    def visualize_restored_data(self, filename="restored_data.png"):
        """Visualize the restored data as a scatter plot."""
        if not self.restored_data:
            raise ValueError("No restored data. Call unshuffle_data() first.")
        
        # Extract IDs and measurements
        ids = [item[0] for item in self.restored_data]
        measurements = [item[1] for item in self.restored_data]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(ids, measurements, c='blue', marker='o')
        plt.plot(ids, measurements, 'b--', alpha=0.5)
        plt.title('Restored Data Measurements')
        plt.xlabel('ID')
        plt.ylabel('Measurement Value')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        
        print(f"Data visualization saved to {filename}")
    
    def save_data_csv(self, filename="restored_data.csv"):
        """Save the restored data to a CSV file."""
        if not self.restored_data:
            raise ValueError("No restored data. Call unshuffle_data() first.")
        
        # Create a DataFrame from the restored data
        df = pd.DataFrame(self.restored_data, columns=['id', 'measurement'])
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Restored data saved to {filename}")
    
    def stop_server(self):
        """Stop the server."""
        self.server.stop()


def create_sample_csv(filename="measurements.csv", num_samples=50):
    """Create a sample CSV file with random measurements."""
    # Set seed for reproducibility
    np.random.seed(123)
    
    # Generate random data
    data = {
        'id': list(range(1, num_samples + 1)),
        'measurement': np.random.normal(50, 15, num_samples).tolist()
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Sample CSV created: {filename} with {num_samples} measurements")
    
    return filename


def run_network_demo():
    """Demonstrate the full system with network communication."""
    print("=== Enhanced Data Integrity Verification Demo ===\n")
    
    # Check if measurements.csv exists, create it if not
    try:
        with open("measurements.csv", "r") as f:
            pass
    except FileNotFoundError:
        create_sample_csv("measurements.csv")
    
    # Start receiver in a thread
    receiver = Receiver()
    receiver_thread = threading.Thread(target=receiver.start_server)
    receiver_thread.daemon = True
    receiver_thread.start()
    
    # Wait for server to initialize
    time.sleep(1)
    
    # Initialize sender
    sender = Sender("measurements.csv")
    
    try:
        # 1. Sender reads and prepares data
        print("Sender reading and preparing data...")
        sender.read_data()
        sender.shuffle_data()
        
        # Visualize the Merkle tree
        sender.visualize_merkle_tree("sender_merkle_tree.png")
        
        # 2. Sender connects to receiver
        print("\nSender connecting to receiver...")
        if not sender.connect_to_receiver():
            print("Failed to connect to receiver. Aborting.")
            return False
        
        # 3. Sender sends data
        print("\nSender transmitting data...")
        sent_data, merkle_root, signature = sender.send_data()
        
        # 4. Receiver waits for and processes data
        success = receiver.wait_for_data()
        if not success:
            print("Receiver did not receive data. Aborting.")
            return False
        
        # 5. Receiver unshuffles data
        receiver.unshuffle_data()
        
        # 6. Receiver verifies data integrity
        integrity_verified = receiver.verify_integrity()
        
        # 7. Receiver visualizes restored data
        receiver.visualize_restored_data()
        
        # 8. Receiver saves restored data
        receiver.save_data_csv()
        
        return integrity_verified
    finally:
        # Clean up
        print("\nCleaning up...")
        sender.disconnect()
        receiver.stop_server()


def run_tampered_network_demo():
    """Demonstrate the system with deliberate tampering."""
    print("=== Enhanced Data Integrity Verification with Tampering Demo ===\n")
    
    # Check if measurements.csv exists, create it if not
    try:
        with open("measurements.csv", "r") as f:
            pass
    except FileNotFoundError:
        create_sample_csv("measurements.csv")
    
    # Start receiver in a thread
    receiver = Receiver()
    receiver_thread = threading.Thread(target=receiver.start_server)
    receiver_thread.daemon = True
    receiver_thread.start()
    
    # Wait for server to initialize
    time.sleep(1)
    
    # Initialize sender
    sender = Sender("measurements.csv")
    
    try:
        # 1. Sender reads and prepares data
        print("Sender reading and preparing data...")
        sender.read_data()
        sender.shuffle_data()
        
        # 2. Sender connects to receiver
        print("\nSender connecting to receiver...")
        if not sender.connect_to_receiver():
            print("Failed to connect to receiver. Aborting.")
            return False
        
        # 3. Sender sends data (we'll tamper with it instead of sending directly)
        print("\nPreparing data for transmission...")
        sent_data, merkle_root, signature = sender.send_data(use_network=False)
        
        # 4. Tamper with the data
        if sent_data:
            tampered_index = np.random.randint(0, len(sent_data))
            original_value = sent_data[tampered_index][1]
            sent_data[tampered_index] = (sent_data[tampered_index][0], original_value + 5.0)
            
            print(f"\n! Simulating tampering: Modified measurement at index {tampered_index}")
            print(f"  Original value: {original_value:.2f}")
            print(f"  Tampered value: {sent_data[tampered_index][1]:.2f}")
        
        # 5. Manually create and send the message
        message = {
            'type': 'data',
            'data': sent_data,
            'merkle_root': merkle_root,
            'signature': signature.hex(),  # Convert bytes to hex string for JSON
            'public_key': CryptoTools.serialize_public_key(sender.public_key).decode('utf-8')
        }
        sender.client.send(message)
        
        # 6. Receiver waits for and processes data
        success = receiver.wait_for_data()
        if not success:
            print("Receiver did not receive data. Aborting.")
            return False
        
        # 7. Receiver unshuffles data
        receiver.unshuffle_data()
        
        # 8. Receiver verifies data integrity (which should fail due to tampering)
        integrity_verified = receiver.verify_integrity()
        
        # 9. Receiver visualizes restored data
        receiver.visualize_restored_data("tampered_data.png")
        
        # This should be False due to tampering
        return integrity_verified
    finally:
        # Clean up
        print("\nCleaning up...")
        sender.disconnect()
        receiver.stop_server()


def run_local_demo():
    """Demonstrate the system without network communication for simpler testing."""
    print("=== Simple Data Integrity Verification Demo (Local) ===\n")
    
    # Check if measurements.csv exists, create it if not
    try:
        with open("measurements.csv", "r") as f:
            pass
    except FileNotFoundError:
        create_sample_csv("measurements.csv")
    
    # 1. Sender reads and shuffles data
    sender = Sender("measurements.csv")
    sender.read_data()
    sender.shuffle_data()
    
    try:
        # Visualize the Merkle tree (wrapped in try-except to continue if visualization fails)
        sender.visualize_merkle_tree("sender_merkle_tree.png")
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing with the demo without visualization...")
    
    # 2. Sender "sends" data (local simulation)
    sent_data, merkle_root, signature = sender.send_data(use_network=False)
    
    # 3. Receiver gets data
    receiver = Receiver()
    receiver.receive_data(
        sent_data, 
        merkle_root, 
        signature,
        CryptoTools.serialize_public_key(sender.public_key)
    )
    
    # 4. Receiver unshuffles data
    receiver.unshuffle_data()
    
    # 5. Receiver verifies data integrity
    integrity_verified = receiver.verify_integrity()
    
    try:
        # 6. Receiver visualizes restored data (wrapped in try-except)
        receiver.visualize_restored_data()
    except Exception as e:
        print(f"Data visualization error: {e}")
        print("Continuing with the demo without data visualization...")
    
    # 7. Report the overall result
    if integrity_verified:
        print("\n✓ Success: Data transmission and reshuffling completed without any tampering!")
    else:
        print("\n✗ Error: Data integrity verification failed. Possible tampering detected!")
    
    return integrity_verified


def run_local_demo_with_tampering():
    """Demonstrate the system with deliberate tampering."""
    print("=== Data Integrity Verification with Tampering Demo (Local) ===\n")
    
    # Check if measurements.csv exists, create it if not
    try:
        with open("measurements.csv", "r") as f:
            pass
    except FileNotFoundError:
        create_sample_csv("measurements.csv")
    
    # 1. Sender reads and shuffles data
    sender = Sender("measurements.csv")
    sender.read_data()
    sender.shuffle_data()
    
    # 2. Sender "sends" data (local simulation)
    sent_data, merkle_root, signature = sender.send_data(use_network=False)
    
    # 3. Simulate tampering by modifying one measurement
    if sent_data:
        tampered_index = np.random.randint(0, len(sent_data))
        original_value = sent_data[tampered_index][1]
        sent_data[tampered_index] = (sent_data[tampered_index][0], original_value * 1.2)
        
        print(f"\n! Simulating tampering: Modified measurement at index {tampered_index}")
        print(f"  Original value: {original_value:.2f}")
        print(f"  Tampered value: {sent_data[tampered_index][1]:.2f}")
    
    # 4. Receiver gets data
    receiver = Receiver()
    receiver.receive_data(
        sent_data, 
        merkle_root, 
        signature,
        CryptoTools.serialize_public_key(sender.public_key)
    )
    
    # 5. Receiver unshuffles data
    receiver.unshuffle_data()
    
    # 6. Receiver verifies data integrity (should fail due to tampering)
    integrity_verified = receiver.verify_integrity()
    
    # 7. Receiver visualizes restored data
    receiver.visualize_restored_data("tampered_data.png")
    
    # 8. Report the overall result
    if integrity_verified:
        print("\n✗ Problem: Tampering was not detected!")
    else:
        print("\n✓ Success: Tampering was correctly detected by Merkle tree verification!")
    
    return integrity_verified


def run_advanced_analysis():
    """Run advanced analysis on the Merkle tree structure and performance."""
    print("=== Advanced Merkle Tree Analysis ===\n")
    
    # Create sample data of different sizes
    sizes = [10, 50, 100, 500, 1000]
    creation_times = []
    verification_times = []
    
    for size in sizes:
        print(f"\nAnalyzing performance with {size} data points...")
        
        # Create random data
        np.random.seed(size)  # Different seed for each size
        data = [(i, float(np.random.normal(50, 15))) for i in range(1, size+1)]
        
        # Measure tree creation time - repeat multiple times for accuracy
        num_repeats = 5
        creation_time_total = 0
        for _ in range(num_repeats):
            start_time = time.time()
            tree = MerkleTree(data)
            end_time = time.time()
            creation_time_total += (end_time - start_time)
        
        # Average time
        tree_time = creation_time_total / num_repeats
        creation_times.append(tree_time)
        
        # Get tree metrics
        tree_depth = len(tree.levels)  # Use our new level-based implementation
        total_nodes = sum(len(level) for level in tree.levels)
        
        print(f"Tree creation time: {tree_time:.6f} seconds")
        print(f"Tree depth: {tree_depth}")
        print(f"Number of nodes: {total_nodes}")
        
        # Select a random leaf for proof generation
        leaf_index = np.random.randint(0, len(data))
        
        # Suppress detailed logging for cleaner output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Measure proof verification time - repeat more times for better accuracy
        verify_repeats = 50  # More repeats for more accurate timing of fast operations
        verify_time_total = 0
        for _ in range(verify_repeats):
            # Generate new proof each time
            start_time = time.time()
            proof = tree.generate_proof(leaf_index)
            leaf_hash = tree.get_leaf_hash(leaf_index)
            verification = tree.verify_proof(leaf_hash, proof)
            end_time = time.time()
            verify_time_total += (end_time - start_time)
        
        # Restore stdout and get the debug output
        debug_output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Calculate average verification time
        verify_time = verify_time_total / verify_repeats
        verification_times.append(verify_time)
        
        print(f"Proof verification time: {verify_time:.6f} seconds (averaged over {verify_repeats} runs)")
        print(f"Proof length: {len(proof)} elements")
        print(f"Verification result: {'Passed' if verification else 'Failed'}")
        
        # Debug output if verification fails
        if not verification:
            print(f"  - Check debug output for details")
            # Print a small part of the debug output
            print(debug_output.split("\n")[:10])
            print("...")
        
        # Only visualize small trees
        if size <= 50:
            try:
                tree.visualize(f"merkle_tree_{size}.png")
            except Exception as e:
                print(f"  - Visualization error: {e}")
    
    # Plot performance results
    try:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(sizes, creation_times, 'o-', color='blue')
        plt.title('Merkle Tree Creation Time')
        plt.xlabel('Number of Data Points')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(sizes, verification_times, 'o-', color='green')
        plt.title('Proof Verification Time')
        plt.xlabel('Number of Data Points')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        # Log scale might be better for small values
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig("merkle_performance.png")
        plt.show()
        
        # Also create a table of results
        print("\nPerformance Results Summary:")
        print("-" * 80)
        print(f"{'Size':<10} | {'Creation Time (s)':<20} | {'Verification Time (s)':<20} | {'Depth':<10}")
        print("-" * 80)
        for i, size in enumerate(sizes):
            print(f"{size:<10} | {creation_times[i]:<20.6f} | {verification_times[i]:<20.6f} | {int(np.log2(size))+1:<10}")
    
    except Exception as e:
        print(f"Error creating performance plots: {e}")
    
    print("\nPerformance analysis completed!")


if __name__ == "__main__":
    # Choose which demo to run
    print("Merkle Tree Authentication System Demos")
    print("--------------------------------------")
    print("1. Local Demo (Without Network)")
    print("2. Local Demo with Tampering")
    print("3. Network Demo")
    print("4. Network Demo with Tampering")
    print("5. Advanced Analysis")
    print("6. Create Sample Data")
    
    choice = input("\nEnter choice (1-6): ")
    
    if choice == '1':
        run_local_demo()
    elif choice == '2':
        run_local_demo_with_tampering()
    elif choice == '3':
        run_network_demo()
    elif choice == '4':
        run_tampered_network_demo()
    elif choice == '5':
        run_advanced_analysis()
    elif choice == '6':
        num_samples = int(input("Enter number of samples: "))
        create_sample_csv("measurements.csv", num_samples)
    else:
        print("Invalid choice. Exiting.")