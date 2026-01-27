from dataclasses import dataclass
from typing import Optional, List
import random

@dataclass
class Packet:
    src: str
    dst: str
    protocol: str # "modbus", "dnp3", "http"
    payload: dict
    timestamp: float
    is_malicious: bool = False

class NetworkSimulator:
    """
    Simulates the Industrial Network environment.
    Handles packet transmission, latency, and drop rates.
    Can be compromised by Red Agents.
    """
    def __init__(self, latency_mean: float = 0.05, drop_rate: float = 0.001):
        self.latency_mean = latency_mean
        self.drop_rate = drop_rate
        self.queue: List[Packet] = []
        self.compromised_nodes = set()
        
    def send(self, packet: Packet):
        """Enqueue a packet for delivery"""
        if random.random() > self.drop_rate:
            # Simple FIFO queue for now
            self.queue.append(packet)
            
    def process_queue(self, time: float) -> List[Packet]:
        """
        Return packets that have arrived.
        For simplicity in this discrete timestep sim, we return all queued packets
        assuming the step size > latency.
        """
        delivered = []
        while self.queue:
            pkt = self.queue.pop(0)
            
            # Red Team capability: Packet Injection / Modification
            if pkt.src in self.compromised_nodes:
                pkt.is_malicious = True # Simplified tagging for ground truth
                
            delivered.append(pkt)
        return delivered

    def compromise_node(self, node_id: str):
        self.compromised_nodes.add(node_id)
