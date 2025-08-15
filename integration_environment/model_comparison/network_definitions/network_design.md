# Channel Models 

## 5G Network Configuration

**Updated Processing Delays (Realistic):**
- Server: 1.0 ms (application server processing)
- Router: 2.0 ms (high-performance networking equipment with protocol processing)
- UPF (User Plane Function): 5.0 ms (5G core network function with full protocol stack)
- iUPF (Intermediate UPF): 5.0 ms (edge UPF for local processing)
- gNB0 (5G NodeB): 7.0 ms (advanced 5G base station with MAC scheduling)
- bgCell (Background Cell): 7.0 ms (interference modeling and processing)
- UE nodes (node0-99): 15.0 ms (modern 5G-capable devices with full protocol stack)

**Network Characteristics:**
- Core network links: 100 Gbps (server↔router, router↔upf, upf↔iUpf)
- Backhaul to gNB: 10 Gbps (iUpf↔gNB0)
- Air interface per UE: 1 Gbps (gNB0↔nodes)
- Propagation speed: 200 Mbps (fiber), 300 Mbps (wireless)

---

## LTE Network Configuration

**Processing Delays:**
- Server: 1.0 ms (application server processing)
- Router: 2.0 ms (networking equipment with protocol processing)
- PGW (Packet Gateway): 5.0 ms (LTE core network gateway)
- eNB0 (LTE Base Station): 7.0 ms (base station processing including MAC scheduling)
- UE nodes (node0-99): 15.0 ms (mobile device processing with full LTE protocol stack)

**Network Characteristics:**
- Core network links: 10 Gbps (server↔router, router↔pgw, pgw↔eNB0)
- Air interface per UE: 100 Mbps (eNB0↔nodes)
- Propagation speed: 200 Mbps (fiber), 300 Mbps (wireless)

**Comparison to 5G:**
- Same processing delays but lower bandwidth capabilities
- 10x lower core network capacity (10 Gbps vs 100 Gbps)
- 10x lower air interface per UE (100 Mbps vs 1 Gbps)

---

## LTE450 Network Configuration  

**Processing Delays:**
- Server: 1.0 ms (slightly higher due to potentially rural/remote server infrastructure)
- Router: 2.0 ms (similar to standard but accounting for potential older equipment)
- PGW: 5.0 ms (core network gateway for rural deployment)
- eNB0: 7.0 ms (LTE450 base station - same processing as regular LTE)
- UE nodes: 15.0 ms (same as LTE - device processing characteristics)

**Network Characteristics (Rural Deployment):**
- Core network links: 1 Gbps (server↔router, router↔pgw)
- Backhaul to eNB: 100 Mbps (pgw↔eNB0) 
- Air interface per UE: 10 Mbps (eNB0↔nodes - typical for LTE450 rural coverage)
- Propagation speed: 200 Mbps (fiber), 300 Mbps (wireless)

**Rural Deployment Characteristics:**
- 100x lower core network capacity vs 5G (1 Gbps vs 100 Gbps)
- 100x lower backhaul capacity (100 Mbps vs 10 Gbps)
- 100x lower air interface per UE (10 Mbps vs 1 Gbps)

---

## Ethernet Network Configuration

**Processing Delays:**
- Central Router: 0.001 ms (1 microsecond - high-performance switching)
- Access Routers (router_0, router_1, router_2): 0.001 ms (edge switching)
- End Nodes (node0-99): 0 ms (minimal processing for wired connections)

**Network Characteristics:**
- All links: 10 Mbps (typical enterprise Ethernet)
- Propagation speed: 200 Mbps (copper/fiber)
- Hierarchical topology: Central router with 3 access routers

**Key Differences:**
- Ultra-low processing delays (microseconds vs milliseconds)
- Wired reliability and consistency
- Lower bandwidth but predictable performance
- No wireless protocol overhead

---

## Technology Comparison Summary

| Technology | UE Processing | Base Station | Core Network | Air Interface | End-to-End Delay* |
|------------|---------------|--------------|--------------|---------------|-------------------|
| **5G**     | 15.0 ms      | 7.0 ms      | 5.0 ms       | 1 Gbps        | ~35-50 ms        |
| **LTE**    | 15.0 ms      | 7.0 ms      | 5.0 ms       | 100 Mbps      | ~35-50 ms        |
| **LTE450** | 15.0 ms      | 7.0 ms      | 5.0 ms       | 10 Mbps       | ~35-50 ms        |
| **Ethernet** | 0 ms        | 0.001 ms    | 0.001 ms     | 10 Mbps       | ~1-5 ms          |

*End-to-end delay includes processing, transmission, propagation, and protocol overhead with realistic jitter

---

## Implementation Notes

### Realistic Protocol Overhead
The channel models now include:
- **MAC scheduling delays**: 1-10ms for wireless technologies
- **Protocol stack processing**: RLC, PDCP, application layers
- **Message size scaling**: Larger messages incur additional delays
- **Jitter modeling**: ±30% random variation to simulate real-world conditions

### Expected Channel Model Results
With these configurations, channel models should now produce:
- **5G delays**: 30-60ms 
- **LTE delays**: 30-60ms
- **Ethernet delays**: 1-8ms
- **Standard deviation**: 20-40% of mean
- **Realistic scaling** with message size and network load

---

## Static Delay Graphs
The static delay graphs are generated as a post-processing step from the detailed communication simulation results. 
This process transforms the dynamic simulation data into simplified network models that capture the essential 
end-to-end delay characteristics between communicating nodes.

**Generation Process:**
1. Unicast message exchange is simulated between a set of 10 nodes for each network technology
2. The resulting delay times are extracted and analyzed
3. A graph with a set of 100 nodes is generated 
4. The end-to-end delay between two nodes is defined by the mean delay measured in the simulation between the nodes, or if there was no direct connection, the global mean delay time is used
5. The static graph provides a fast lookup table for delay estimation without full simulation overhead