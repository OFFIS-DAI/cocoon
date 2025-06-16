# Channel models 

## LTE network configuration

Server: 2 ms → 0.5 ms (application server processing)
Router: 5 ms → 0.01 ms (10 microseconds - typical router forwarding)
PGW: 10 ms → 0.05 ms (50 microseconds - core network gateway)
eNB0: 8 ms → 0.5 ms (base station processing including scheduling)
All UE nodes (node0-99): 0 ms → 2 ms (mobile device processing)

Infrastructure nodes (router, PGW) now have microsecond-level delays, which is realistic for high-performance network equipment
eNB0 has moderate delay (0.5 ms) to account for radio resource scheduling and protocol processing
UE nodes now have 2 ms processing delay, which represents realistic mobile device processing including protocol stack overhead and application response time
Server has reduced delay (0.5 ms) appropriate for a well-configured application server


## LTE450 network configuration

Server: 1.0 ms (slightly higher due to potentially rural/remote server infrastructure)
Router: 0.02 ms (20 microseconds - similar to standard but accounting for potential older equipment)
PGW: 0.1 ms (100 microseconds - slightly higher for rural core network)
eNB0: 2.0 ms (significantly higher than standard LTE due to 450 MHz processing complexity)
UE nodes: 5 ms (higher than standard LTE due to device characteristics)

Backhaul bandwidth: Reduced to 1 Gbps and 100 Mbps (more realistic for rural deployments)
Air interface: Reduced to 10 Mbps per UE (typical for LTE450 rural coverage scenarios)

## 5G

Key 5G Processing Delays:

Server: 0.2 ms (optimized edge/cloud server for 5G applications)
Router: 0.005 ms (5 microseconds - high-performance networking equipment)
UPF (User Plane Function): 0.02 ms (20 microseconds - 5G core network function)
iUPF (Intermediate UPF): 0.01 ms (10 microseconds - edge UPF for lower latency)
gNB0 (5G NodeB): 0.1 ms (100 microseconds - advanced 5G base station)
bgCell (Background Cell): 0.05 ms (50 microseconds - interference modeling)
NRUe nodes (5G devices): 0.5 ms (modern 5G-capable devices)

5G Network Characteristics Reflected:
Ultra-Low Latency: 5G targets <1ms end-to-end latency for critical applications, so all processing delays are significantly reduced compared to LTE.
Enhanced Infrastructure:

100 Gbps core network links (vs 10 Gbps in LTE)
10 Gbps backhaul to gNB (vs 100 Mbps-1 Gbps in LTE)
1 Gbps air interface per UE (vs 10-100 Mbps in LTE)

Advanced Processing:

Edge computing integration (iUPF for local processing)
Network slicing capabilities (reduced UPF delays)
Massive MIMO and beamforming (efficient gNB processing)

5G vs LTE/LTE450 Comparison:

5G UE: 0.5 ms vs LTE: 2 ms vs LTE450: 5 ms
5G Base Station: 0.1 ms vs LTE eNB: 0.5 ms vs LTE450 eNB: 2 ms
5G Core: 0.01-0.02 ms vs LTE Core: 0.05-0.1 ms vs LTE450: 0.1 ms

# Static Delay Graphs
