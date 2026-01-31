## Channel models 
The Channel Models are generated from detailed OMNeT++ Simulations

1. Extract topology:
Parse the NED file to get nodes, positions, and connections.
Auto-add wireless links (NAN5G / NANLTE) between UEs and antennas.

2. Fit parameters from detailed results: Transmission rate → slope of delay vs. message size.
Processing delay → constant offset. Jitter → delay variance. Propagation speed → 2×10⁸ m/s (wired), 3×10⁸ m/s (wireless).

3. Build and validate: Use these in your ChannelModelScheduler.

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