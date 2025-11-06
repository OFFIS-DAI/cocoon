import re
import collections
from typing import Dict, Any, List, Set, Tuple, Optional


def _parse_rate_bps(channel_name: str) -> Optional[int]:
    """
    Interpretiert Kanalnamen wie 'C100kbps', 'C1Mbps', 'C10Gbps', 'Eth10G', 'Eth100M'.
    Fällt sonst auf sinnvolle Defaults zurück (None -> Caller setzt Default).
    """
    m = re.search(r'(\d+(?:\.\d+)?)\s*(k|M|G)?bps', channel_name, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or '').upper()
        mult = {'': 1, 'K': 1e3, 'M': 1e6, 'G': 1e9}[unit]
        return int(val * mult)

    m = re.search(r'Eth\s*(\d+)\s*([GM])', channel_name, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        unit = m.group(2).upper()
        return int(val * (1e9 if unit == 'G' else 1e6))

    m = re.search(r'Eth\s*(\d+)\s*M', channel_name, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        return int(val * 1e6)

    return None


def parse_ned_to_channel_topology(ned_text: str,
                                  default_processing_delay_ms: float = 1.0,
                                  default_propagation_speed_mps: int = 300_000_000,
                                  default_transmission_rate_bps: int = 100_000_000
                                  ) -> Dict[str, Any]:
    # --- 1. Parse nodes with positions ---
    node_pattern = re.compile(
        r'\b([A-Za-z_]\w*)\s*:\s*([A-Za-z_][\w\.]*)\s*\{[^{}]*?@display\("p=\s*([^,"]+)\s*,\s*([^"]+)"\);[^{}]*?\}',
        re.DOTALL)
    nodes: Dict[str, Dict[str, Any]] = {}
    node_types: Dict[str, str] = {}
    for m in node_pattern.finditer(ned_text):
        name, typ, x, y = m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
        try:
            xf, yf = float(x), float(y)
        except ValueError:
            continue
        nodes[name] = {
            "node_id": name,
            "position": [xf, yf],
            "processing_delay_ms": float(default_processing_delay_ms),
        }
        node_types[name] = typ

    # --- 2. Parse all connections (x <--> conn <--> y) ---
    conn_pattern = re.compile(r'(\S+)\s*<-->\s*(\S+)\s*<-->\s*(\S+)', re.MULTILINE)
    edges: List[Tuple[str, str, str]] = [
        (m.group(1).split('.')[0], m.group(3).split('.')[0], m.group(2))
        for m in conn_pattern.finditer(ned_text)
    ]

    # --- 3. Build networks from explicit channels ---
    net_members: Dict[str, Set[str]] = collections.defaultdict(set)
    for a, b, ch in edges:
        net_members[ch].update([a, b])

    networks: List[Dict[str, Any]] = []
    for net_id, members in net_members.items():
        rate = _parse_rate_bps(net_id) or int(default_transmission_rate_bps)
        networks.append({
            "network_id": net_id,
            "transmission_rate_bps": rate,
            "propagation_speed_mps": int(default_propagation_speed_mps),
        })

    node2nets = collections.defaultdict(list)
    for net_id, members in net_members.items():
        for n in members:
            node2nets[n].append(net_id)

    # --- 4. Detect system type (5G/LTE) ---
    is_5g = "5G" in ned_text or "gNB" in ned_text
    if is_5g:
        wireless_rate = 500_000_000
        default_wireless_net_id = "NAN5G"
    else:
        wireless_rate = 100_000_000
        default_wireless_net_id = "NANLTE"

    default_wireless_net = {
        "network_id": default_wireless_net_id,
        "transmission_rate_bps": wireless_rate,
        "propagation_speed_mps": 3e8,
    }

    # --- 5. Detect antennas and UEs ---
    antennas = [n for n, t in node_types.items() if t.lower().startswith(("gnodeb", "enodeb"))]
    ues = [n for n, t in node_types.items() if t.lower().startswith(("nrue", "ue"))]

    # --- 6. Add missing UE–antenna links + multi-homing ---
    added_wireless = False
    for ue in ues:
        if any(ue in (a, b) for a, b, _ in edges):
            continue
        if antennas:
            antenna = antennas[0]
            edges.append((ue, antenna, default_wireless_net_id))
            net_members[default_wireless_net_id].update([ue, antenna])
            node2nets[ue].append(default_wireless_net_id)
            node2nets[antenna].append(default_wireless_net_id)
            added_wireless = True

    # --- 7. Ensure default wireless network exists ---
    if added_wireless and all(net["network_id"] != default_wireless_net_id for net in networks):
        networks.append(default_wireless_net)

    # --- 8. Build node entries ---
    node_entries: List[Dict[str, Any]] = []
    for nid, attrs in nodes.items():
        nets = sorted(set(node2nets.get(nid, [])))
        if len(nets) == 1:
            attrs["network"] = nets[0]
        elif len(nets) > 1:
            attrs["networks"] = nets
        node_entries.append(attrs)

    return {"nodes": node_entries, "networks": networks}
