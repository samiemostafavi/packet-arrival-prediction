import matplotlib.pyplot as plt
import sys

from edaf.core.uplink.analyze_packet import ULPacketAnalyzer


packet_analyzer = ULPacketAnalyzer("data/240928_082545_results/database.db")
UE_PACKET_INSERTIONS = 1000
uids_arr = list(range(packet_analyzer.first_ueipid, packet_analyzer.first_ueipid + UE_PACKET_INSERTIONS))
packets_dict = packet_analyzer.figure_packettx_from_ueipids(uids_arr)
print(packets_dict)