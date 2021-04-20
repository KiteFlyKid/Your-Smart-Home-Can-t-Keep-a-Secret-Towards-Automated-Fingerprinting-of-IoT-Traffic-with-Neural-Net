#!/usr/bin/env python3

import argparse
import os.path
import sys
import pandas as pd
import pyshark
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, UDP, TCP
from tqdm import tqdm
#--------------------------------------------------

def render_csv_row(pkt_sh, pkt_sc):
    """Write one packet entry into the CSV file.
    pkt_sh is the PyShark representation of the packet

    pkt_sc is a 'bytes' representation of the packet as returned from
    scapy's RawPcapReader

    fh_csv is the csv file handle
    """
    ether_pkt_sc = Ether(pkt_sc)
    try:
        if ether_pkt_sc.type != 0x800:
        # print('Ignoring non-IP packet')
            return [pkt_sh.no,pkt_sh.time,pkt_sh.protocol,None,None,None,None,None,pkt_sh.length]
    except:
        return False

    ip_pkt_sc = ether_pkt_sc[IP]       # <<<< Assuming Ethernet + IPv4 here
    proto = ip_pkt_sc.fields['proto']
    if proto == 17:
        udp_pkt_sc = ip_pkt_sc[UDP]
        l4_payload_bytes = bytes(udp_pkt_sc.payload)
        l4_proto_name = 'UDP'
        l4_sport = udp_pkt_sc.sport
        l4_dport = udp_pkt_sc.dport
    elif proto == 6:
        tcp_pkt_sc = ip_pkt_sc[TCP]
        l4_payload_bytes = bytes(tcp_pkt_sc.payload)
        l4_proto_name = 'TCP'
        l4_sport = tcp_pkt_sc.sport
        l4_dport = tcp_pkt_sc.dport
    else:
        # Currently not handling packets that are not UDP or TCP
        # print('Ignoring non-UDP/TCP packet')
        return False

    # Each line of the CSV has this format
    fmt = '{0}|{1}|{2}({3})|{4}|{5}:{6}|{7}:{8}|{9}|{10}'
    #       |   |   |   |    |   |   |   |   |   |   |
    #       |   |   |   |    |   |   |   |   |   |   o-> {10} L4 payload hexdump
    #       |   |   |   |    |   |   |   |   |   o-----> {9}  total pkt length
    #       |   |   |   |    |   |   |   |   o---------> {8}  dst port
    #       |   |   |   |    |   |   |   o-------------> {7}  dst ip address
    #       |   |   |   |    |   |   o-----------------> {6}  src port
    #       |   |   |   |    |   o---------------------> {5}  src ip address
    #       |   |   |   |    o-------------------------> {4}  text description
    #       |   |   |   o------------------------------> {3}  L4 protocol
    #       |   |   o----------------------------------> {2}  highest protocol
    #       |   o--------------------------------------> {1}  time
    #       o------------------------------------------> {0}  frame number

    # Example:
    # 1|0.0|DNS(UDP)|Standard query 0xf3de A www.cisco.com|192.168.1.116:57922|1.1.1.1:53|73|f3de010000010000000000000377777705636973636f03636f6d0000010001
    return  [pkt_sh.no,               # {0}
                     pkt_sh.time,             # {1}
                     pkt_sh.protocol,         # {2}
                     l4_proto_name,           # {3}
                                 # {4}
                     pkt_sh.source,           # {5}
                     int(l4_sport),                # {6}
                     pkt_sh.destination,      # {7}
                     int(l4_dport),                # {8}
                     int(pkt_sh.length)]          # {9}] # {10}]


    # print(fmt.format(pkt_sh.no,               # {0}
    #                  pkt_sh.time,             # {1}
    #                  pkt_sh.protocol,         # {2}
    #                  l4_proto_name,           # {3}
    #                  pkt_sh.info,             # {4}
    #                  pkt_sh.source,           # {5}
    #                  l4_sport,                # {6}
    #                  pkt_sh.destination,      # {7}
    #                  l4_dport,                # {8}
    #                  pkt_sh.length,           # {9}
    #                  l4_payload_bytes.hex()), # {10}
    #       file=fh_csv)

    # return True
    #--------------------------------------------------

def pcap2csv(in_pcap):
    """Main entry function called from main to process the pcap and
    generate the csv file.

    in_pcap = name of the input pcap file (guaranteed to exist)
    out_csv = name of the output csv file (will be created)

    This function walks over each packet in the pcap file, and for
    each packet invokes the render_csv_row() function to write one row
    of the csv.
    """

    # Open the pcap file with PyShark in "summary-only" mode, since this
    # is the mode where the brief textual description of the packet (e.g.
    # "Standard query 0xf3de A www.cisco.com", "Client Hello" etc.) are
    # made available.
    print('-----Start extract {}----'.format(in_pcap))
    pcap_pyshark = pyshark.FileCapture(in_pcap, only_summaries=True)
    pcap_pyshark.load_packets()
    pcap_pyshark.reset()
    result_columns=['frame number','time','highest protocal','L4 Protocal','src ip','src port','dst ip','dst port','pkt len']
    results=[]
    frame_num = 0
    ignored_packets = 0
        # Open the pcap file with scapy's RawPcapReader, and iterate over
        # each packet. In each iteration get the PyShark packet as well,
        # and then call render_csv_row() with both representations to generate
        # the CSV row.
    for (pkt_scapy, _) in RawPcapReader(in_pcap):
            try:
                pkt_pyshark = pcap_pyshark.next_packet()
                frame_num += 1
                result_row=render_csv_row(pkt_pyshark, pkt_scapy)
                if not result_row:
                    ignored_packets += 1
                else:
                    results.append(result_row)
            except StopIteration:
                # Shouldn't happen because the RawPcapReader iterator should also
                # exit before this happens.
                break
    pd.DataFrame(results,columns=result_columns).to_csv('scapy_output/{}.csv'.format(in_pcap.split('.')[0].split('/')[-1]),index=False)
    print('{} packets read, {} packets not written to CSV'.
          format(frame_num, ignored_packets))
#--------------------------------------------------

def main():
    # existing_files=[]
    # for root, dirs, fnames in os.walk('output/'):
    #     for fname in fnames:
    #         if '.csv' in fname:
    #             existing_files.append(fname.split('.csv')[0])

    for root, dirs, fnames in os.walk('data/'):
        for fname in tqdm(fnames):
            # if fname.split('.pcapng')[0] in existing_files:
            #     print('{}already exists'.format(fname))
            #     continue
            if 'pcapng' in fname:
                pcap2csv(os.path.join(root,fname))
#--------------------------------------------------

if __name__ == '__main__':
    main()
