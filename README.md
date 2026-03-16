


# Generate dataset from PCAP files
To generate the csv dataset from the PCAP files, run the following command:

```bash
zeek -r pcaps/Wednesday-workingHours.pcap
```
Afterward convert it to a csv file using the following two commands:
```bash
echo "ts,uid,id.orig_h,id.orig_p,id.resp_h,id.resp_p,proto,service,conn_state,duration,orig_bytes,resp_bytes,orig_pkts,resp_pkts,missed_bytes" > conn.csv
zeek-cut ts uid id.orig_h id.orig_p id.resp_h id.resp_p proto service conn_state duration orig_bytes resp_bytes orig_pkts resp_pkts missed_bytes < conn.log | tr '\t' ',' >> conn.csv
```
Then all that is left is to run all cells i the Jupyter Notebook file