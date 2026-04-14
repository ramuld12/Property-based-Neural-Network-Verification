
# Generate dataset from PCAP files
## CICIDS2017
To generate the tsv dataset from the PCAP files, run the following command in the zeek_logs folder:

```bash
zeek -C -r ./../pcaps/Wednesday-workingHours.pcap 
```
Afterward convert it to a tsv file using the following commands:
```bash
zeek-cut -m < conn.log >> conn.tsv
```
Then all that is left is to run all cells i the [`labelling`](data/CICIDS2017/label_cicids2017.ipynb) Jupyter Notebook file

## CICIoT2023
Since we have more direct naming from filenames, here we just need to run the script provided in [label_ciciot2023.sh](data/CICIoT2023/label_ciciot2023.sh).

Make sure to make it executable at first by running
```bash
chmod -X label_ciciot2023.sh
```
Then run it pointing to individual pcaps or all files ending with .pcap:

```bash
./label_ciciot2023.sh pcaps/file1.pcap pcaps/file2.pcap
./label_ciciot2023.sh *.pcap
```