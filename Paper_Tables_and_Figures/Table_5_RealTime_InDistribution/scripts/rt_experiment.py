import sys, os, time, csv, math, pickle, datetime
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

if len(sys.argv) < 4:
    print("Usage: python3 rt_experiment.py <Class> <RunNumber> <WindowNumber>")
    sys.exit(1)

CLASS  = sys.argv[1]
RUN    = sys.argv[2]
WINDOW = sys.argv[3]
CSV_FILE = f"/tmp/live_experiments/{CLASS}/run{RUN}/csvs/pcap_{WINDOW}.csv"
LOG_CSV  = f"/tmp/live_experiments/results_log.csv"

with open("/tmp/best_model_weights.obj","rb") as f:
    obj = pickle.load(f)
W = obj["pesos"]

Wq=torch.tensor(W[0],dtype=torch.float32); bq=torch.tensor(W[1],dtype=torch.float32)
Wk=torch.tensor(W[2],dtype=torch.float32); bk=torch.tensor(W[3],dtype=torch.float32)
Wv=torch.tensor(W[4],dtype=torch.float32); bv=torch.tensor(W[5],dtype=torch.float32)
Wo=torch.tensor(W[6],dtype=torch.float32); bo=torch.tensor(W[7],dtype=torch.float32)
lg=torch.tensor(W[8],dtype=torch.float32); lb=torch.tensor(W[9],dtype=torch.float32)
c1w=torch.tensor(W[10],dtype=torch.float32).squeeze(0).T.unsqueeze(-1)
c1b=torch.tensor(W[11],dtype=torch.float32)
c2w=torch.tensor(W[12],dtype=torch.float32).squeeze(0).T.unsqueeze(-1)
c2b=torch.tensor(W[13],dtype=torch.float32)
d1w=torch.tensor(W[14].T,dtype=torch.float32); d1b=torch.tensor(W[15],dtype=torch.float32)
d2w=torch.tensor(W[16].T,dtype=torch.float32); d2b=torch.tensor(W[17],dtype=torch.float32)

PROTOS={"TCP","UDP","SSL","TLSv1","TLSv1.2","TLSv1.3","SSHv2","QUIC"}
LRAW_HEX=800; NHEX=4; N_TOK=200; NUM_PKT=20; MAX_T=30.0; MIN_PKTS=20

def hex_to_ints(h):
    h=h.replace(":","").replace(" ","")[:LRAW_HEX]
    vals=[]
    for i in range(0,len(h),NHEX):
        chunk=h[i:i+NHEX].ljust(NHEX,'0')
        try: vals.append(int(chunk,16))
        except: vals.append(0)
    return (vals+[0]*N_TOK)[:N_TOK]

def preprocess(csv_file):
    t0 = time.perf_counter()
    flows = defaultdict(list)
    df = pd.read_csv(csv_file, low_memory=False)
    df = df.rename(columns={
        "_ws.col.number":"_ws.col.No.", "_ws.col.cls_time":"_ws.col.Time",
        "_ws.col.def_src":"_ws.col.Source", "_ws.col.def_dst":"_ws.col.Destination",
        "_ws.col.protocol":"_ws.col.Protocol", "_ws.col.packet_length":"_ws.col.Length"})
    total_packets = len(df)
    for _,row in df.iterrows():
        proto=str(row.get("_ws.col.Protocol","")).strip()
        if proto not in PROTOS: continue
        if proto=="TCP":
            try:
                if float(row.get("tcp.len",0) or 0)<=2: continue
                pay=str(row.get("tcp.payload","") or "")
            except: continue
        else:
            pay=str(row.get("udp.payload","") or "")
        if not pay or pay=="nan": continue
        sp=str(row.get("tcp.srcport",row.get("udp.srcport","")) or "")
        dp=str(row.get("tcp.dstport",row.get("udp.dstport","")) or "")
        src=str(row.get("_ws.col.Source","") or "")
        dst=str(row.get("_ws.col.Destination","") or "")
        t=float(row.get("_ws.col.Time",0) or 0)
        flows[tuple(sorted([(src,sp),(dst,dp)]))].append((t,pay))
    semiflows=[]
    for key,pkts in flows.items():
        pkts.sort(key=lambda x:x[0])
        if len(pkts)<MIN_PKTS: continue
        i=0
        while i+NUM_PKT<=len(pkts):
            chunk=pkts[i:i+NUM_PKT]
            if chunk[-1][0]-chunk[0][0]>MAX_T: i+=1; continue
            arr=np.array([hex_to_ints(p) for _,p in chunk],dtype=np.float32)
            arr=np.concatenate([arr,np.full((20,1),65535,dtype=np.float32)],axis=1)/65535.0
            semiflows.append(arr); i+=NUM_PKT
    t_pre = time.perf_counter() - t0
    return semiflows, total_packets, len(flows), t_pre

def infer(semiflows):
    t0 = time.perf_counter()
    X=torch.tensor(np.stack(semiflows),dtype=torch.float32)
    with torch.no_grad():
        Q=torch.einsum('bti,ihd->bthd',X,Wq)+bq
        K=torch.einsum('bti,ihd->bthd',X,Wk)+bk
        V=torch.einsum('bti,ihd->bthd',X,Wv)+bv
        Q=Q.permute(0,2,1,3); K=K.permute(0,2,1,3); V=V.permute(0,2,1,3)
        attn=torch.softmax(Q@K.transpose(-2,-1)/(6**0.5),dim=-1)
        out=(attn@V).permute(0,2,1,3)
        out=torch.einsum('bthd,hdo->bto',out,Wo)+bo
        res=out+X
        mean=res.mean(-1,keepdim=True); var=res.var(-1,keepdim=True,unbiased=False)
        y=lg*(res-mean)/torch.sqrt(var+1e-6)+lb
        y=torch.relu(torch.nn.functional.conv1d(y.permute(0,2,1),c1w,c1b)).permute(0,2,1)
        y=torch.nn.functional.conv1d(y.permute(0,2,1),c2w,c2b).permute(0,2,1)
        enc=y+res; pool=enc.mean(dim=2)
        h=torch.relu(pool@d1w.T+d1b)
        probs=torch.softmax(h@d2w.T+d2b,dim=1).numpy()
    t_inf = time.perf_counter() - t0
    return probs, t_inf

CLASSES=["Chat","Email","FileTransfer","Streaming","VoIP"]

semiflows, n_packets, n_flows, t_pre = preprocess(CSV_FILE)
n_semiflows = len(semiflows)

if n_semiflows == 0:
    print(f"  [W{WINDOW}] 0 semiflows — window skipped (logged)")
    write_header = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp","class","run","window","n_packets","n_flows","n_semiflows",
                        "t_preprocess","t_inference","t_total","semiflows_per_s","packets_per_s",
                        "dominant","mean_confidence",
                        "n_Chat","n_Email","n_FileTransfer","n_Streaming","n_VoIP"])
        w.writerow([datetime.datetime.now().isoformat(), CLASS, RUN, WINDOW,
                    n_packets, n_flows, 0, round(t_pre,4), 0, round(t_pre,4),
                    0, 0, "None", 0, 0,0,0,0,0])
    sys.exit(0)

probs, t_inf = infer(semiflows)
t_total = t_pre + t_inf
preds = probs.argmax(axis=1)
counts = np.bincount(preds, minlength=5)
dominant = CLASSES[int(counts.argmax())]
mean_conf = float(probs.max(axis=1).mean())

print(f"\n  Window {WINDOW} | Class: {CLASS} | Run: {RUN}")
print(f"  Packets: {n_packets} | Flows: {n_flows} | Semiflows: {n_semiflows}")
print(f"  Preprocessing: {t_pre:.4f}s | Inference: {t_inf:.4f}s | Total: {t_total:.4f}s")
print(f"  RTF: {t_total/8:.4f} ({8/t_total:.1f}x faster than real-time)")
print(f"  Throughput: {n_semiflows/t_total:.1f} SF/s | {n_packets/t_total:.1f} pkts/s")
print(f"  Dominant: {dominant} | Confidence: {mean_conf*100:.1f}%")
print(f"  Distribution: " + " | ".join(f"{c}:{counts[i]}" for i,c in enumerate(CLASSES)))

write_header = not os.path.exists(LOG_CSV)
with open(LOG_CSV, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["timestamp","class","run","window","n_packets","n_flows","n_semiflows",
                    "t_preprocess","t_inference","t_total","semiflows_per_s","packets_per_s",
                    "dominant","mean_confidence",
                    "n_Chat","n_Email","n_FileTransfer","n_Streaming","n_VoIP"])
    w.writerow([datetime.datetime.now().isoformat(), CLASS, RUN, WINDOW,
                n_packets, n_flows, n_semiflows,
                round(t_pre,4), round(t_inf,4), round(t_total,4),
                round(n_semiflows/t_total,2), round(n_packets/t_total,2),
                dominant, round(mean_conf,4),
                int(counts[0]),int(counts[1]),int(counts[2]),int(counts[3]),int(counts[4])])
