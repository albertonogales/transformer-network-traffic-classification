# Table 1 — Features of the 5 selected classes from ISCXVPN2016
# Source: ISCXVPN2016 dataset (Draper-Gil et al., 2016)
# This table contains static values from the dataset paper. No raw CSV is available.
data = [
    ("Chat",         6,  20,        153451,    164.12, 0.19),
    ("File Transfer",4,  32,      11007165,    190.88, 38.66),
    ("Streaming",    7,  30,       4991536,    294.65, 30.08),
    ("VoIP",         4,  23,       5682877,    726.23, 31.04),
    ("Email",        1,   4,         75079,    207.45,  0.03),
    ("Total",       22, 109,      21910108,   1583.33, 100.0),
]
print(f"{'Class':<15} {'Apps':>5} {'PCAPs':>6} {'Packets':>12} {'Time(min)':>10} {'%':>7}")
print("-"*60)
for row in data:
    print(f"{row[0]:<15} {row[1]:>5} {row[2]:>6} {row[3]:>12,} {row[4]:>10.2f} {row[5]:>7.2f}%")
